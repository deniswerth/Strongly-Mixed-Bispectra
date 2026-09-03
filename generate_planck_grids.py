"""
Generates theoretical bispectrum-shape data for a Planck full-shape analysis:
a 100x100 grid in (rho/H, mu_eff), with the FULL shape S(k1,k2,k3) -- not just
summary statistics -- evaluated on a fixed triangle grid at every parameter
point, for each of the 5 pi-sigma cubic interactions.

Format: one output .npz file per interaction, using the SAME (lam, mu) grid
convention as the existing gridmaps_100x100.npz files (axis 0 = lam =
rho/H, axis 1 = mu = mu_eff, both linspace(0.05, 6.0, 50)), so a colleague's
existing loading code for those files needs minimal changes. Each file also
keeps the 'cos' (cosine with equilateral) and 'seq' (equilateral value)
summary fields for backward compatibility, and adds:
    k1, k2, k3   : (n_tri,)              the triangle grid (shared across all
                                          parameter points -- see make_triangle_grid)
    shape        : (n_lam, n_mu, n_tri)  the full shape at every grid point
    interaction  : str                   which of the 5 interactions this file holds

Resumable: after every completed lam-row, progress is checkpointed to disk.
If interrupted, re-running with the same output path picks up where it left
off (already-finite rows are skipped) -- same idea as the worker checkpoint
files this format is modeled on, but single-file and row-granular rather
than multi-worker, since that is enough for a run of this size.


Usage
-----
    python generate_planck_grids.py                                    # full 50x50, defaults
    python generate_planck_grids.py --n_grid 5 --n_tri1 8 --n_tri2 6   # quick smoke test
    python generate_planck_grids.py --n_jobs -1                        # use all CPU cores (default n_grid=50, n_tri1=n_tri2=30, this is very expensive)
    python generate_planck_grids.py --out_dir data/ --interactions pidot3 double
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import FastShapes as ps

# ----------------------------------------------------------------------
# The 5 interactions, and their (single-triangle, batch) shape functions.
# ----------------------------------------------------------------------
INTERACTIONS = {
    'pidot3':          (ps.shape_pidot3,          ps.shape_many_pidot3),
    'single_parallel': (ps.shape_single_parallel, ps.shape_many_single_parallel),
    'single_perp':     (ps.shape_single_perp,     ps.shape_many_single_perp),
    'double':          (ps.shape_double,          ps.shape_many_double),
    'triple':          (ps.shape_triple,          ps.shape_many_triple),
}


def compute_row(lam, mu_grid, k1s, k2s, k3s, cache_dir):
    """All 5 interactions, at fixed lam, for every mu in mu_grid. Returns a
    dict of {interaction: (shape[n_mu, n_tri], cos[n_mu], seq[n_mu])}.
    The kernel table (the expensive step) is built once per (lam, mu) point
    and shared across all 5 interactions -- not rebuilt per interaction."""
    n_mu = len(mu_grid)
    n_tri = len(k1s)
    out = {name: (np.empty((n_mu, n_tri)), np.empty(n_mu), np.empty(n_mu))
           for name in INTERACTIONS}

    for j, mu_eff in enumerate(mu_grid):
        meff_over_H = np.sqrt(2.25 + mu_eff**2)   # heavy field: nu_eff = i*mu_eff
        table = ps.build_kernel_table(meff_over_H, lam, cache_dir=cache_dir)
        for name, (shape_fn, shape_many_fn) in INTERACTIONS.items():
            S = shape_many_fn(k1s, k2s, k3s, table)
            S_eq = shape_fn(1.0, 1.0, 1.0, table)
            out[name][0][j, :] = S
            out[name][1][j] = _cosine_from_shape(S, k1s, k2s, k3s)
            out[name][2][j] = S_eq
    return out


def _cosine_from_shape(S, k1s, k2s, k3s):
    """cos(S, S_eq) computed directly from an already-evaluated shape array
    on the (unweighted) triangle grid -- a simple dot-product estimate,
    consistent with (but cheaper than) pisigma_allshapes.py's dedicated
    cosine_with_equilateral_* functions, which use a separately-weighted
    grid. Good enough for the quick-reference field stored alongside the
    full shape; use the dedicated functions for a careful correlation
    calculation."""
    kt = k1s + k2s + k3s
    S_eq_shape = (k1s/kt) * (k2s/kt) * (k3s/kt)
    num = np.sum(S * S_eq_shape)
    den = np.sqrt(np.sum(S*S) * np.sum(S_eq_shape*S_eq_shape))
    return num / den


def _row_worker(args):
    lam, mu_grid, k1s, k2s, k3s, cache_dir = args
    return compute_row(lam, mu_grid, k1s, k2s, k3s, cache_dir)


def generate(n_grid=50, n_tri1=30, n_tri2=30, x1_min=1e-3,
             lam_range=(0.05, 4.0), mu_range=(0.05, 4.0),
             out_dir='.', interactions=None, cache_dir='kernel_cache',
             n_jobs=6, checkpoint_every=1):
    """
    Compute the full-shape grids and write one .npz file per interaction.

    Parameters
    ----------
    n_grid : int
        Resolution of the (lam, mu) parameter grid (n_grid x n_grid).
    n_tri1, n_tri2, x1_min : int, int, float
        Passed to make_triangle_grid -- resolution of the triangle grid the
        shape is evaluated on. n_tri1*n_tri2 is the number of triangles.
    lam_range, mu_range : (float, float)
        Range of rho/H and mu_eff to scan.
    out_dir : str
        Directory for the output .npz files (one per interaction) and the
        kernel cache.
    interactions : list of str, optional
        Which interactions to compute (default: all 5). Names must be keys
        of INTERACTIONS above.
    cache_dir : str
        Where build_kernel_table caches its (expensive) per-point kernel
        tabulation -- shared across interactions AND across a resumed run.
    n_jobs : int
        Number of worker processes (grid ROWS, i.e. fixed lam, are the
        parallel unit here since a row already shares kernel tables
        internally across mu and interactions). n_jobs=-1 uses all cores.
    checkpoint_every : int
        Save progress to disk every this many completed rows.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / cache_dir

    lam_grid = np.round(np.linspace(*lam_range, n_grid), 6)
    mu_grid = np.round(np.linspace(*mu_range, n_grid), 6)
    k1s, k2s, k3s = ps.make_triangle_grid(n1=n_tri1, n2=n_tri2, x1_min=x1_min)
    n_tri = len(k1s)

    names = interactions if interactions is not None else list(INTERACTIONS)
    for name in names:
        if name not in INTERACTIONS:
            raise ValueError(f"unknown interaction {name!r}; choose from {list(INTERACTIONS)}")

    # ---- load or initialize checkpoints (one per interaction) ----
    paths = {name: out_dir / f"shapes_{n_grid}x{n_grid}_{name}.npz" for name in names}
    state = {}
    for name in names:
        if paths[name].exists():
            d = np.load(paths[name])
            state[name] = dict(shape=d['shape'].copy(), cos=d['cos'].copy(), seq=d['seq'].copy())
            print(f"[{name}] resuming from existing checkpoint "
                  f"({np.isfinite(d['seq']).sum()}/{n_grid*n_grid} points already done)")
        else:
            state[name] = dict(shape=np.full((n_grid, n_grid, n_tri), np.nan),
                                cos=np.full((n_grid, n_grid), np.nan),
                                seq=np.full((n_grid, n_grid), np.nan))

    def save(name):
        np.savez(paths[name], lam=lam_grid, mu=mu_grid, k1=k1s, k2=k2s, k3=k3s,
                  shape=state[name]['shape'], cos=state[name]['cos'], seq=state[name]['seq'],
                  interaction=name)

    # ---- rows still needing work: a row is "done" if its lam-slice of ANY
    #      interaction's seq is all-finite (they're always computed together) ----
    todo_rows = [i for i in range(n_grid) if not np.all(np.isfinite(state[names[0]]['seq'][i]))]
    print(f"{len(todo_rows)}/{n_grid} rows to compute, {n_tri} triangles/point, "
          f"{len(names)} interaction(s)")

    t_start = time.time()

    def handle_row(i, row_out):
        for name in names:
            state[name]['shape'][i] = row_out[name][0]
            state[name]['cos'][i] = row_out[name][1]
            state[name]['seq'][i] = row_out[name][2]

    if n_jobs == 1:
        for count, i in enumerate(todo_rows):
            row_out = compute_row(lam_grid[i], mu_grid, k1s, k2s, k3s, cache_path)
            handle_row(i, row_out)
            if (count + 1) % checkpoint_every == 0 or i == todo_rows[-1]:
                for name in names:
                    save(name)
            elapsed = time.time() - t_start
            print(f"row {count+1}/{len(todo_rows)} (lam={lam_grid[i]:.3f}) done, "
                  f"{elapsed:.0f}s elapsed, ~{elapsed/(count+1)*(len(todo_rows)-count-1):.0f}s remaining")
    else:
        from concurrent.futures import ProcessPoolExecutor
        import os
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        tasks = [(lam_grid[i], mu_grid, k1s, k2s, k3s, cache_path) for i in todo_rows]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, (i, row_out) in enumerate(zip(todo_rows, pool.map(_row_worker, tasks))):
                handle_row(i, row_out)
                if (count + 1) % checkpoint_every == 0 or count == len(todo_rows) - 1:
                    for name in names:
                        save(name)
                elapsed = time.time() - t_start
                print(f"row {count+1}/{len(todo_rows)} (lam={lam_grid[i]:.3f}) done, "
                      f"{elapsed:.0f}s elapsed")

    print(f"Done. Wrote: {[str(paths[n]) for n in names]}")
    return {name: paths[name] for name in names}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n_grid', type=int, default=50, help='(lam, mu) grid resolution')
    parser.add_argument('--n_tri1', type=int, default=30, help='triangle grid: x1 resolution')
    parser.add_argument('--n_tri2', type=int, default=30, help='triangle grid: x2 resolution')
    parser.add_argument('--out_dir', type=str, default='.', help='output directory')
    parser.add_argument('--interactions', nargs='+', default=None,
                         help=f'subset of {list(INTERACTIONS)} (default: all)')
    parser.add_argument('--n_jobs', type=int, default=1, help='parallel worker processes (-1 = all cores)')
    parser.add_argument('--checkpoint_every', type=int, default=1, help='save progress every N rows')
    args = parser.parse_args()

    generate(n_grid=args.n_grid, n_tri1=args.n_tri1, n_tri2=args.n_tri2,
              out_dir=args.out_dir, interactions=args.interactions,
              n_jobs=args.n_jobs, checkpoint_every=args.checkpoint_every)
