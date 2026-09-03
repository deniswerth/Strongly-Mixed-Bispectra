[![Python](https://img.shields.io/badge/python-3.8.2-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://choosealicense.com/licenses/mit/)
[![ArXiv](https://img.shields.io/badge/arXiv-2302...-yellowgreen.svg)](https://google.com)

# Strongly-mixed bispectra in the π-σ model

Code accompanying *"Full-shape constraints of strongly-mixed bispectra with Planck data"* (Philcox, Pinol, Roest & Werth, in prep.) [[arXiv:XXXX.XXXXX]](https://arxiv.org/abs/XXXX.XXXXX).

This repository computes the tree-level bispectrum of every cubic self- and cross-interaction of the curvature perturbation coupled to a massive isocurvature field, with the quadratic mixing ρ treated **exactly** (non-perturbatively), following the exact mode functions and Schwinger-parameter reduction. It also contains the scripts used to produce the public data release of pre-computed shapes.

---

## What's here

| File | What it does |
|---|---|
| `FastShapes.py` | **Core library.** Exact, semi-analytical bispectrum shapes for all 5 cubic interactions, and their cosine correlation with the equilateral template, as a function of `(m_eff/H, ρ/H)`. This is the main deliverable of the paper's method — see "Method" below. |
| `FastShapes.ipynb` | Tutorial notebook: reproduces the letter's Figure 1, computes full 3D shapes, cross-checks against `CosmoFlow`, and scans the cosine correlation across parameter space. Start here. |
| `Solver.py`, `Theory.py`, `Parameters.py` | `CosmoFlow`: an independent numerical solver that integrates the same in-in problem as a system of ODEs in time, without assuming any hierarchy between ρ and H. Used only as a cross-check — orders of magnitude slower than `FastShapes.py`, and not intended for large parameter scans. |
| `generate_planck_grids.py` | Production script: computes the full shape (and cosine) on a `(μ_eff, lambda)` grid, for all 5 interactions, and writes the public data release. Resumable/checkpointed. |
| `read_planck_grids.py` | Loads the `.npz` files produced above and makes the standard diagnostic plots (cosine map, single full 3D shape). |

### The five interactions

The minimal EFT of an inflationary Goldstone $\pi$ coupled to a massive isocurvature field $\sigma$ through a quadratic mixing ρ has exactly five independent cubic self- and cross-interactions (at unit sound speed):

| Interaction | Coupling | `FastShapes.py` function |
|---|---|---|
| $\dot\pi_c^3$ | $\kappa_1$ | `shape_pidot3` |
| $\dot\pi_c^2\sigma$ | $\lambda_{11}$ | `shape_single_parallel` |
| $(\partial_i\pi_c)^2\sigma/a^2$ | $\lambda_{12}$ | `shape_single_perp` |
| $\dot\pi_c\sigma^2$ | $\lambda_2$ | `shape_double` |
| $\sigma^3$ | $\lambda_3$ | `shape_triple` |

This is the convention used consistently by `Theory.py` (the `CosmoFlow` Lagrangian) and by `FastShapes.py`'s own docstring, and is what `FastShapes.ipynb` cross-checks one against the other.

---

## Method (brief)

Because the quadratic mixing is resummed exactly rather than treated as a small perturbation, every one of the five interactions above reduces, at tree level, to a **single-vertex (contact) diagram** — the number of σ legs it carries is simply a bookkeeping label, not an internal exchange line. Each such diagram reduces further to a one-dimensional Schwinger-parameter integral over a product of leg kernels, all built from one weight function $\omega_a(u)$ tabulated once per $(\mu_{\rm eff},\lambda)$ point. This is what makes `FastShapes.py` many orders of magnitude faster than solving the coupled in-in problem numerically (`CosmoFlow`), and is what makes a full parameter-space scan for a Planck analysis computationally tractable at all. See the paper for the full derivation; `FastShapes.py`'s module docstring gives a self-contained summary of the numerical scheme (kernel tabulation, the log-substitution used to handle a marginal boundary singularity, and the channel-sum factorisation).

---

## Installation

```bash
pip install numpy scipy mpmath matplotlib joblib
```

No package-level installation is needed — the scripts are meant to be used directly from a clone of this repository (`FastShapes.py` etc. are imported by file name, not as an installed package).

Python ≥ 3.9 recommended. `joblib` is only needed for the `CosmoFlow` cross-check cells in the notebook (`FastShapes.py`'s own parallelism uses the standard-library `concurrent.futures` and has no extra dependency).

---

## Quickstart

### 1. A single shape, at a single triangle

```python
from FastShapes import build_kernel_table, shape_pidot3

# table caches everything that depends on (m_eff/H, rho/H) -- build once, reuse for
# any number of triangles and any of the 5 interactions.
table = build_kernel_table(meff_over_H=3.5, rho_over_H=3.16)

S = shape_pidot3(k1=1.0, k2=0.8, k3=0.6, table=table)
```

### 2. The full shape over a triangle grid

```python
from FastShapes import build_kernel_table, make_triangle_grid, shape_many_double

table = build_kernel_table(meff_over_H=3.5, rho_over_H=3.16)
k1s, k2s, k3s = make_triangle_grid(n1=60, n2=60)   # n1*n2 triangles, log-spaced in the squeezed direction
S_grid = shape_many_double(k1s, k2s, k3s, table)
```

### 3. Cosine correlation with the equilateral template, across parameter space

```python
from FastShapes import scan_cosine_grid_triple
import numpy as np

mu_eff_grid = np.linspace(0.5, 3.0, 20)
lam_grid = np.linspace(0.5, 3.0, 20)
cos_grid = scan_cosine_grid_triple(mu_eff_grid, lam_grid, cache_dir='kernel_cache', n_jobs=6)
```

`cache_dir` matters here: it caches the expensive per-point kernel tabulation to disk, so repeated or resumed scans don't redo work already done — always set it for anything beyond a quick test.

For all three, see `FastShapes.ipynb` for the exact cell-by-cell workflow, including the equivalent calls for the other four interactions and the accompanying plots.

---

## Reproducing the public data release

The paper's full-shape Planck analysis uses a pre-computed grid: `50×50` in `(ρ/H, μ_eff)`, with the full shape evaluated on a `100×100` triangle grid at every point, for all 5 interactions. To regenerate it:

```bash
python generate_planck_grids.py --n_jobs -1
```

**This is expensive** — at the default `100×100` triangle resolution this is a multi-hour run even with full parallelisation (`--n_jobs -1` uses all available cores); budget accordingly. For a quick smoke test first:

```bash
python generate_planck_grids.py --n_grid 5 --n_tri1 8 --n_tri2 6
```

The run is checkpointed after every completed row of the parameter grid, so an interrupted run resumes automatically (already-finite rows are skipped) — just re-run the same command. Output is one `shapes_{n_grid}x{n_grid}_{interaction}.npz` file per interaction; see `generate_planck_grids.py`'s module docstring for the exact array format (axes, units, what `cos`/`seq` mean).

The pre-computed grids themselves are hosted at **\[data release URL — Zenodo / GitHub release, TBD\]** rather than committed to this repository directly (they are large).

To load and inspect them:

```bash
python read_planck_grids.py shapes_50x50_pidot3.npz --mu_eff 3.0 --rho_over_H 3.16
```

or from Python, via `read_planck_grids.load`, `plot_cosine_map`, `plot_full_shape` (see that file's docstring).

