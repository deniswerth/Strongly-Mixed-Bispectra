"""
All five tree-level bispectrum shapes of the minimal pi-sigma EFT (quadratic
mixing rho treated exactly/non-perturbatively), classified by "exchange
order" nsigma = number of sigma legs at the single cubic vertex:

    nsigma=0  pidot_c^3                        (no exchange,   coupling kapp1)
    nsigma=1  pidot_c^2 sigma                  (single, "||",  coupling lambda11)
    nsigma=1  (d_i pi_c)^2 sigma               (single, "T",   coupling lambda12)
    nsigma=2  pidot_c sigma^2                  (double,        coupling lambda2)
    nsigma=3  sigma^3                          (triple,        coupling lambda3)


-----------------------------------------------------------------------
Method summary
-----------------------------------------------------------------------
Every leg of the single cubic vertex is one of three types, each reducing
to a 1D kernel built from the SAME weight function omega_a(u):

    velocity leg   pidot_c            -> W_2^a(beta) = int omega_a(u)(1+2u)^2/(1+u) e^{-beta u} du
    undiff. leg    pi_c (gradient sq) -> P^a(beta)   = W_0^a(beta) + (beta/2) W_1^a(beta)
    sigma leg      sigma              -> V^a(beta)   = int omega_a(u) u e^{-beta u} du

with W_n^a(beta) = int omega_a(u)(1+2u)^n/(1+u) e^{-beta u} for
n=0,1,2. All four kernels (W0, W1, W2, V) are computed here from a SINGLE
cached tabulation of omega_{+1}(u) per (m_eff/H, rho/H) point (channel -1
follows by conjugation) -- this is the expensive "Level 1" step, done once
per parameter point regardless of how many of the five shapes or how many
triangles are later evaluated with it.

For every shape, the sum over the 2^3 channel assignments a_j=+-1 (one per
leg) factorizes into a PRODUCT of three independent two-term sums (one per
leg), because each leg's channel index appears only in that leg's own
factor.

    Q_K(beta) = sum_{a=+-1} exp(a*pi*lambda/2) r_a K^{-a}(beta),   K in {W2, P, V}

The outer Schwinger-parameter (xi) integral uses the measure xi^N e^{-xi}
with N=2 for four of the five shapes and N=0 for the "perp"/Lambda1 shape
(the one vertex with two undifferentiated legs), not assumed.


-----------------------------------------------------------------------
Public functions (each of the 5 shapes follows the same 4-function pattern)
-----------------------------------------------------------------------
    build_kernel_table(meff_over_H, rho_over_H, cache_dir=None)   [Level 1, shared]
    make_triangle_grid(n1, n2, x1_min)                            [helper, shared]

    shape_pidot3(k1,k2,k3, table)                       nsigma=0, Eq. 143
    shape_single_parallel(k1,k2,k3, table, Lambda2_over_H=1.0)   nsigma=1 "||", Eq. 172
    shape_single_perp(k1,k2,k3, table, Lambda1_over_H=1.0)       nsigma=1 "T",  Eq. 173
    shape_double(k1,k2,k3, table, alpha=1.0)             nsigma=2, Eq. 207
    shape_triple(k1,k2,k3, table, mu_over_H=1.0)         nsigma=3, Eq. 256

    shape_many_<name>(k1s,k2s,k3s, table, ...)           batch version of each
    cosine_with_equilateral_<name>(table, ..., grid=None)
    scan_cosine_grid_<name>(mu_eff_grid, lam_grid, ..., cache_dir=None, n_jobs=1)
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import mpmath as mp

mp.mp.dps = 20


# ============================================================================
# 1. omega_a(u), r_a, R -- mpmath, used only for
#    the Level-1 tabulation below.
# ============================================================================

def _omega_a(u, a, lam, nu_eff):
    j = mp.mpc(0, 1)
    p = j * a * lam / 2
    return (u**(p - 1) * (1 + u)**(-p) / mp.gamma(p)
            * mp.hyp2f1(mp.mpf('0.5') - nu_eff, mp.mpf('0.5') + nu_eff, 1 + 2*p, -u))


def _r_a(a, lam, nu_eff):
    j = mp.mpc(0, 1)
    num = mp.gamma(mp.mpf('0.5') + a*j*lam/2) * mp.gamma(mp.mpf('0.75') - nu_eff/2) * mp.gamma(mp.mpf('0.75') + nu_eff/2)
    den = mp.sqrt(mp.pi) * mp.gamma(mp.mpf('0.75') - nu_eff/2 + a*j*lam/2) * mp.gamma(mp.mpf('0.75') + nu_eff/2 + a*j*lam/2)
    return num / den


def _power_spectrum_ratio(lam, nu_eff):
    i_lam_half = mp.mpc(0, 1) * lam / 2
    num = mp.gamma(mp.mpf('0.75') - nu_eff/2) * mp.gamma(mp.mpf('0.75') + nu_eff/2)
    den = mp.gamma(mp.mpf('0.75') - nu_eff/2 + i_lam_half) * mp.gamma(mp.mpf('0.75') + nu_eff/2 + i_lam_half)
    return abs(num/den)**2


# ============================================================================
# 2. Fixed quadrature grids (module-level constants)
# ============================================================================

def _panel_gauss_legendre(edges, n_per_panel):
    x, w = np.polynomial.legendre.leggauss(n_per_panel)
    nodes, weights = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        nodes.append(0.5*(b - a)*x + 0.5*(a + b))
        weights.append(0.5*(b - a)*w)
    return np.concatenate(nodes), np.concatenate(weights)


# u = exp(t) grid for the leg-kernel integral (same range/resolution as
# pisigma_pidot3.py, validated there for accuracy down to k_min/k_max~1e-5).
_T_MIN, _T_MAX, _T_PANEL, _T_PER_PANEL = -30.0, 35.0, 3.0, 14
_T_EDGES = np.arange(_T_MIN, _T_MAX + _T_PANEL, _T_PANEL)
_T_NODES, _T_WEIGHTS = _panel_gauss_legendre(_T_EDGES, _T_PER_PANEL)
_U_NODES = np.exp(_T_NODES)

# xi = exp(t) grids for the outer Schwinger integral, weight xi^N e^{-xi}
# (Jacobian xi from the log substitution folded in). N=2 for four of the
# five shapes; N=0 for the "perp"/Lambda1 shape only.
_XI_T_MIN, _XI_T_MAX, _XI_T_PANEL, _XI_T_PER_PANEL = -16.0, 4.4, 1.6, 14
_XI_T_EDGES = np.arange(_XI_T_MIN, _XI_T_MAX + _XI_T_PANEL, _XI_T_PANEL)
_xi_t_nodes, _xi_t_weights = _panel_gauss_legendre(_XI_T_EDGES, _XI_T_PER_PANEL)
_XI_NODES = np.exp(_xi_t_nodes)
_XI_WEIGHTS_N2 = _xi_t_weights * _XI_NODES**3 * np.exp(-_XI_NODES)
_XI_WEIGHTS_N0 = _xi_t_weights * _XI_NODES**1 * np.exp(-_XI_NODES)


# ============================================================================
# 3. Level 1: build (and optionally cache) the leg-kernel table
# ============================================================================

def build_kernel_table(meff_over_H, rho_over_H, cache_dir=None):
    """
    The expensive, one-time-per-parameter-point step, shared by all five
    shapes: tabulate omega_{+1}(u) at a fixed grid of u values via
    mpmath's hypergeometric function. W0, W1, W2, V (and P = W0 + (beta/2)W1)
    are all derived from this ONE cached tabulation at evaluation time (see
    _Wn_plus, _V_plus below) -- no separate integration per kernel type.
    Channel a=-1 follows from a=+1 by complex conjugation.

    Parameters
    ----------
    meff_over_H, rho_over_H : float
        The two dimensionless model parameters m_eff/H and rho/H.
    cache_dir : str or Path, optional
        If given, cached to disk as an .npz file, reused on later calls.

    Returns
    -------
    dict with the cached tabulation (u nodes, omega values, two head
    corrections, r_plus, R, lam, nu_eff).
    """
    meff_over_H = float(meff_over_H)
    rho_over_H = float(rho_over_H)

    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"allshapes_meff{meff_over_H:.6f}_rho{rho_over_H:.6f}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            return dict(u=d['u'], omega=d['omega'], head_W=complex(d['head_W']),
                        head_V=complex(d['head_V']), r_plus=complex(d['r_plus']),
                        R=float(d['R']), lam=float(d['lam']), nu_eff=complex(d['nu_eff']))

    lam = rho_over_H
    nu_eff = mp.sqrt(mp.mpf('2.25') - mp.mpf(meff_over_H)**2)  # real (light) or i*mu (heavy)
    j = mp.mpc(0, 1)
    p = j * lam / 2

    omega_vals = np.empty(len(_U_NODES), dtype=complex)
    for i, u in enumerate(_U_NODES):
        omega_vals[i] = complex(_omega_a(mp.mpf(u), 1, lam, nu_eff))

    # Closed-form corrections for the truncated tail t < T_MIN (u -> 0),
    # where omega_+1(u) ~ C u^(p-1), C = 1/Gamma(p). The Wn kernels (and P,
    # built from them) all have an extra factor -> 1 at u=0, giving head_W;
    # the V kernel has an extra factor u -> 0 at u=0 (one higher power),
    # giving head_V. Same Mellin-regularization convention as
    # pisigma_pidot3.py: the u -> 0 (t -> -infinity) boundary term is
    # dropped by analytic continuation.
    C = 1 / mp.gamma(p)
    head_W = complex(C * mp.exp(p * _T_MIN) / p)
    head_V = complex(C * mp.exp((p + 1) * _T_MIN) / (p + 1))

    r_plus = complex(_r_a(1, lam, nu_eff))
    R = float(_power_spectrum_ratio(lam, nu_eff))

    table = dict(u=_U_NODES, omega=omega_vals, head_W=head_W, head_V=head_V,
                 r_plus=r_plus, R=R, lam=lam, nu_eff=complex(nu_eff))

    if cache_path is not None:
        np.savez(cache_path, **table)

    return table


# ============================================================================
# 4. Level 2: fast (pure-numpy) kernel evaluation and channel-sum
#    factorization ("Q_leg"), shared by all five shapes
# ============================================================================

def _Wn_plus(n, beta, table):
    """W_n^{+1}(beta) = int omega_+1(u) (1+2u)^n/(1+u) e^{-beta u}, n=0,1,2."""
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    flat = beta.reshape(-1)
    u = table['u']
    integrand = table['omega'] * (1 + 2*u)**n / (1 + u) * _T_WEIGHTS * u
    out = (np.exp(-np.outer(flat, u)) @ integrand) + table['head_W']
    return out.reshape(beta.shape)


def _V_plus(beta, table):
    """V_{+1}(beta) = int omega_+1(u) u e^{-beta u}."""
    beta = np.atleast_1d(np.asarray(beta, dtype=float))
    flat = beta.reshape(-1)
    u = table['u']
    integrand = table['omega'] * u * _T_WEIGHTS * u
    out = (np.exp(-np.outer(flat, u)) @ integrand) + table['head_V']
    return out.reshape(beta.shape)


def _K_plus(kernel, beta, table):
    """Dispatch to the a=+1 evaluation of one of the three leg-kernel types."""
    if kernel == 'W2':
        return _Wn_plus(2, beta, table)
    if kernel == 'V':
        return _V_plus(beta, table)
    if kernel == 'P':
        # P^a(beta) = W0^a(beta) + (beta/2) W1^a(beta).
        beta_arr = np.atleast_1d(np.asarray(beta, dtype=float))
        return _Wn_plus(0, beta_arr, table) + (beta_arr/2) * _Wn_plus(1, beta_arr, table)
    raise ValueError(f"unknown kernel type {kernel!r}")


def _Q_leg(kernel, beta, table):
    """
    Sum over the two oscillator channels a=+-1 for ONE leg at fixed beta,
    for any of the three leg-kernel types:

        Q_K(beta) = sum_{a=+-1} exp(a*pi*lambda/2) r_a K^{-a}(beta)

    Using r_{-1}=conj(r_{+1}) and K^{-1}=conj(K^{+1}) (both follow from
    omega_{-1}=conj(omega_{+1})), this needs only the a=+1 evaluation.
    This is what lets the sum over the 8 (a1,a2,a3) channel combinations in
    each shape's master formula factorize into a PRODUCT of 3 independent
    per-leg terms rather than an explicit 8-way loop -- see the module
    docstring.
    """
    boost = np.exp(np.pi * table['lam'] / 2)
    Kp = _K_plus(kernel, beta, table)
    r = table['r_plus']
    return boost * r * np.conj(Kp) + np.conj(r) * Kp / boost


# ============================================================================
# 5. The five shape functions
# ============================================================================

def shape_pidot3(k1, k2, k3, table):
    """(nsigma=0): the pidot_c^3 shape, coupling lambda2 set to 1."""
    kt = k1 + k2 + k3
    e1, e2, e3 = k1/kt, k2/kt, k3/kt
    Q1 = _Q_leg('W2', 2*_XI_NODES*e1, table)
    Q2 = _Q_leg('W2', 2*_XI_NODES*e2, table)
    Q3 = _Q_leg('W2', 2*_XI_NODES*e3, table)
    integral = np.sum(_XI_WEIGHTS_N2 * Q1 * Q2 * Q3)
    N = 3 / (32 * np.pi) * table['R']**(-1.5)
    return N * e1 * e2 * e3 * integral.real


def shape_single_parallel(k1, k2, k3, table, Lambda2_over_H=1.0):
    """(nsigma=1, "||"): the pidot_c^2 sigma shape. c indexes which
    of the three legs carries the sigma (V kernel); the other two are
    velocity legs (W2)."""
    kt = k1 + k2 + k3
    e = [k1/kt, k2/kt, k3/kt]
    total = 0j
    for c in range(3):
        prod = _Q_leg('V', 2*_XI_NODES*e[c], table)
        for jgleg in range(3):
            if jgleg != c:
                prod = prod * _Q_leg('W2', 2*_XI_NODES*e[jgleg], table)
        total += np.sum(_XI_WEIGHTS_N2 * prod)
    N2 = -1 / (16 * np.pi) * (1.0 / Lambda2_over_H) / (table['lam'] * table['R']**1.5)
    return N2 * e[0] * e[1] * e[2] * total.real


def shape_single_perp(k1, k2, k3, table, Lambda1_over_H=1.0):
    """(nsigma=1, "perp"): the (d_i pidot_c)^2 sigma / a^2 shape. c
    indexes which leg carries the sigma (V kernel); the other two are
    undifferentiated legs (P kernel). Uses the N=0 (not N=2) xi-measure.
    See the module-level caveat about this channel's overall sign."""
    kt = k1 + k2 + k3
    e = [k1/kt, k2/kt, k3/kt]
    total = 0j
    for c in range(3):
        a_e, b_e = e[(c+1) % 3], e[(c+2) % 3]
        Fc = e[c] * (a_e**2 + b_e**2 - e[c]**2) / (2 * a_e * b_e)
        prod = _Q_leg('V', 2*_XI_NODES*e[c], table)
        for jgleg in range(3):
            if jgleg != c:
                prod = prod * _Q_leg('P', 2*_XI_NODES*e[jgleg], table)
        total += Fc * np.sum(_XI_WEIGHTS_N0 * prod)
    N1 = +1 / (16 * np.pi) * (1.0 / Lambda1_over_H) / (table['lam'] * table['R']**1.5)
    return N1 * total.real


def shape_double(k1, k2, k3, table, alpha=1.0):
    """(nsigma=2): the pidot_c sigma^2 shape. c indexes which of the
    three legs carries the velocity (W2 kernel); the other two are sigma
    legs (V kernel)."""
    kt = k1 + k2 + k3
    e = [k1/kt, k2/kt, k3/kt]
    total = 0j
    for c in range(3):
        prod = _Q_leg('W2', 2*_XI_NODES*e[c], table)
        for jgleg in range(3):
            if jgleg != c:
                prod = prod * _Q_leg('V', 2*_XI_NODES*e[jgleg], table)
        total += np.sum(_XI_WEIGHTS_N2 * prod)
    Nalpha = alpha / (4 * np.pi * table['lam']**2 * table['R']**1.5)
    return Nalpha * e[0] * e[1] * e[2] * total.real


def shape_triple(k1, k2, k3, table, mu_over_H=1.0):
    """(nsigma=3): the sigma^3 shape. All three legs carry sigma (V
    kernel); no sum over placements (the vertex is fully symmetric)."""
    kt = k1 + k2 + k3
    e = [k1/kt, k2/kt, k3/kt]
    Q1 = _Q_leg('V', 2*_XI_NODES*e[0], table)
    Q2 = _Q_leg('V', 2*_XI_NODES*e[1], table)
    Q3 = _Q_leg('V', 2*_XI_NODES*e[2], table)
    total = np.sum(_XI_WEIGHTS_N2 * Q1 * Q2 * Q3)
    Nmu = -6 * mu_over_H / (np.pi * table['lam']**3 * table['R']**1.5)
    return Nmu * e[0] * e[1] * e[2] * total.real


_SHAPE_FUNCS = dict(pidot3=shape_pidot3, single_parallel=shape_single_parallel,
                     single_perp=shape_single_perp, double=shape_double, triple=shape_triple)


# ============================================================================
# 6. Batch evaluation over many triangles, and the standard triangle grid
# ============================================================================

def make_triangle_grid(n1=60, n2=60, x1_min=1e-3):
    """Standard normalized triangle grid, k3 fixed to 1, x1=k1/k3 log-spaced,
    x2=k2/k3 spanning the allowed triangle region for each x1. Shared by all
    five shapes. Returns k1s, k2s, k3s arrays (k3s is all ones)."""
    x1_vals = np.geomspace(x1_min, 1.0, n1)
    k1s, k2s, k3s = [], [], []
    for x1 in x1_vals:
        lo = max(x1, 1 - x1)
        for x2 in np.linspace(lo, 1.0, n2):
            k1s.append(x1)
            k2s.append(x2)
            k3s.append(1.0)
    return np.array(k1s), np.array(k2s), np.array(k3s)


def _shape_many(shape_name, k1s, k2s, k3s, table, **kwargs):
    f = _SHAPE_FUNCS[shape_name]
    k1s = np.asarray(k1s, dtype=float)
    k2s = np.asarray(k2s, dtype=float)
    k3s = np.asarray(k3s, dtype=float)
    return np.array([f(a, b, c, table, **kwargs) for a, b, c in zip(k1s, k2s, k3s)])


def shape_many_pidot3(k1s, k2s, k3s, table):
    return _shape_many('pidot3', k1s, k2s, k3s, table)


def shape_many_single_parallel(k1s, k2s, k3s, table, Lambda2_over_H=1.0):
    return _shape_many('single_parallel', k1s, k2s, k3s, table, Lambda2_over_H=Lambda2_over_H)


def shape_many_single_perp(k1s, k2s, k3s, table, Lambda1_over_H=1.0):
    return _shape_many('single_perp', k1s, k2s, k3s, table, Lambda1_over_H=Lambda1_over_H)


def shape_many_double(k1s, k2s, k3s, table, alpha=1.0):
    return _shape_many('double', k1s, k2s, k3s, table, alpha=alpha)


def shape_many_triple(k1s, k2s, k3s, table, mu_over_H=1.0):
    return _shape_many('triple', k1s, k2s, k3s, table, mu_over_H=mu_over_H)


# ============================================================================
# 7. Scan over the (m_eff/H, rho/H) parameter grid -- for a Planck analysis
# ============================================================================

def _scan_one_point(args):
    shape_name, meff, rho, k1s, k2s, k3s, cache_dir, kwargs = args
    table = build_kernel_table(meff, rho, cache_dir=cache_dir)
    return _shape_many(shape_name, k1s, k2s, k3s, table, **kwargs)


def _scan_parameter_grid(shape_name, meff_grid, rho_grid, k1s, k2s, k3s,
                          cache_dir=None, verbose=True, n_jobs=1, **kwargs):
    meff_grid = np.asarray(meff_grid, dtype=float)
    rho_grid = np.asarray(rho_grid, dtype=float)
    n_tri = len(k1s)

    points = [(meff, rho) for meff in meff_grid for rho in rho_grid]
    tasks = [(shape_name, meff, rho, k1s, k2s, k3s, cache_dir, kwargs) for meff, rho in points]

    result = np.empty((len(meff_grid), len(rho_grid), n_tri))

    if n_jobs == 1:
        for count, ((meff, rho), task) in enumerate(zip(points, tasks)):
            i, k = count // len(rho_grid), count % len(rho_grid)
            result[i, k, :] = _scan_one_point(task)
            if verbose and (k == len(rho_grid) - 1):
                print(f"scan_parameter_grid[{shape_name}]: finished m_eff/H = {meff:.4g} "
                      f"({i+1}/{len(meff_grid)})")
    else:
        from concurrent.futures import ProcessPoolExecutor
        import os
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, shapes in enumerate(pool.map(_scan_one_point, tasks)):
                i, k = count // len(rho_grid), count % len(rho_grid)
                result[i, k, :] = shapes
                if verbose and (k == len(rho_grid) - 1):
                    print(f"scan_parameter_grid[{shape_name}]: finished m_eff/H = "
                          f"{meff_grid[i]:.4g} ({i+1}/{len(meff_grid)})")
    return result


def scan_parameter_grid_pidot3(meff_grid, rho_grid, k1s, k2s, k3s, cache_dir=None, verbose=True, n_jobs=1):
    return _scan_parameter_grid('pidot3', meff_grid, rho_grid, k1s, k2s, k3s, cache_dir, verbose, n_jobs)


def scan_parameter_grid_single_parallel(meff_grid, rho_grid, k1s, k2s, k3s, cache_dir=None,
                                         verbose=True, n_jobs=1, Lambda2_over_H=1.0):
    return _scan_parameter_grid('single_parallel', meff_grid, rho_grid, k1s, k2s, k3s,
                                 cache_dir, verbose, n_jobs, Lambda2_over_H=Lambda2_over_H)


def scan_parameter_grid_single_perp(meff_grid, rho_grid, k1s, k2s, k3s, cache_dir=None,
                                     verbose=True, n_jobs=1, Lambda1_over_H=1.0):
    return _scan_parameter_grid('single_perp', meff_grid, rho_grid, k1s, k2s, k3s,
                                 cache_dir, verbose, n_jobs, Lambda1_over_H=Lambda1_over_H)


def scan_parameter_grid_double(meff_grid, rho_grid, k1s, k2s, k3s, cache_dir=None,
                                verbose=True, n_jobs=1, alpha=1.0):
    return _scan_parameter_grid('double', meff_grid, rho_grid, k1s, k2s, k3s,
                                 cache_dir, verbose, n_jobs, alpha=alpha)


def scan_parameter_grid_triple(meff_grid, rho_grid, k1s, k2s, k3s, cache_dir=None,
                                verbose=True, n_jobs=1, mu_over_H=1.0):
    return _scan_parameter_grid('triple', meff_grid, rho_grid, k1s, k2s, k3s,
                                 cache_dir, verbose, n_jobs, mu_over_H=mu_over_H)


# ============================================================================
# 8. Cosine correlation with the equilateral template, for each shape
# ============================================================================

def _correlation_grid(n1=50, n2=20, x1_min=1e-3):
    """Weighted grid over the triangle domain approximating the uniform
    measure dx1*dx2, for the correlation integral. Same construction as
    pisigma_pidot3.py. Returns (x1, x2, weight) flat arrays."""
    x1 = np.geomspace(x1_min, 1.0, n1)
    w1 = np.gradient(x1)
    x1_all, x2_all, w_all = [], [], []
    for x1i, w1i in zip(x1, w1):
        lo = max(x1i, 1 - x1i)
        x2 = np.linspace(lo, 1.0, n2)
        w2 = np.full(n2, (1.0 - lo) / (n2 - 1))
        w2[0] *= 0.5
        w2[-1] *= 0.5
        x1_all.append(np.full(n2, x1i))
        x2_all.append(x2)
        w_all.append(w1i * w2)
    return np.concatenate(x1_all), np.concatenate(x2_all), np.concatenate(w_all)


_CORR_GRID = _correlation_grid()


def _cosine_with_equilateral(shape_name, table, grid=None, **kwargs):
    x1, x2, w = _CORR_GRID if grid is None else grid
    k1, k2, k3 = x1, x2, np.ones_like(x1)
    S = _shape_many(shape_name, k1, k2, k3, table, **kwargs)
    kt = k1 + k2 + k3
    S_eq = (k1/kt) * (k2/kt) * (k3/kt)
    numerator = np.sum(w * S * S_eq)
    denominator = np.sqrt(np.sum(w * S**2) * np.sum(w * S_eq**2))
    return numerator / denominator


def cosine_with_equilateral_pidot3(table, grid=None):
    return _cosine_with_equilateral('pidot3', table, grid)


def cosine_with_equilateral_single_parallel(table, grid=None, Lambda2_over_H=1.0):
    return _cosine_with_equilateral('single_parallel', table, grid, Lambda2_over_H=Lambda2_over_H)


def cosine_with_equilateral_single_perp(table, grid=None, Lambda1_over_H=1.0):
    return _cosine_with_equilateral('single_perp', table, grid, Lambda1_over_H=Lambda1_over_H)


def cosine_with_equilateral_double(table, grid=None, alpha=1.0):
    return _cosine_with_equilateral('double', table, grid, alpha=alpha)


def cosine_with_equilateral_triple(table, grid=None, mu_over_H=1.0):
    return _cosine_with_equilateral('triple', table, grid, mu_over_H=mu_over_H)


def _scan_one_cosine_point(args):
    shape_name, mu_eff, lam, cache_dir, grid, kwargs = args
    meff_over_H = np.sqrt(2.25 + mu_eff**2)
    table = build_kernel_table(meff_over_H, lam, cache_dir=cache_dir)
    return _cosine_with_equilateral(shape_name, table, grid, **kwargs)


def _scan_cosine_grid(shape_name, mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                       cache_dir=None, verbose=True, n_jobs=1, **kwargs):
    mu_eff_grid = np.asarray(mu_eff_grid, dtype=float)
    lam_grid = np.asarray(lam_grid, dtype=float)
    grid = _correlation_grid(n1, n2, x1_min)

    points = [(mu, lam) for mu in mu_eff_grid for lam in lam_grid]
    tasks = [(shape_name, mu, lam, cache_dir, grid, kwargs) for mu, lam in points]

    result = np.empty((len(mu_eff_grid), len(lam_grid)))

    if n_jobs == 1:
        for count, (mu, lam) in enumerate(points):
            i, jx = count // len(lam_grid), count % len(lam_grid)
            result[i, jx] = _scan_one_cosine_point(tasks[count])
            if verbose and (jx == len(lam_grid) - 1):
                print(f"scan_cosine_grid[{shape_name}]: finished mu_eff = {mu:.4g} "
                      f"({i+1}/{len(mu_eff_grid)})")
    else:
        from concurrent.futures import ProcessPoolExecutor
        import os
        workers = os.cpu_count() if n_jobs == -1 else n_jobs
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count, cos_val in enumerate(pool.map(_scan_one_cosine_point, tasks)):
                i, jx = count // len(lam_grid), count % len(lam_grid)
                result[i, jx] = cos_val
                if verbose and (jx == len(lam_grid) - 1):
                    print(f"scan_cosine_grid[{shape_name}]: finished mu_eff = "
                          f"{mu_eff_grid[i]:.4g} ({i+1}/{len(mu_eff_grid)})")
    return result


def scan_cosine_grid_pidot3(mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                             cache_dir=None, verbose=True, n_jobs=1):
    return _scan_cosine_grid('pidot3', mu_eff_grid, lam_grid, n1, n2, x1_min, cache_dir, verbose, n_jobs)


def scan_cosine_grid_single_parallel(mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                                      cache_dir=None, verbose=True, n_jobs=1, Lambda2_over_H=1.0):
    return _scan_cosine_grid('single_parallel', mu_eff_grid, lam_grid, n1, n2, x1_min,
                              cache_dir, verbose, n_jobs, Lambda2_over_H=Lambda2_over_H)


def scan_cosine_grid_single_perp(mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                                  cache_dir=None, verbose=True, n_jobs=1, Lambda1_over_H=1.0):
    return _scan_cosine_grid('single_perp', mu_eff_grid, lam_grid, n1, n2, x1_min,
                              cache_dir, verbose, n_jobs, Lambda1_over_H=Lambda1_over_H)


def scan_cosine_grid_double(mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                             cache_dir=None, verbose=True, n_jobs=1, alpha=1.0):
    return _scan_cosine_grid('double', mu_eff_grid, lam_grid, n1, n2, x1_min,
                              cache_dir, verbose, n_jobs, alpha=alpha)


def scan_cosine_grid_triple(mu_eff_grid, lam_grid, n1=50, n2=20, x1_min=1e-3,
                             cache_dir=None, verbose=True, n_jobs=1, mu_over_H=1.0):
    return _scan_cosine_grid('triple', mu_eff_grid, lam_grid, n1, n2, x1_min,
                              cache_dir, verbose, n_jobs, mu_over_H=mu_over_H)
