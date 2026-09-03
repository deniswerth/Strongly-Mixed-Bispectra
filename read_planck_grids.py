"""
Load the .npz files produced by generate_planck_grids.py and make some
standard diagnostic plots: the cos(S,Seq) map, and the full 3D shape at one chosen
(mu_eff, rho/H) grid point.

Usage
-----
    python read_planck_grids.py pisigma_shapes_50x50_pidot3.npz

or, from another script / notebook:

    from read_planck_grids import load, plot_cosine_map, plot_full_shape
    data = load("shapes_50x50_pidot3.npz")
    plot_cosine_map(data)
    plot_full_shape(data, mu_eff=3.0, rho_over_H=3.16)
"""

import argparse

import numpy as np
import matplotlib.pyplot as plt


def load(path):
    """Load one interaction's file into a plain dict (lam, mu, k1, k2, k3,
    shape, cos, seq, interaction). shape has axes [lam_index, mu_index,
    triangle_index]."""
    d = np.load(path)
    return {k: d[k] for k in d.keys()}


def nearest_index(grid, value):
    """Index of the grid point closest to value -- for picking a (lam, mu)
    point by physical value rather than by raw index."""
    return int(np.argmin(np.abs(grid - value)))


def plot_cosine_map(data, ax=None):
    """The cos(S, S_eq) map over the (mu_eff, rho/H) plane, in the same
    style used throughout this project (RdBu, +-1, m^2=0 dotted line)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    lam, mu, cos = data['lam'], data['mu'], data['cos']
    im = ax.pcolormesh(mu, lam, cos, cmap='RdBu_r', vmin=-1, vmax=1, shading='auto')
    m_vals = np.linspace(mu.min(), mu.max(), 200)
    ax.plot(m_vals, np.sqrt(2.25 + m_vals**2), 'k:', label=r'$m^2=0$')
    ax.set_xlim(data["mu"][0], data["lam"][-1])
    ax.set_ylim(data["lam"][0], data["lam"][-1])
    ax.set_xlabel(r'$\mu_{\rm eff}$')
    ax.set_ylabel(r'$\rho/H$')
    ax.set_title(str(data.get('interaction', '')))
    plt.colorbar(im, ax=ax, label=r'$\cos(S,S_{\rm eq})$')
    ax.legend(frameon=False)
    return ax


def plot_full_shape(data, mu_eff, rho_over_H, ax=None, normalize=True):
    """The full 3D shape S(k1,k2,k3) at the (mu_eff, rho/H) grid point
    nearest the requested values (the grid is discrete; see the printed
    message for the point actually used)."""
    lam, mu = data['lam'], data['mu']
    i_lam = nearest_index(lam, rho_over_H)
    i_mu = nearest_index(mu, mu_eff)
    print(f"using nearest grid point: rho/H={lam[i_lam]:.4f}, mu_eff={mu[i_mu]:.4f} "
          f"(requested {rho_over_H:.4f}, {mu_eff:.4f})")

    S = data['shape'][i_lam, i_mu, :]
    k1, k2, k3 = data['k1'], data['k2'], data['k3']
    if normalize:
        S = S / data['seq'][i_lam, i_mu]

    x1, x2 = k1/k3, k2/k3
    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(x2, x1, S, cmap='coolwarm', linewidth=0.1, antialiased=True)
    ax.set_xlabel(r'$x_2=k_2/k_3$')
    ax.set_ylabel(r'$x_1=k_1/k_3$')
    ax.set_zlabel(r'$S/S_{\rm eq}$' if normalize else r'$S$')
    ax.set_title(f"{data.get('interaction','')}: "
                 rf"$\mu_{{\rm eff}}={mu[i_mu]:.2f}$, $\rho/H={lam[i_lam]:.2f}$")
    ax.view_init(elev=25, azim=40)
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', help='.npz file from generate_planck_grids.py')
    parser.add_argument('--mu_eff', type=float, default=3.0, help='mu_eff for the full-shape plot')
    parser.add_argument('--rho_over_H', type=float, default=3.0, help='rho/H for the full-shape plot')
    parser.add_argument('--save', type=str, default=None, help='save figures to this path prefix instead of showing')
    args = parser.parse_args()

    data = load(args.path)
    print(f"loaded {args.path}: interaction={data.get('interaction')}, "
          f"grid={data['lam'].shape[0]}x{data['mu'].shape[0]}, "
          f"{data['k1'].shape[0]} triangles/point")

    plot_cosine_map(data)
    if args.save:
        plt.savefig(f"{args.save}_cosine.png", dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plot_full_shape(data, mu_eff=args.mu_eff, rho_over_H=args.rho_over_H)
    if args.save:
        plt.savefig(f"{args.save}_shape.png", dpi=150, bbox_inches='tight')
    else:
        plt.show()
