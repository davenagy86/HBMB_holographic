"""HKLL vs HBMB (AdS toy): direct numerical cross-check.

We compare two equivalent reconstructions of a bulk-slice field from boundary data:

1) HBMB mode synthesis:
   psi_HBMB(r,Omega) = sum_{l<=lmax,m} c_{lm} f_l(r) Y_{lm}(Omega)

2) HKLL-like smearing integral (angular sector):
   psi_HKLL(r,Omega) = ∫ dOmega' K_r(gamma) psi_bdy(Omega')
where the kernel is built from a truncated Legendre series:
   K_r(gamma) = sum_{l<=lmax} f_l(r) (2l+1)/(4pi) P_l(cos gamma)

We evaluate the HKLL integral on a random subset of grid points for speed.

All figures are saved automatically.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from hbmb_ads_utils import (
    GridS2,
    ensure_dir,
    ads_toy_kernel_f_l,
    precompute_Ylm,
    project_to_clm,
    reconstruct_from_clm,
    blob_on_s2,
    build_legendre_kernel,
)


def main() -> None:
    out_dir = "figs_ads_hkll_compare"
    ensure_dir(out_dir)

    # Parameters (consistent with the AdS section)
    n_mu = 60
    n_phi = 160
    l_ref = 23
    l_max = 21

    Rb = 1.0
    L = 1.0
    Delta = 3.0
    r_slice = 0.45

    # Boundary target: same blob as the dS toy
    theta0 = 2.1
    phi0 = 1.6
    sigma = 0.25

    # Sampling for HKLL evaluation
    rng = np.random.default_rng(0)
    n_eval = 350

    grid = GridS2.build(n_mu=n_mu, n_phi=n_phi)

    t0 = time.time()
    Y = precompute_Ylm(grid, l_ref)
    print(f"Precompute Ylm up to l_ref={l_ref}: {time.time()-t0:.3f} s")

    psi_bdy = blob_on_s2(grid, theta0=theta0, phi0=phi0, sigma=sigma)

    # Project to coefficients (reference)
    c_ref = project_to_clm(grid, psi_bdy, Y, l_ref)

    # HBMB synthesis on full grid (bulk slice)
    radial = {l: ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_max + 1)}
    psi_hbmb = reconstruct_from_clm(grid, c_ref, Y, l_max, radial_factors=radial)

    # Prepare HKLL kernel weights and geometry
    # Robust per-point dΩ weights on the full (mu,phi) grid.
    # (Some older helper variants return only the 1D Gauss weights.)
    W = ((grid.w_mu[:, None]) * grid.dphi * np.ones((1, grid.n_phi), dtype=float)).reshape(-1)
    psi_src = psi_bdy.reshape(-1)
    n_src = grid.unit_vectors()  # (N,3)

    # radial factors f_l(r) for l=0..l_max
    f_l = np.array([ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_max + 1)], dtype=float)

    # Random evaluation points
    N = n_src.shape[0]
    idx_eval = rng.choice(N, size=min(n_eval, N), replace=False)
    n_eval_vec = n_src[idx_eval]

    psi_hkll_eval = np.zeros(idx_eval.size, dtype=np.complex128)
    psi_hbmb_eval = psi_hbmb.reshape(-1)[idx_eval]

    t1 = time.time()
    for k, n_vec in enumerate(n_eval_vec):
        cosg = n_src @ n_vec  # (N,)
        cosg = np.clip(cosg, -1.0, 1.0)
        K = build_legendre_kernel(cosg, f_l=f_l, l_max=l_max)  # (N,)
        psi_hkll_eval[k] = np.sum(K * psi_src * W)
    t_hkll = time.time() - t1

    rel = np.linalg.norm(psi_hkll_eval - psi_hbmb_eval) / (np.linalg.norm(psi_hbmb_eval) + 1e-300)
    mx = float(np.max(np.abs(psi_hkll_eval - psi_hbmb_eval)))

    print("=== HKLL vs HBMB (AdS toy) ===")
    print(f"grid: n_mu={n_mu}, n_phi={n_phi}, N={N}")
    print(f"r_slice={r_slice}, l_max={l_max}, samples={idx_eval.size}")
    print(f"HKLL eval time: {t_hkll:.3f} s")
    print(f"rel L2 (samples) = {rel:.3e}")
    print(f"max |diff|       = {mx:.3e}")

    # Scatter plot (real part)
    plt.figure(figsize=(6.5, 6.0))
    plt.plot()
    plt.plot(np.real(psi_hkll_eval), "x", markersize=3)
    plt.plot(np.real(psi_hbmb_eval), ".", markersize=3)
    plt.xlabel("Re(HBMB mode synthesis)")
    plt.ylabel("Re(HKLL smearing)")
    plt.title("HKLL vs HBMB (real part) at sampled points")
    plt.savefig(os.path.join(out_dir, "hkll_vs_hbmb_real_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Scatter plot (imag part)
    plt.figure(figsize=(6.5, 6.0))
    plt.plot(np.imag(psi_hbmb_eval), np.imag(psi_hkll_eval), ".", markersize=3)
    plt.plot(np.imag(psi_hbmb_eval), np.imag(psi_hkll_eval), ".", markersize=3)
    plt.xlabel("Im(HBMB mode synthesis)")
    plt.ylabel("Im(HKLL smearing)")
    plt.title("HKLL vs HBMB (imag part) at sampled points")
    plt.savefig(os.path.join(out_dir, "hkll_vs_hbmb_imag_scatter.png"), dpi=200, bbox_inches="tight")
    plt.close()
    # --- Plot: direct pointwise comparison (amplitude) ---
    x = np.arange(len(psi_hbmb_eval))

    plt.figure(figsize=(8, 4))

    plt.scatter(x, np.abs(psi_hbmb_eval),
                s=28, marker='o', color='tab:blue', alpha=0.85,
                label=r'HBMB eval: $|\psi_{\rm HBMB}|$')

    plt.scatter(x, np.abs(psi_hkll_eval),
                s=40, marker='x', color='tab:red', alpha=0.85,
                label=r'HKLL eval: $|\psi_{\rm HKLL}|$')

    plt.xlabel('eval point index')
    plt.ylabel(r'$|\psi|$')
    plt.title('AdS HKLL vs HBMB (pointwise): amplitude')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ads_hkll_vs_hbmb_pointwise_abs.png', dpi=200)
    plt.close()

    # --- Plot: phase comparison ---
    plt.figure(figsize=(8, 4))

    plt.scatter(x, np.angle(psi_hbmb_eval),
                s=28, marker='o', color='tab:blue', alpha=0.85,
                label=r'HBMB eval: $\arg(\psi_{\rm HBMB})$')

    plt.scatter(x, np.angle(psi_hkll_eval),
                s=40, marker='x', color='tab:red', alpha=0.85,
                label=r'HKLL eval: $\arg(\psi_{\rm HKLL})$')

    plt.xlabel('eval point index')
    plt.ylabel(r'phase [rad]')
    plt.title('AdS HKLL vs HBMB (pointwise): phase')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ads_hkll_vs_hbmb_pointwise_phase.png', dpi=200)
    plt.close()

    # --- Plot: parity plot (abs) ---
    plt.figure(figsize=(5, 5))

    a = np.abs(psi_hbmb_eval)
    b = np.abs(psi_hkll_eval)

    plt.scatter(a, b, s=34, marker='x', color='tab:red', alpha=0.85, label='HKLL vs HBMB')
    mn = min(a.min(), b.min())
    mx = max(a.max(), b.max())
    plt.plot([mn, mx], [mn, mx], linestyle='--', linewidth=1.5, color='k', alpha=0.6, label='y=x')

    plt.xlabel(r'$|\psi_{\rm HBMB}|$')
    plt.ylabel(r'$|\psi_{\rm HKLL}|$')
    plt.title('AdS HKLL vs HBMB: parity (amplitude)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ads_hkll_vs_hbmb_parity_abs.png', dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
