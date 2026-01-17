"""High-ℓ stress test in AdS toy geometry.

Purpose
-------
Demonstrate the *sharp* HBMB cutoff logic:
- If the target contains a critical high-ℓ component (ℓ*,m*), the reconstruction error
  stays high for ℓ_max < ℓ*, and collapses (to machine precision) once ℓ_max >= ℓ*.

We compute and plot both:
1) Boundary/horizon error (r=Rb): shows the sharp threshold clearly.
2) Bulk-slice error (r=r_slice): may look much smaller due to AdS radial suppression
   f_ℓ(r) ~ (r/Rb)^ℓ.

Outputs
-------
Saved under `figs_ads_high_l_stress/`.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from hbmb_ads_utils import (
    GridS2,
    ensure_dir,
    precompute_Ylm,
    reconstruct_from_clm,
    rel_L2,
    coverage_energy,
    ads_toy_kernel_f_l,
    build_legendre_kernel,
)


def main() -> None:
    # Match AdS section parameters
    n_mu = 60
    n_phi = 160
    l_ref = 23
    l_hbmb = 21

    Rb = 1.0
    L = 1.0
    Delta = 3.0
    r_slice = 0.45

    l_star = 21
    m_star = 7
    A = 1.0 + 0.0j

    out_dir = "figs_ads_high_l_stress"
    ensure_dir(out_dir)

    grid = GridS2.build(n_mu=n_mu, n_phi=n_phi)

    # Precompute Ylm up to l_ref
    t0 = time.time()
    Y = precompute_Ylm(grid, l_ref)
    print(f"Precompute Ylm up to l_ref={l_ref}: {time.time()-t0:.3f} s")

    # Build analytic coefficients (exact target): eps*Y00 + A*Y_{l*,m*}
    # We choose eps << 1 so the energy is overwhelmingly at ℓ*.
    c = {(l, m): 0.0 + 0.0j for l in range(l_ref + 1) for m in range(-l, l + 1)}
    eps = 1e-15
    c[(0, 0)] = eps + 0.0j
    if l_star <= l_ref:
        c[(l_star, m_star)] = A

    # Reference fields
    psi_bdy_ref = reconstruct_from_clm(grid, c, Y, l_ref, radial_factors=None)
    radial_ref = {l: ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_ref + 1)}
    psi_bulk_ref = reconstruct_from_clm(grid, c, Y, l_ref, radial_factors=radial_ref)

    # Sweep lmax
    lvals = np.arange(l_ref + 1)
    err_bdy = np.zeros(l_ref + 1)
    err_bulk = np.zeros(l_ref + 1)
    cov = np.zeros(l_ref + 1)

    for lmax in range(l_ref + 1):
        psi_bdy_rec = reconstruct_from_clm(grid, c, Y, lmax, radial_factors=None)
        radial = {l: radial_ref[l] for l in range(lmax + 1)}
        psi_bulk_rec = reconstruct_from_clm(grid, c, Y, lmax, radial_factors=radial)

        err_bdy[lmax] = rel_L2(grid, psi_bdy_rec, psi_bdy_ref)
        err_bulk[lmax] = rel_L2(grid, psi_bulk_rec, psi_bulk_ref)
        cov[lmax] = coverage_energy(c, l_max=lmax, l_ref=l_ref)

    # Plot boundary error with plateau annotation
    plt.figure(figsize=(8.5, 5.5))
    y = np.maximum(err_bdy, 1e-300)
    plt.semilogy(lvals, y, linewidth=2)
    plt.axvline(l_star, linestyle="--", label=fr"$\ell_\star={l_star}$")

    # Plateau estimate: median for l < l_star
    plateau = float(np.median(y[lvals < l_star])) if np.any(lvals < l_star) else None
    if plateau is not None:
        plt.axhline(plateau, linestyle=":")
        plt.text(0.02, 0.92, f"pre-ℓ* level ~ {plateau:.3e}", transform=plt.gca().transAxes)

    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("rel L2 error (log)")
    plt.title("AdS high-ℓ stress: boundary error vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_stress_error_vs_lmax_boundary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Plot bulk error with plateau annotation
    plt.figure(figsize=(8.5, 5.5))
    yb = np.maximum(err_bulk, 1e-300)
    plt.semilogy(lvals, yb, linewidth=2)
    plt.axvline(l_star, linestyle="--", label=fr"$\ell_\star={l_star}$")

    plateau_b = float(np.median(yb[lvals < l_star])) if np.any(lvals < l_star) else None
    if plateau_b is not None:
        plt.axhline(plateau_b, linestyle=":")
        plt.text(0.02, 0.92, f"pre-ℓ* level ~ {plateau_b:.3e}", transform=plt.gca().transAxes)

    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("rel L2 error (log)")
    plt.title("AdS high-ℓ stress: bulk-slice error vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_stress_error_vs_lmax_bulk.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Coverage plot
    plt.figure()
    plt.plot(lvals, cov, linewidth=2)
    plt.axvline(l_star, linestyle="--", label=fr"$\ell_\star={l_star}$")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel(r"$E(\ell_{\max})$")
    plt.title("AdS high-ℓ stress: spectral coverage vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_stress_coverage_vs_lmax.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Maps at lmax=20 and lmax=21 (boundary, |psi|^2)
    for lmax in [l_star - 1, l_star]:
        psi_bdy_rec = reconstruct_from_clm(grid, c, Y, lmax, radial_factors=None)
        fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

        im0 = ax[0].imshow(np.abs(psi_bdy_ref) ** 2, origin="lower", aspect="auto")
        ax[0].set_title("Target |psi|^2 (boundary)")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(np.abs(psi_bdy_rec) ** 2, origin="lower", aspect="auto")
        ax[1].set_title(fr"Recon |psi|^2, $\ell_{{\max}}={lmax}$")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        im2 = ax[2].imshow(np.abs(psi_bdy_rec - psi_bdy_ref), origin="lower", aspect="auto")
        ax[2].set_title("Abs error |Δpsi|")
        plt.colorbar(im2, ax=ax[2], fraction=0.046)

        fig.suptitle("AdS high-ℓ stress: boundary maps", fontsize=12)
        fig.savefig(os.path.join(out_dir, f"ads_horizon_recon_abs2_lmax{lmax}.png"), dpi=200)
        plt.close(fig)

    # Print key numbers
    print("=== AdS high-ℓ stress summary ===")
    print(f"l_star={l_star}, m_star={m_star}, l_ref={l_ref}")
    print(f"Boundary: err(lmax=20)={err_bdy[l_star-1]:.6e}, err(lmax=21)={err_bdy[l_star]:.6e}")
    print(f"Bulk:     err(lmax=20)={err_bulk[l_star-1]:.6e}, err(lmax=21)={err_bulk[l_star]:.6e}")
    print(f"Bulk plateau (median, l<l*): {plateau_b:.6e}" if plateau_b is not None else "Bulk plateau: n/a")


if __name__ == "__main__":
    main()
