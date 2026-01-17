"""AdS toy (FAST): boundary blob -> bulk-slice reconstruction with HBMB cutoff.

This script reproduces the AdS toy numbers stored in ads_section_summary.json:
- Sbit=490.873852 -> l_hbmb=21
- l_ref=23
- grid: n_mu=60, n_phi=160
- kernel: Delta=3, Rb=1, L=1
- bulk slice: r_slice=0.45

All plots/figures are saved automatically.
"""

from __future__ import annotations

import json
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
    rel_L2,
    coverage_energy,
    blob_on_s2,
)


def main() -> None:
    # -----------------------
    # Parameters (match summary)
    # -----------------------
    n_mu = 60
    n_phi = 160
    l_ref = 23
    Sbit = 490.873852
    l_hbmb = 21

    Rb = 1.0
    L = 1.0
    Delta = 3.0
    r_slice = 0.45

    # Blob target (same as dS toy)
    theta0 = 2.1
    phi0 = 1.6
    sigma = 0.25

    out_dir = "figs_ads_toy_fast"
    ensure_dir(out_dir)

    # -----------------------
    # Build grid and precompute Ylm
    # -----------------------
    grid = GridS2.build(n_mu=n_mu, n_phi=n_phi)

    t0 = time.time()
    Y = precompute_Ylm(grid, l_ref)
    t_pre = time.time() - t0
    print(f"Precompute Ylm up to l_ref={l_ref}: {t_pre:.3f} s")

    # -----------------------
    # Boundary target and reference coefficients
    # -----------------------
    psi_bdy = blob_on_s2(grid, theta0=theta0, phi0=phi0, sigma=sigma)

    c_ref = project_to_clm(grid, psi_bdy, Y, l_ref)

    # Reference (ground-truth) bulk slice uses all modes up to l_ref
    radial_ref = {l: ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_ref + 1)}
    psi_bulk_ref = reconstruct_from_clm(grid, c_ref, Y, l_ref, radial_factors=radial_ref)

    # -----------------------
    # Scan over cutoff
    # -----------------------
    relL2 = np.zeros(l_ref + 1, dtype=float)
    cov = np.zeros(l_ref + 1, dtype=float)

    for lmax in range(l_ref + 1):
        radial = {l: radial_ref[l] for l in range(lmax + 1)}
        psi_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)
        relL2[lmax] = rel_L2(grid, psi_rec, psi_bulk_ref)
        cov[lmax] = coverage_energy(c_ref, l_max=lmax, l_ref=l_ref)

    # -----------------------
    # Save plots
    # -----------------------
    lvals = np.arange(l_ref + 1)

    plt.figure()
    plt.semilogy(lvals, np.maximum(relL2, 1e-300))
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("rel L2 error")
    plt.title("AdS toy: bulk-slice error vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_toy_bulk_error_vs_lmax_fast.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(lvals, cov)
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel(r"$E(\ell_{\max})$")
    plt.title("AdS toy: spectral coverage vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_toy_coverage_vs_lmax_fast.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------
    # Save map figures at lmax=10 and lmax=21
    # -----------------------
    for lmax in [10, l_hbmb]:
        radial = {l: radial_ref[l] for l in range(lmax + 1)}
        psi_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)

        fig, ax = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        im0 = ax[0].imshow(np.abs(psi_bulk_ref), origin="lower", aspect="auto")
        ax[0].set_title("Target |psi| (bulk slice)")
        plt.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(np.abs(psi_rec), origin="lower", aspect="auto")
        ax[1].set_title(fr"Recon |psi| (bulk), $\ell_{{\max}}={lmax}$")
        plt.colorbar(im1, ax=ax[1], fraction=0.046)

        im2 = ax[2].imshow(np.abs(psi_rec - psi_bulk_ref), origin="lower", aspect="auto")
        ax[2].set_title("Abs error |Δpsi|")
        plt.colorbar(im2, ax=ax[2], fraction=0.046)

        fig.suptitle("AdS toy: boundary blob -> bulk slice", fontsize=12)
        fig.savefig(os.path.join(out_dir, f"ads_toy_bulk_maps_lmax_{lmax}_fast.png"), dpi=200)
        plt.close(fig)

    # -----------------------
    # Kernel profile plot (for the paper)
    # -----------------------
    r_grid = np.linspace(0.05, Rb, 300)
    ell_list = [0, 1, 2, 5, 10, 21]

    plt.figure()
    for l in ell_list:
        prof = np.array([ads_toy_kernel_f_l(r, l, Rb=Rb, L=L, Delta=Delta) for r in r_grid])
        plt.plot(r_grid, prof, label=fr"$\ell={l}$")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$f_\ell(r)$")
    plt.title("AdS toy radial kernel profiles (normalized at r=Rb)")
    plt.legend(ncol=3)
    plt.savefig(os.path.join(out_dir, "ads_kernel_profiles.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # -----------------------
    # Print key numbers (match the summary)
    # -----------------------
    print(f"Sbit={Sbit}, HBMB l_max={l_hbmb}, l_ref={l_ref}")
    print(f"relL2(lmax=10)={relL2[10]:.12e}")
    print(f"relL2(lmax=21)={relL2[l_hbmb]:.12e}")
    print(f"coverage(lmax=10)={cov[10]:.12e}")
    print(f"coverage(lmax=21)={cov[l_hbmb]:.12e}")


if __name__ == "__main__":
    main()
