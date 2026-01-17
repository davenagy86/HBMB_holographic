"""Hydrogen control (AdS): phase-stable HBMB reconstruction with an AdS radial kernel.

Default target: 2p_z angular state -> Y_{1,0} on the boundary screen (r=Rb).
We reconstruct:
- Horizon/boundary field (radial factor = 1)
- Bulk slice at r=r_slice (radial factor = f_l(r_slice))

All plots/figures are saved automatically.
"""

from __future__ import annotations

import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm

from hbmb_ads_utils import (
    GridS2,
    ensure_dir,
    ads_toy_kernel_f_l,
    precompute_Ylm,
    project_to_clm,
    reconstruct_from_clm,
    rel_L2,
    coverage_energy,
)


def main() -> None:
    # Parameters (match AdS section summary)
    n_mu = 60
    n_phi = 160
    l_ref = 23
    l_hbmb = 21

    Rb = 1.0
    L = 1.0
    Delta = 3.0
    r_slice = 0.45

    out_dir = "figs_ads_hydrogen_control"
    ensure_dir(out_dir)

    grid = GridS2.build(n_mu=n_mu, n_phi=n_phi)

    t0 = time.time()
    Y = precompute_Ylm(grid, l_ref)
    print(f"Precompute Ylm: {time.time()-t0:.3f} s")

    # Target boundary field: Y_{1,0}
    l_t, m_t = 1, 0
    psi_bdy = sph_harm(m_t, l_t, grid.PH, grid.TH).astype(np.complex128)

    # Project to c_{lm} up to l_ref
    c_ref = project_to_clm(grid, psi_bdy, Y, l_ref)

    # Ground truth (reference) boundary and bulk-slice
    psi_bdy_ref = reconstruct_from_clm(grid, c_ref, Y, l_ref, radial_factors=None)
    radial_ref = {l: ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_ref + 1)}
    psi_bulk_ref = reconstruct_from_clm(grid, c_ref, Y, l_ref, radial_factors=radial_ref)

    # Scan over cutoff
    lvals = np.arange(l_ref + 1)
    err_bdy = np.zeros(l_ref + 1)
    err_bulk = np.zeros(l_ref + 1)
    cov = np.zeros(l_ref + 1)

    for lmax in range(l_ref + 1):
        psi_bdy_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=None)
        radial = {l: radial_ref[l] for l in range(lmax + 1)}
        psi_bulk_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)

        err_bdy[lmax] = rel_L2(grid, psi_bdy_rec, psi_bdy_ref)
        err_bulk[lmax] = rel_L2(grid, psi_bulk_rec, psi_bulk_ref)
        cov[lmax] = coverage_energy(c_ref, l_max=lmax, l_ref=l_ref)

    # Curves
    plt.figure()
    plt.semilogy(lvals, np.maximum(err_bdy, 1e-300), label="Boundary error")
    plt.semilogy(lvals, np.maximum(err_bulk, 1e-300), label="Bulk-slice error")
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("rel L2 error")
    plt.title("Hydrogen control (AdS): error vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_hydrogen_error_vs_lmax.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(lvals, cov)
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel(r"$E(\ell_{\max})$")
    plt.title("Hydrogen control (AdS): spectral coverage")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_hydrogen_coverage_vs_lmax.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Maps at HBMB cutoff
    lmax = l_hbmb
    psi_bdy_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=None)
    radial = {l: radial_ref[l] for l in range(lmax + 1)}
    psi_bulk_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    im0 = ax[0, 0].imshow(np.abs(psi_bdy_ref) ** 2, origin="lower", aspect="auto")
    ax[0, 0].set_title("Target |psi|^2 (boundary)")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)

    im1 = ax[0, 1].imshow(np.abs(psi_bdy_rec) ** 2, origin="lower", aspect="auto")
    ax[0, 1].set_title(fr"Recon |psi|^2 (boundary), $\ell_{{\max}}={lmax}$")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)

    im2 = ax[0, 2].imshow(np.abs(psi_bdy_rec - psi_bdy_ref), origin="lower", aspect="auto")
    ax[0, 2].set_title("Boundary abs error |Δpsi|")
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

    im3 = ax[1, 0].imshow(np.abs(psi_bulk_ref) ** 2, origin="lower", aspect="auto")
    ax[1, 0].set_title("Target |psi|^2 (bulk slice)")
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046)

    im4 = ax[1, 1].imshow(np.abs(psi_bulk_rec) ** 2, origin="lower", aspect="auto")
    ax[1, 1].set_title(fr"Recon |psi|^2 (bulk), $\ell_{{\max}}={lmax}$")
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046)

    im5 = ax[1, 2].imshow(np.abs(psi_bulk_rec - psi_bulk_ref), origin="lower", aspect="auto")
    ax[1, 2].set_title("Bulk abs error |Δpsi|")
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046)

    fig.suptitle("Hydrogen control (AdS): 2p_z angular target", fontsize=12)
    fig.savefig(os.path.join(out_dir, "ads_hydrogen_bulk_maps_lmax_21.png"), dpi=200)
    plt.close(fig)

    # Print key sanity
    print(f"Hydrogen (AdS) relL2 boundary @ lmax={l_hbmb}: {err_bdy[l_hbmb]:.3e}")
    print(f"Hydrogen (AdS) relL2 bulk     @ lmax={l_hbmb}: {err_bulk[l_hbmb]:.3e}")


if __name__ == "__main__":
    main()
