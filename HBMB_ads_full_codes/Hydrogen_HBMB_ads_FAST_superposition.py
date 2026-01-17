"""Hydrogen superposition (AdS): complex weights + phase correctness.

We build a complex superposition on the boundary:
    psi = w0*Y00 + w1*Y10 + w2*Y20
and reconstruct both boundary and a bulk slice.

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

    out_dir = "figs_ads_superposition"
    ensure_dir(out_dir)

    grid = GridS2.build(n_mu=n_mu, n_phi=n_phi)

    t0 = time.time()
    Y = precompute_Ylm(grid, l_ref)
    print(f"Precompute Ylm: {time.time()-t0:.3f} s")

    # Complex superposition (phase-sensitive)
    w0 = 1.0 + 0.0j
    w1 = 0.6 - 0.2j
    w2 = -0.35 + 0.55j

    psi_bdy = (
        w0 * sph_harm(0, 0, grid.PH, grid.TH)
        + w1 * sph_harm(0, 1, grid.PH, grid.TH)
        + w2 * sph_harm(0, 2, grid.PH, grid.TH)
    ).astype(np.complex128)

    c_ref = project_to_clm(grid, psi_bdy, Y, l_ref)

    psi_bdy_ref = reconstruct_from_clm(grid, c_ref, Y, l_ref, radial_factors=None)
    radial_ref = {l: ads_toy_kernel_f_l(r_slice, l, Rb=Rb, L=L, Delta=Delta) for l in range(l_ref + 1)}
    psi_bulk_ref = reconstruct_from_clm(grid, c_ref, Y, l_ref, radial_factors=radial_ref)

    # Scan over cutoff
    lvals = np.arange(l_ref + 1)
    err = np.zeros(l_ref + 1)
    cov = np.zeros(l_ref + 1)

    for lmax in range(l_ref + 1):
        radial = {l: radial_ref[l] for l in range(lmax + 1)}
        psi_bulk_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)
        err[lmax] = rel_L2(grid, psi_bulk_rec, psi_bulk_ref)
        cov[lmax] = coverage_energy(c_ref, l_max=lmax, l_ref=l_ref)

    # Curves
    plt.figure()
    plt.semilogy(lvals, np.maximum(err, 1e-300))
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("rel L2 error")
    plt.title("AdS superposition: bulk-slice error vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_superposition_error_vs_lmax.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(lvals, cov)
    plt.axvline(l_hbmb, linestyle="--", label=fr"HBMB $\ell_{{\max}}$={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel(r"$E(\ell_{\max})$")
    plt.title("AdS superposition: spectral coverage vs cutoff")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "ads_superposition_coverage_vs_lmax.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Maps at HBMB cutoff
    lmax = l_hbmb
    radial = {l: radial_ref[l] for l in range(lmax + 1)}
    psi_bulk_rec = reconstruct_from_clm(grid, c_ref, Y, lmax, radial_factors=radial)

    # Phase is undefined where magnitude is ~0, so mask low magnitude for display
    mag = np.abs(psi_bulk_ref)
    phase_ref = np.angle(psi_bulk_ref)
    phase_rec = np.angle(psi_bulk_rec)

    fig, ax = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)

    im0 = ax[0, 0].imshow(np.abs(psi_bulk_ref) ** 2, origin="lower", aspect="auto")
    ax[0, 0].set_title("Target |psi|^2 (bulk)")
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)

    im1 = ax[0, 1].imshow(np.abs(psi_bulk_rec) ** 2, origin="lower", aspect="auto")
    ax[0, 1].set_title(fr"Recon |psi|^2 (bulk), $\ell_{{\max}}={lmax}$")
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046)

    im2 = ax[0, 2].imshow(np.abs(psi_bulk_rec - psi_bulk_ref), origin="lower", aspect="auto")
    ax[0, 2].set_title("Abs error |Δpsi|")
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046)

    im3 = ax[1, 0].imshow(phase_ref, origin="lower", aspect="auto")
    ax[1, 0].set_title("Target arg(psi) (bulk)")
    plt.colorbar(im3, ax=ax[1, 0], fraction=0.046)

    im4 = ax[1, 1].imshow(phase_rec, origin="lower", aspect="auto")
    ax[1, 1].set_title(fr"Recon arg(psi) (bulk), $\ell_{{\max}}={lmax}$")
    plt.colorbar(im4, ax=ax[1, 1], fraction=0.046)

    im5 = ax[1, 2].imshow(np.angle(psi_bulk_rec * np.conjugate(psi_bulk_ref)), origin="lower", aspect="auto")
    ax[1, 2].set_title("Phase diff arg(rec*conj(target))")
    plt.colorbar(im5, ax=ax[1, 2], fraction=0.046)

    fig.suptitle("AdS superposition: phase-correct HBMB reconstruction", fontsize=12)
    fig.savefig(os.path.join(out_dir, "ads_superposition_phase_lmax_21.png"), dpi=200)
    plt.close(fig)

    print(f"Superposition (AdS) relL2 bulk @ lmax={l_hbmb}: {err[l_hbmb]:.3e}")


if __name__ == "__main__":
    main()
