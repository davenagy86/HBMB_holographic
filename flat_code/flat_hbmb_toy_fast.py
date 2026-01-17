# hbmb_flat_toy_fast.py
# Flat toy: local-horizon eigenmode reconstruction on S^2 (HBMB cutoff)
# Comments in English (publication/GitHub-friendly)

import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import __version__ as scipy_version
from scipy.special import sph_harm as _sph_harm

# -----------------------------------------------------------------------------
# Backend selection: use physics convention Y_lm(theta_polar, phi_azimuth)
# -----------------------------------------------------------------------------
try:
    # Newer SciPy: physics convention directly
    from scipy.special import sph_harm_y as _sph_harm_y  # signature: (n, m, theta, phi)
    BACKEND = "sph_harm_y"

    def Ylm_block(l: int, m_block, TH, PH):
        """
        Return Y_{l m}(TH, PH) using physics convention:
        TH = theta (polar), PH = phi (azimuth).
        """
        return _sph_harm_y(l, m_block, TH, PH)

except Exception:
    # Older SciPy: sph_harm uses swapped naming:
    # sph_harm(m, n, theta_azimuth, phi_polar)
    BACKEND = "sph_harm"

    def Ylm_block(l: int, m_block, TH, PH):
        """
        Return Y_{l m}(TH, PH) in physics convention by swapping args:
        sph_harm expects (theta_azimuth, phi_polar), so feed (PH, TH).
        """
        return _sph_harm(m_block, l, PH, TH)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def hbmb_lmax(R: float, lp_eff: float) -> tuple[float, int]:
    """HBMB/Bekenstein-style cutoff: S_bit = pi (R/lp_eff)^2, lmax ~ floor(sqrt(S)-1)."""
    S_bit = np.pi * (R / lp_eff) ** 2
    lmax = int(np.floor(np.sqrt(S_bit) - 1.0))
    return S_bit, lmax

def angular_distance_on_s2(TH, PH, th0: float, ph0: float):
    """Great-circle distance gamma on the unit sphere."""
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)

    x0 = np.sin(th0) * np.cos(ph0)
    y0 = np.sin(th0) * np.sin(ph0)
    z0 = np.cos(th0)

    dot = x * x0 + y * y0 + z * z0
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def target_psi(TH, PH, th0=2.1, ph0=1.6, sigma=0.25):
    """
    A smooth "blob" on the sphere (band-limited-ish in practice).
    TH: theta (polar), PH: phi (azimuth).
    """
    gamma = angular_distance_on_s2(TH, PH, th0, ph0)
    return np.exp(-(gamma / sigma) ** 2)

def rel_l2_error(psi, psi_rec, W):
    num = np.sum(np.abs(psi - psi_rec) ** 2 * W)
    den = np.sum(np.abs(psi) ** 2 * W) + 1e-300
    return float(np.sqrt(num / den))

def max_error(psi, psi_rec):
    denom = np.max(np.abs(psi)) + 1e-300
    return float(np.max(np.abs(psi - psi_rec)) / denom)

def spectral_coverage(energies_l):
    """Return cumulative coverage E(lmax) normalized by total energy."""
    energies_l = np.asarray(energies_l, dtype=float)
    total = float(np.sum(energies_l)) + 1e-300
    return np.cumsum(energies_l) / total

# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    # ---- Toy parameters (edit freely) ----
    R = 1.0
    lp_eff = 0.08

    # Grid resolution (theta includes endpoints; phi is periodic)
    n_theta = 120
    n_phi = 256

    # Reference max degree (used for sanity + curves)
    l_ref = 40

    # Which lmax maps to save
    lmax_maps = [10, None, 30]  # We'll insert HBMB lmax into the middle slot

    # Output dir
    out_dir = "figs_flat_toy_fast"
    os.makedirs(out_dir, exist_ok=True)

    # ---- HBMB cutoff ----
    S_bit, l_hbmb = hbmb_lmax(R, lp_eff)
    # lmax_maps are illustrative (e.g., 10 and 30)
    # (HBMB cutoff is indicated by the vertical line)
    # lmax_maps[1] = l_hbmb

    print(f"Backend: {BACKEND} (SciPy {scipy_version})")
    print(f"S_bit={S_bit:.3f}, HBMB l_max={l_hbmb}")
    print(f"Grid: n_theta={n_theta}, n_phi={n_phi}, l_ref={l_ref}")

    # ---- Build grid ----
    theta = np.linspace(0.0, np.pi, n_theta, endpoint=True)           # polar
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)        # azimuth
    TH, PH = np.meshgrid(theta, phi, indexing="ij")

    dtheta = np.pi / (n_theta - 1)
    dphi = 2.0 * np.pi / n_phi
    W = np.sin(TH) * dtheta * dphi  # quadrature weights

    # ---- Target signal on the local horizon S^2 ----
    psi = target_psi(TH, PH)

    # ---- Iterative reconstruction up to l_ref (fast block evaluation per l) ----
    psi_rec = np.zeros_like(psi, dtype=np.complex128)

    relL2 = np.zeros(l_ref + 1, dtype=float)
    mx = np.zeros(l_ref + 1, dtype=float)
    energies_l = np.zeros(l_ref + 1, dtype=float)

    # Save maps at chosen lmax (we keep snapshots)
    snapshots = {lm: None for lm in lmax_maps}

    for l in range(l_ref + 1):
        m_vals = np.arange(-l, l + 1)[:, None, None]  # shape (2l+1, 1, 1)

        # Y has shape (2l+1, n_theta, n_phi)
        Y = Ylm_block(l, m_vals, TH[None, :, :], PH[None, :, :])

        # Coefficients c_{l m} = ∫ psi * conj(Y_lm) dΩ
        # Broadcasting: psi[None,...] * conj(Y) * W[None,...] -> sum over theta,phi
        c_lm = np.sum(psi[None, :, :] * np.conjugate(Y) * W[None, :, :], axis=(1, 2))

        # Energy at degree l
        energies_l[l] = float(np.sum(np.abs(c_lm) ** 2))

        # Add contribution to reconstruction: sum_m c_lm * Y_lm
        psi_rec += np.sum(c_lm[:, None, None] * Y, axis=0)

        # Metrics at current lmax=l
        relL2[l] = rel_l2_error(psi, psi_rec, W)
        mx[l] = max_error(psi, psi_rec)

        if l in snapshots:
            snapshots[l] = psi_rec.copy()

    cov = spectral_coverage(energies_l)

    # ---- Sanity check at l_ref (should be very small residual if conventions match) ----
    print(f"Sanity @ l_ref={l_ref}: relL2={relL2[-1]:.6f}, max={mx[-1]:.6f}")
    if relL2[-1] > 1e-2:
        print("WARNING: Large residual at l_ref. This usually indicates a theta/phi convention mismatch.")
        print("Check that you are using physics convention (theta polar, phi azimuth) consistently.")

    # ---- Plots ----
    # Error vs cutoff
    plt.figure()
    plt.plot(np.arange(l_ref + 1), relL2, label="Rel L2")
    plt.plot(np.arange(l_ref + 1), mx, label="Max")
    plt.axvline(l_hbmb, linestyle="--", label=f"HBMB l_max={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel("Error")
    plt.title("Flat toy: reconstruction error vs cutoff (FAST)")
    plt.legend()
    p1 = os.path.join(out_dir, "flat_toy_error_vs_lmax_fast.png")
    plt.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close()

    # Coverage vs cutoff
    plt.figure()
    plt.plot(np.arange(l_ref + 1), cov, label="Coverage")
    plt.axvline(l_hbmb, linestyle="--", label=f"HBMB l_max={l_hbmb}")
    plt.xlabel(r"$\ell_{\max}$")
    plt.ylabel(r"$E(\ell_{\max})$")
    plt.title("Flat toy: spectral coverage vs cutoff (FAST)")
    plt.legend()
    p2 = os.path.join(out_dir, "flat_toy_coverage_vs_lmax_fast.png")
    plt.savefig(p2, dpi=200, bbox_inches="tight")
    plt.close()

    # Maps (|psi|) at selected lmax
    for lm in lmax_maps:
        psi_lm = snapshots.get(lm, None)
        if psi_lm is None:
            continue

        fig, ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        im0 = ax[0].imshow(np.abs(psi), origin="lower", aspect="auto")
        ax[0].set_title("Target |psi|")
        ax[0].set_xlabel("phi")
        ax[0].set_ylabel("theta")
        fig.colorbar(im0, ax=ax[0], fraction=0.046)

        im1 = ax[1].imshow(np.abs(psi_lm), origin="lower", aspect="auto")
        ax[1].set_title(f"Recon |psi| (lmax={lm})")
        ax[1].set_xlabel("phi")
        ax[1].set_ylabel("theta")
        fig.colorbar(im1, ax=ax[1], fraction=0.046)

        fig.suptitle("Flat toy: angular reconstruction on the local horizon (FAST)")
        p = os.path.join(out_dir, f"flat_toy_maps_lmax_{lm}_fast.png")
        plt.savefig(p, dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figures to: {out_dir}")

if __name__ == "__main__":
    main()
