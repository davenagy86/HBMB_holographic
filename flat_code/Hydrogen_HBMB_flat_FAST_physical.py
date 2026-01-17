import json
import time
from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
OUTDIR = Path("figs_hydrogen_hbmb_flat_fast_physical")
OUTDIR.mkdir(parents=True, exist_ok=True)

# HBMB "bit entropy" toy setup
R_HORIZON = 1.0
LP_EFF = 0.08  # keep consistent with your earlier runs
S_BIT = pi * (R_HORIZON / LP_EFF) ** 2
HBMB_LMAX = int(np.floor(np.sqrt(S_BIT) - 1))

# Numerical setup (analysis grid: Gauss in mu, FFT in phi)
L_REF = 60
N_THETA_GAUSS = 140
N_PHI = 320  # should satisfy N_PHI >= 2*L_REF+1 to avoid aliasing

# Plot grid (publication-friendly: uniform theta, uniform phi)
N_THETA_PLOT = 180
N_PHI_PLOT = 360

# Which lmax maps to save
LMAX_SHOW = [10, HBMB_LMAX, 30]

# Toy bulk extension radii (kernel: (r/R)^l)
BULK_R_LIST = [1.0, 0.7, 0.4]

# -----------------------------------------------------------------------------
# Target definition: ORBITAL or SUPERPOSITION
# -----------------------------------------------------------------------------
ORBITAL = "2p_z"  # try: "2p_z", "2p_x", "2p_y", "3d_xy", "3d_x2y2", ...

# If non-empty, overrides ORBITAL
SUPERPOSITION = []


# Optional: add a synthetic high-l probe to make HBMB cutoff "bite" visually
# Example: one mode near HBMB_LMAX
CUSTOM_TERMS = [
    # (l, m, coeff)
    # (20, 7, 0.20 * np.exp(1j * 0.3)),
]
# -----------------------------------------------------------------------------
# Spherical-harmonics utilities (Legendre + FFT, robust conventions)
# -----------------------------------------------------------------------------
def N_lm(l: int, m_abs: int) -> float:
    """Normalization for complex spherical harmonics (orthonormal on S^2)."""
    return float(np.sqrt((2*l + 1) / (4*pi) *
                         special.factorial(l - m_abs) / special.factorial(l + m_abs)))

def T_lm_theta(l: int, m_abs: int, mu: np.ndarray) -> np.ndarray:
    """
    Theta-dependent factor: N_lm * P_l^m(mu).
    SciPy lpmv includes Condon–Shortley phase.
    """
    return N_lm(l, m_abs) * special.lpmv(m_abs, l, mu)

def list_orbital_coeffs(name: str):
    """
    Return list of (l, m, coeff) for common *real* hydrogen orbital angular parts.
    Coeffs are in the complex Y_lm basis (normalized up to an overall factor;
    we normalize the final alm anyway).
    """
    name = name.strip().lower()

    if name == "1s":
        return [(0, 0, 1.0 + 0j)]

    # p orbitals (l=1)
    if name in ["2p_z", "pz"]:
        return [(1, 0, 1.0 + 0j)]
    if name in ["2p_x", "px"]:
        # p_x ∝ (Y_1,-1 - Y_1,1)/sqrt(2)
        return [(1, -1, 1/np.sqrt(2) + 0j), (1, 1, -1/np.sqrt(2) + 0j)]
    if name in ["2p_y", "py"]:
        # p_y ∝ i(Y_1,-1 + Y_1,1)/sqrt(2)
        return [(1, -1, 1j/np.sqrt(2)), (1, 1, 1j/np.sqrt(2))]

    # d orbitals (l=2)
    if name in ["3d_z2", "dz2"]:
        return [(2, 0, 1.0 + 0j)]
    if name in ["3d_xz", "dxz"]:
        return [(2, -1, 1/np.sqrt(2) + 0j), (2, 1, -1/np.sqrt(2) + 0j)]
    if name in ["3d_yz", "dyz"]:
        return [(2, -1, 1j/np.sqrt(2)), (2, 1, 1j/np.sqrt(2))]
    if name in ["3d_xy", "dxy"]:
        # d_xy ∝ i(Y_2,-2 - Y_2,2)/sqrt(2)
        return [(2, -2, 1j/np.sqrt(2)), (2, 2, -1j/np.sqrt(2))]
    if name in ["3d_x2y2", "dx2y2", "d_x2-y2"]:
        # d_(x^2-y^2) ∝ (Y_2,-2 + Y_2,2)/sqrt(2)
        return [(2, -2, 1/np.sqrt(2) + 0j), (2, 2, 1/np.sqrt(2) + 0j)]

    raise ValueError(f"Unknown orbital name: {name}")

def build_target_alm(l_ref: int):
    """Build coefficient array a_{l m} up to l_ref, then L2-normalize on S^2."""
    m0 = l_ref
    alm = np.zeros((l_ref + 1, 2*l_ref + 1), dtype=complex)

    terms = []
    if SUPERPOSITION:
        for orb, w in SUPERPOSITION:
            for (l, m, c) in list_orbital_coeffs(orb):
                terms.append((l, m, w * c))
    else:
        terms = list_orbital_coeffs(ORBITAL)

    for (l, m, c) in terms:
        if l > l_ref:
            raise ValueError("Increase L_REF to include this orbital component.")
        alm[l, m0 + m] += c

    for (l, m, c) in CUSTOM_TERMS:
        if l > l_ref:
            raise ValueError("Increase L_REF to include CUSTOM_TERMS.")
        if abs(m) > l:
            raise ValueError("Invalid (l,m) in CUSTOM_TERMS.")
        alm[l, m0 + m] += c

    norm2 = np.sum(np.abs(alm)**2)
    if norm2 > 0:
        alm /= np.sqrt(norm2)
    return alm

def reconstruct_from_alm(alm: np.ndarray, mu: np.ndarray, phi: np.ndarray, lmax: int, R_scale: float = 1.0):
    """
    Fast synthesis on (mu,phi) grid:
      psi(mu,phi) = sum_{l<=lmax} sum_{m=-l..l} a_{lm} Y_{lm}(mu,phi)
    with optional toy bulk kernel: a_{lm} -> a_{lm} * (R_scale)^l
    """
    l_ref = alm.shape[0] - 1
    m0 = l_ref
    n_theta = mu.size
    n_phi = phi.size

    Cpos = np.zeros((n_theta, lmax + 1), dtype=complex)  # Fourier coeff for +m
    Cneg = np.zeros((n_theta, lmax + 1), dtype=complex)  # Fourier coeff for -m (stored by m_abs)

    for l in range(lmax + 1):
        scale_l = (R_scale ** l)
        for m_abs in range(l + 1):
            T = T_lm_theta(l, m_abs, mu)
            # +m
            a_pos = alm[l, m0 + m_abs] * scale_l
            Cpos[:, m_abs] += a_pos * T
            if m_abs > 0:
                # -m handling (consistent with Y_{l,-m} relation)
                a_neg = alm[l, m0 - m_abs] * scale_l
                Cneg[:, m_abs] += a_neg * ((-1) ** m_abs) * T

    psi = np.zeros((n_theta, n_phi), dtype=complex)
    for i in range(n_theta):
        X = np.zeros(n_phi, dtype=complex)
        X[0] = Cpos[i, 0]
        for m_abs in range(1, lmax + 1):
            X[m_abs] = Cpos[i, m_abs]
            X[n_phi - m_abs] = Cneg[i, m_abs]
        psi[i, :] = np.fft.ifft(X) * n_phi  # undo numpy's 1/n
    return psi

def project_fft_gauss(psi: np.ndarray, mu: np.ndarray, w: np.ndarray, l_ref: int):
    """
    Fast analysis:
      - FFT in phi gives approx ∫ psi e^{-imphi} dphi
      - Gauss in mu gives ∫ ... dmu
    Returns alm (l_ref+1, 2*l_ref+1).
    """
    m0 = l_ref
    n_theta, n_phi = psi.shape
    fft = np.fft.fft(psi, axis=1)
    I = (2*pi / n_phi) * fft  # approximates phi-integral

    alm = np.zeros((l_ref + 1, 2*l_ref + 1), dtype=complex)
    for l in range(l_ref + 1):
        for m_abs in range(l + 1):
            T = T_lm_theta(l, m_abs, mu)
            alm[l, m0 + m_abs] = np.sum(w * I[:, m_abs] * T)
            if m_abs > 0:
                alm[l, m0 - m_abs] = np.sum(w * I[:, n_phi - m_abs] * ((-1) ** m_abs) * T)
    return alm

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else np.nan

def energy_coverage(alm: np.ndarray, lmax: int) -> float:
    l_ref = alm.shape[0] - 1
    m0 = l_ref
    tot = np.sum(np.abs(alm)**2)
    if tot <= 0:
        return np.nan
    s = 0.0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            s += np.abs(alm[l, m0 + m])**2
    return float(s / tot)

# -----------------------------------------------------------------------------
# Plot helpers (publication-friendly theta axis)
# -----------------------------------------------------------------------------
def save_map(data2d: np.ndarray, theta: np.ndarray, phi: np.ndarray, title: str, fname: str):
    plt.figure(figsize=(12, 3.8))
    plt.imshow(
        data2d,
        aspect="auto",
        origin="lower",
        extent=[phi[0], phi[-1] + (phi[1]-phi[0]), theta[0], theta[-1]],
    )
    plt.xlabel("phi")
    plt.ylabel("theta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=170)
    plt.close()

def save_phase_map(psi: np.ndarray, theta: np.ndarray, phi: np.ndarray, title: str, fname: str, amp_floor: float = 1e-8):
    amp = np.abs(psi)
    phase = np.angle(psi)
    phase = np.where(amp > amp_floor * np.max(amp), phase, np.nan)  # mask noisy phase
    save_map(phase, theta, phi, title, fname)

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
def main():
    print(f"S_bit={S_BIT:.6f}, HBMB l_max={HBMB_LMAX}")
    print(f"Config: L_REF={L_REF}, N_THETA_GAUSS={N_THETA_GAUSS}, N_PHI={N_PHI}")
    if SUPERPOSITION:
        print("Target: SUPERPOSITION")
        for orb, w in SUPERPOSITION:
            print(f"  - {orb} weight={w}")
    else:
        print(f"Target: ORBITAL={ORBITAL}")
    if CUSTOM_TERMS:
        print("Custom terms:")
        for (l, m, c) in CUSTOM_TERMS:
            print(f"  - (l={l}, m={m}) coeff={c}")

    # --- analysis grid
    mu_g, w_mu = np.polynomial.legendre.leggauss(N_THETA_GAUSS)
    phi_g = np.linspace(0, 2*pi, N_PHI, endpoint=False)

    # --- plot grid (uniform theta)
    theta_p = np.linspace(0, pi, N_THETA_PLOT, endpoint=False) + pi/(2*N_THETA_PLOT)
    mu_p = np.cos(theta_p)
    phi_p = np.linspace(0, 2*pi, N_PHI_PLOT, endpoint=False)

    # Build "true" coefficients and synthesize target horizon field (analysis grid)
    alm_true = build_target_alm(L_REF)

    t0 = time.time()
    psi_target_g = reconstruct_from_alm(alm_true, mu_g, phi_g, lmax=L_REF, R_scale=1.0)
    t_build = time.time() - t0
    print(f"Built target psi on horizon (analysis grid) in {t_build:.2f} s")

    # Project back to coefficients (analysis) and sanity check at L_REF
    t0 = time.time()
    alm_proj = project_fft_gauss(psi_target_g, mu_g, w_mu, l_ref=L_REF)
    t_proj = time.time() - t0
    print(f"Projection (FFT+Gauss) time: {t_proj:.2f} s")

    psi_check_g = reconstruct_from_alm(alm_proj, mu_g, phi_g, lmax=L_REF, R_scale=1.0)
    sanity_complex = rel_l2(psi_check_g, psi_target_g)
    sanity_abs = rel_l2(np.abs(psi_check_g), np.abs(psi_target_g))
    sanity_max = float(np.max(np.abs(psi_check_g - psi_target_g)))
    print(f"Sanity @ L_REF: relL2(complex)={sanity_complex:.3e}, relL2(|.|)={sanity_abs:.3e}, max(|.|)={sanity_max:.3e}")

    # Error + coverage vs cutoff on the horizon (analysis grid)
    errs_c, errs_abs, cov = [], [], []
    l_list = list(range(0, L_REF + 1))
    for lmax in l_list:
        psi_rec_g = reconstruct_from_alm(alm_proj, mu_g, phi_g, lmax=lmax, R_scale=1.0)
        errs_c.append(rel_l2(psi_rec_g, psi_target_g))
        errs_abs.append(rel_l2(np.abs(psi_rec_g), np.abs(psi_target_g)))
        cov.append(energy_coverage(alm_proj, lmax))

    # Plots: error / coverage
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(l_list, errs_c, label="Rel L2 (complex)")
    plt.plot(l_list, errs_abs, label="Rel L2 (|.|)")
    plt.axvline(HBMB_LMAX, linestyle="--", label=f"HBMB l_max={HBMB_LMAX}")
    plt.xlabel("l_max")
    plt.ylabel("Error")
    plt.title("Hydrogen horizon: reconstruction error vs cutoff (FAST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "hydrogen_horizon_error_vs_lmax.png", dpi=170)
    plt.close()

    plt.figure(figsize=(7.2, 4.4))
    plt.plot(l_list, cov, label="Coverage E(l_max)")
    plt.axvline(HBMB_LMAX, linestyle="--", label=f"HBMB l_max={HBMB_LMAX}")
    plt.xlabel("l_max")
    plt.ylabel("E(l_max)")
    plt.title("Hydrogen horizon: spectral coverage vs cutoff (FAST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "hydrogen_horizon_coverage_vs_lmax.png", dpi=170)
    plt.close()

    # Horizon maps on plot grid (correct theta axis)
    psi_target_p = reconstruct_from_alm(alm_true, mu_p, phi_p, lmax=L_REF, R_scale=1.0)
    save_map(np.abs(psi_target_p)**2, theta_p, phi_p, "Horizon: target |psi|^2", "horizon_target_abs2.png")
    save_phase_map(psi_target_p, theta_p, phi_p, "Horizon: target arg(psi)", "horizon_target_phase.png")

    for lmax in LMAX_SHOW:
        lmax = int(min(max(lmax, 0), L_REF))
        psi_rec_p = reconstruct_from_alm(alm_proj, mu_p, phi_p, lmax=lmax, R_scale=1.0)
        save_map(np.abs(psi_rec_p)**2, theta_p, phi_p, f"Horizon: recon |psi|^2 (lmax={lmax})", f"horizon_recon_abs2_lmax{lmax}.png")
        save_phase_map(psi_rec_p, theta_p, phi_p, f"Horizon: recon arg(psi) (lmax={lmax})", f"horizon_recon_phase_lmax{lmax}.png")

    # Toy BULK extension
    for r_bulk in BULK_R_LIST:
        scale = r_bulk / R_HORIZON
        psi_bulk_p = reconstruct_from_alm(alm_proj, mu_p, phi_p, lmax=L_REF, R_scale=scale)
        save_map(np.abs(psi_bulk_p)**2, theta_p, phi_p,
                 f"Toy bulk: target |psi|^2 at r={r_bulk:.2f} (kernel=(r/R)^l)",
                 f"bulk_target_abs2_r{r_bulk:.2f}.png")
        save_phase_map(psi_bulk_p, theta_p, phi_p,
                       f"Toy bulk: target arg(psi) at r={r_bulk:.2f}",
                       f"bulk_target_phase_r{r_bulk:.2f}.png")

        psi_bulk_hbmb_p = reconstruct_from_alm(alm_proj, mu_p, phi_p, lmax=HBMB_LMAX, R_scale=scale)
        save_map(np.abs(psi_bulk_hbmb_p)**2, theta_p, phi_p,
                 f"Toy bulk: recon |psi|^2 at r={r_bulk:.2f} (lmax={HBMB_LMAX})",
                 f"bulk_recon_abs2_r{r_bulk:.2f}_lmax{HBMB_LMAX}.png")

    # Save run metadata for reproducibility
    meta = {
        "S_BIT": float(S_BIT),
        "HBMB_LMAX": int(HBMB_LMAX),
        "L_REF": int(L_REF),
        "N_THETA_GAUSS": int(N_THETA_GAUSS),
        "N_PHI": int(N_PHI),
        "N_THETA_PLOT": int(N_THETA_PLOT),
        "N_PHI_PLOT": int(N_PHI_PLOT),
        "ORBITAL": ORBITAL,
        "SUPERPOSITION": [(o, str(w)) for (o, w) in SUPERPOSITION],
        "CUSTOM_TERMS": [(int(l), int(m), str(c)) for (l, m, c) in CUSTOM_TERMS],
        "timing_sec": {"build": float(t_build), "project": float(t_proj)},
        "sanity": {"relL2_complex": float(sanity_complex), "relL2_abs": float(sanity_abs), "max_abs": float(sanity_max)},
        "outputs_dir": str(OUTDIR.resolve()),
    }
    (OUTDIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved all outputs in: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
