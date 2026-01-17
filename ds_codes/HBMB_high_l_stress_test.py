# HBMB_high_l_stress_test.py
# High-l stress test for HBMB truncation on S^2:
# show that reconstruction fails for l_max < L_HBMB and succeeds for l_max >= L_HBMB.
#
# Dependencies: numpy, scipy, matplotlib
#
# Outputs: figures + run_metadata.json in OUTDIR

import json
import time
from math import pi
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


# -----------------------------------------------------------------------------
# User settings
# -----------------------------------------------------------------------------
OUTDIR = Path("figs_hbmb_high_l_stress")
OUTDIR.mkdir(parents=True, exist_ok=True)

# HBMB parameters (match your earlier runs: S_bit ~ 490.87 -> L_HBMB = 21)
R_HORIZON = 1.0
LP_EFF = 0.08
S_BIT = pi * (R_HORIZON / LP_EFF) ** 2
L_HBMB = int(np.floor(np.sqrt(S_BIT) - 1.0))

# Spectral & quadrature settings
L_REF = 60                 # reference l_max for "ground truth"
N_THETA_GAUSS = 140        # Gauss-Legendre nodes in mu=cos(theta)
N_PHI = 320                # must satisfy N_PHI >= 2*L_REF+1 to avoid aliasing

# Plot grid (publication-friendly, uniform theta)
N_THETA_PLOT = 180
N_PHI_PLOT = 360

# Show these truncations as maps
LMAX_MAPS = [10, L_HBMB - 1, L_HBMB, 30]

# High-l stress term (hydrogenic allowed if n >= l+1; angular part is Y_{l,m} anyway)
L_HIGH = L_HBMB
M_HIGH = 7
AMP_HIGH = 0.45 * np.exp(1j * 0.2)   # tune 0.25..0.8 for more/less dramatic effect

# Base low-l content (kept modest so the high-l term is visible)
# Pick m!=0 components to create phi-structure.
BASE_SUPERPOSITION = [
    ("1s", 1.0 + 0.0j),
    ("2p_x", 0.60 * np.exp(1j * 0.6)),
    ("3d_xy", 0.35 * np.exp(1j * 1.2)),
]


# -----------------------------------------------------------------------------
# Spherical-harmonics helpers (complex Y_lm)
# Convention:
#   Y_{l,m}(mu,phi) = N_{l,m} P_l^m(mu) e^{i m phi}   for m>=0
#   Y_{l,-m} = (-1)^m conj(Y_{l,m})
# SciPy lpmv includes the Condon–Shortley phase.
# -----------------------------------------------------------------------------
def N_lm(l: int, m_abs: int) -> float:
    return float(np.sqrt((2*l + 1) / (4*pi) *
                         special.factorial(l - m_abs) / special.factorial(l + m_abs)))

def T_lm_theta(l: int, m_abs: int, mu: np.ndarray) -> np.ndarray:
    return N_lm(l, m_abs) * special.lpmv(m_abs, l, mu)

def list_orbital_coeffs(name: str):
    """Return (l,m,coeff) list for common *real* orbitals expressed in complex Y_lm basis."""
    name = name.strip().lower()

    if name == "1s":
        return [(0, 0, 1.0 + 0.0j)]

    # p orbitals (l=1)
    if name in ["2p_z", "pz"]:
        return [(1, 0, 1.0 + 0.0j)]
    if name in ["2p_x", "px"]:
        # p_x ∝ (Y_1,-1 - Y_1,1)/sqrt(2)
        return [(1, -1, 1/np.sqrt(2) + 0j), (1, 1, -1/np.sqrt(2) + 0j)]
    if name in ["2p_y", "py"]:
        # p_y ∝ i(Y_1,-1 + Y_1,1)/sqrt(2)
        return [(1, -1, 1j/np.sqrt(2)), (1, 1, 1j/np.sqrt(2))]

    # d orbitals (l=2)
    if name in ["3d_z2", "dz2"]:
        return [(2, 0, 1.0 + 0.0j)]
    if name in ["3d_xy", "dxy"]:
        # d_xy ∝ i(Y_2,-2 - Y_2,2)/sqrt(2)
        return [(2, -2, 1j/np.sqrt(2)), (2, 2, -1j/np.sqrt(2))]
    if name in ["3d_x2y2", "dx2y2", "d_x2-y2"]:
        # d_(x^2-y^2) ∝ (Y_2,-2 + Y_2,2)/sqrt(2)
        return [(2, -2, 1/np.sqrt(2) + 0j), (2, 2, 1/np.sqrt(2) + 0j)]

    raise ValueError(f"Unknown orbital name: {name}")

def build_target_alm(l_ref: int) -> np.ndarray:
    """
    Build 'true' coefficients a_{l,m} up to l_ref and L2-normalize:
      sum_{l,m} |a_{l,m}|^2 = 1.
    """
    m0 = l_ref
    alm = np.zeros((l_ref + 1, 2*l_ref + 1), dtype=complex)

    # Base low-l orbitals
    for orb, w in BASE_SUPERPOSITION:
        for (l, m, c) in list_orbital_coeffs(orb):
            if l > l_ref:
                raise ValueError("Increase L_REF to include this component.")
            alm[l, m0 + m] += w * c

    # High-l stress mode (represents e.g. hydrogenic n=22, l=21, m=7 angular part)
    if L_HIGH > l_ref:
        raise ValueError("Increase L_REF to include the high-l term.")
    alm[L_HIGH, m0 + M_HIGH] += AMP_HIGH

    # Normalize in coefficient space (orthonormal Y_lm basis)
    norm2 = np.sum(np.abs(alm)**2)
    alm /= np.sqrt(max(norm2, 1e-300))
    return alm

def synthesize_from_alm(alm: np.ndarray, mu: np.ndarray, phi: np.ndarray, lmax: int, R_scale: float = 1.0) -> np.ndarray:
    """
    Fast synthesis using FFT in phi.
    If R_scale != 1, applies a toy bulk kernel: a_{l,m} -> a_{l,m} * (R_scale)^l.
    """
    l_ref = alm.shape[0] - 1
    m0 = l_ref
    n_theta = mu.size
    n_phi = phi.size

    # Fourier coefficients per theta row:
    Cpos = np.zeros((n_theta, lmax + 1), dtype=complex)  # for +m
    Cneg = np.zeros((n_theta, lmax + 1), dtype=complex)  # for -m, stored by m_abs

    for l in range(lmax + 1):
        scale_l = (R_scale ** l)
        for m_abs in range(l + 1):
            T = T_lm_theta(l, m_abs, mu)
            # +m
            Cpos[:, m_abs] += (alm[l, m0 + m_abs] * scale_l) * T
            if m_abs > 0:
                # -m term includes (-1)^m factor due to Y_{l,-m}
                Cneg[:, m_abs] += (alm[l, m0 - m_abs] * scale_l) * ((-1) ** m_abs) * T

    psi = np.zeros((n_theta, n_phi), dtype=complex)
    for i in range(n_theta):
        X = np.zeros(n_phi, dtype=complex)
        X[0] = Cpos[i, 0]
        for m_abs in range(1, lmax + 1):
            X[m_abs] = Cpos[i, m_abs]
            X[n_phi - m_abs] = Cneg[i, m_abs]
        psi[i, :] = np.fft.ifft(X) * n_phi  # undo numpy's 1/n factor
    return psi

def project_fft_gauss(psi: np.ndarray, mu: np.ndarray, w: np.ndarray, l_ref: int) -> np.ndarray:
    """
    Fast analysis:
      - FFT in phi gives approx phi-integrals
      - Gauss-Legendre in mu integrates over mu in [-1,1]
    """
    m0 = l_ref
    n_theta, n_phi = psi.shape

    fft = np.fft.fft(psi, axis=1)
    I = (2*pi / n_phi) * fft  # approximates ∫ psi e^{-i m phi} dphi

    alm = np.zeros((l_ref + 1, 2*l_ref + 1), dtype=complex)
    for l in range(l_ref + 1):
        for m_abs in range(l + 1):
            T = T_lm_theta(l, m_abs, mu)
            # +m
            alm[l, m0 + m_abs] = np.sum(w * I[:, m_abs] * T)
            if m_abs > 0:
                # -m: Y_{l,-m}^* includes (-1)^m factor
                alm[l, m0 - m_abs] = np.sum(w * I[:, n_phi - m_abs] * ((-1) ** m_abs) * T)
    return alm

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    num = np.linalg.norm((a - b).ravel())
    den = np.linalg.norm(b.ravel())
    return float(num / den) if den > 0 else np.nan

def coverage_E(alm: np.ndarray, lmax: int) -> float:
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
# Plot helpers
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
    phase = np.where(amp > amp_floor * np.max(amp), phase, np.nan)
    save_map(phase, theta, phi, title, fname)

def save_lm_power_heatmap(alm: np.ndarray, l_show: int, title: str, fname: str):
    """
    Heatmap of |a_{l,m}|^2 for 0<=l<=l_show, -l_show<=m<=l_show.
    This is the HBMB analog of a "mode selection grid".
    """
    l_ref = alm.shape[0] - 1
    m0 = l_ref
    l_show = int(min(l_show, l_ref))
    m_show = l_show

    mat = np.full((l_show + 1, 2*m_show + 1), np.nan, dtype=float)
    for l in range(l_show + 1):
        for m in range(-l, l + 1):
            mat[l, m + m_show] = float(np.abs(alm[l, m0 + m])**2)

    plt.figure(figsize=(10.5, 5.0))
    plt.imshow(mat, aspect="auto", origin="lower",
               extent=[-m_show - 0.5, m_show + 0.5, -0.5, l_show + 0.5])
    plt.xlabel("m")
    plt.ylabel("l")
    plt.title(title)
    plt.axhline(L_HBMB, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTDIR / fname, dpi=170)
    plt.close()

def main():
    print(f"S_bit={S_BIT:.6f}, HBMB l_max={L_HBMB}")
    print(f"Config: L_REF={L_REF}, N_THETA_GAUSS={N_THETA_GAUSS}, N_PHI={N_PHI}")
    print(f"High-l term: l={L_HIGH}, m={M_HIGH}, amp={AMP_HIGH}")

    # Analysis grid: Gauss in mu, uniform phi
    mu_g, w_g = np.polynomial.legendre.leggauss(N_THETA_GAUSS)
    phi_g = np.linspace(0.0, 2*pi, N_PHI, endpoint=False)

    # Plot grid: uniform theta
    theta_p = np.linspace(0.0, pi, N_THETA_PLOT, endpoint=False) + pi/(2*N_THETA_PLOT)
    mu_p = np.cos(theta_p)
    phi_p = np.linspace(0.0, 2*pi, N_PHI_PLOT, endpoint=False)

    # Build target coefficients (ground truth)
    alm_true = build_target_alm(L_REF)

    # Heatmap of target |a_lm|^2 (nice "mode selection" figure)
    save_lm_power_heatmap(alm_true, l_show=min(35, L_REF),
                          title="Target spectral power |a_{l,m}|^2 (ground truth)",
                          fname="target_lm_power_heatmap.png")

    # Synthesize target on analysis grid and project back (FAST)
    t0 = time.time()
    psi_target_g = synthesize_from_alm(alm_true, mu_g, phi_g, lmax=L_REF)
    t_build = time.time() - t0

    t0 = time.time()
    alm_proj = project_fft_gauss(psi_target_g, mu_g, w_g, l_ref=L_REF)
    t_proj = time.time() - t0

    # Sanity: reconstruct at L_REF on analysis grid
    psi_check_g = synthesize_from_alm(alm_proj, mu_g, phi_g, lmax=L_REF)
    sanity_c = rel_l2(psi_check_g, psi_target_g)
    sanity_abs = rel_l2(np.abs(psi_check_g), np.abs(psi_target_g))
    sanity_max = float(np.max(np.abs(psi_check_g - psi_target_g)))

    print(f"Timing: build={t_build:.2f} s, project={t_proj:.2f} s")
    print(f"Sanity @ L_REF: relL2(complex)={sanity_c:.3e}, relL2(|.|)={sanity_abs:.3e}, max(|.|)={sanity_max:.3e}")

    # Error + coverage vs lmax (on plot grid for consistent visuals)
    psi_target_p = synthesize_from_alm(alm_true, mu_p, phi_p, lmax=L_REF)

    lvals = np.arange(0, L_REF + 1)
    err_c, err_abs, cov = [], [], []
    for lmax in lvals:
        psi_rec_p = synthesize_from_alm(alm_proj, mu_p, phi_p, lmax=int(lmax))
        err_c.append(rel_l2(psi_rec_p, psi_target_p))
        err_abs.append(rel_l2(np.abs(psi_rec_p), np.abs(psi_target_p)))
        cov.append(coverage_E(alm_proj, int(lmax)))

    # Plot: error vs lmax
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(lvals, err_c, label="Rel L2 (complex)")
    plt.plot(lvals, err_abs, label="Rel L2 (|.|)")
    plt.axvline(L_HBMB - 1, linestyle="--", label=f"l_max={L_HBMB-1}")
    plt.axvline(L_HBMB, linestyle="--", label=f"HBMB l_max={L_HBMB}")
    plt.xlabel("l_max")
    plt.ylabel("Error")
    plt.title("High-l stress test: reconstruction error vs cutoff (FAST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "stress_error_vs_lmax.png", dpi=170)
    plt.close()

    # Plot: coverage vs lmax
    plt.figure(figsize=(7.2, 4.4))
    plt.plot(lvals, cov, label="Coverage E(l_max)")
    plt.axvline(L_HBMB - 1, linestyle="--", label=f"l_max={L_HBMB-1}")
    plt.axvline(L_HBMB, linestyle="--", label=f"HBMB l_max={L_HBMB}")
    plt.xlabel("l_max")
    plt.ylabel("E(l_max)")
    plt.title("High-l stress test: spectral coverage vs cutoff (FAST)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "stress_coverage_vs_lmax.png", dpi=170)
    plt.close()

    # Horizon target maps
    save_map(np.abs(psi_target_p)**2, theta_p, phi_p, "Horizon target: |psi|^2", "horizon_target_abs2.png")
    save_phase_map(psi_target_p, theta_p, phi_p, "Horizon target: arg(psi)", "horizon_target_phase.png")

    # Recon maps for selected lmax + difference maps
    for lmax in sorted(set(int(v) for v in LMAX_MAPS)):
        lmax = int(max(0, min(L_REF, lmax)))
        psi_rec_p = synthesize_from_alm(alm_proj, mu_p, phi_p, lmax=lmax)

        save_map(np.abs(psi_rec_p)**2, theta_p, phi_p, f"Horizon recon: |psi|^2 (lmax={lmax})",
                 f"horizon_recon_abs2_lmax{lmax}.png")
        save_phase_map(psi_rec_p, theta_p, phi_p, f"Horizon recon: arg(psi) (lmax={lmax})",
                       f"horizon_recon_phase_lmax{lmax}.png")

        diff = np.abs(psi_target_p)**2 - np.abs(psi_rec_p)**2
        save_map(diff, theta_p, phi_p, f"Horizon diff: |psi|^2_target - |psi|^2_recon (lmax={lmax})",
                 f"horizon_diff_abs2_lmax{lmax}.png")

    # Optional toy-bulk: apply mode-diagonal kernel (r/R)^l
    r_list = [1.0, 0.7, 0.4]
    for r in r_list:
        scale = r / R_HORIZON
        psi_bulk = synthesize_from_alm(alm_proj, mu_p, phi_p, lmax=L_REF, R_scale=scale)
        save_map(np.abs(psi_bulk)**2, theta_p, phi_p,
                 f"Toy bulk: |psi|^2 at r={r:.2f} (kernel=(r/R)^l)", f"bulk_abs2_r{r:.2f}.png")

    # Save metadata
    meta = {
        "S_BIT": float(S_BIT),
        "L_HBMB": int(L_HBMB),
        "L_REF": int(L_REF),
        "N_THETA_GAUSS": int(N_THETA_GAUSS),
        "N_PHI": int(N_PHI),
        "N_THETA_PLOT": int(N_THETA_PLOT),
        "N_PHI_PLOT": int(N_PHI_PLOT),
        "BASE_SUPERPOSITION": [(o, str(w)) for (o, w) in BASE_SUPERPOSITION],
        "HIGH_L_TERM": {"n_example": int(L_HIGH + 1), "l": int(L_HIGH), "m": int(M_HIGH), "amp": str(AMP_HIGH)},
        "timing_sec": {"build": float(t_build), "project": float(t_proj)},
        "sanity": {"relL2_complex": float(sanity_c), "relL2_abs": float(sanity_abs), "max_abs": float(sanity_max)},
        "outputs_dir": str(OUTDIR.resolve()),
    }
    (OUTDIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved outputs to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
