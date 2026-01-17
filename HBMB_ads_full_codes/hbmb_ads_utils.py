"""HBMB AdS toy utilities.

- Angular grid on S^2 using Gauss-Legendre (mu=cos(theta)) + uniform phi.
- Spherical harmonics from scipy.special.sph_harm (complex Y_lm).
- Boundary projection to coefficients c_{lm} and truncated reconstructions.
- AdS-like radial kernel (toy) normalized at r=Rb.

Notes
-----
scipy.special.sph_harm signature: sph_harm(m, l, phi, theta).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import sph_harm, eval_legendre


@dataclass
class GridS2:
    """Quadrature grid on S^2.

    We integrate using:
        int_{S^2} f(θ,φ) dΩ = int_{0}^{2π} dφ int_{-1}^{1} dμ f(μ,φ)
    where μ = cosθ.
    """

    n_mu: int
    n_phi: int

    mu: np.ndarray          # shape (n_mu,)
    w_mu: np.ndarray        # shape (n_mu,)
    theta: np.ndarray       # shape (n_mu,)
    phi: np.ndarray         # shape (n_phi,)
    dphi: float

    # Meshes (broadcast-friendly)
    TH: np.ndarray          # shape (n_mu, 1)
    PH: np.ndarray          # shape (1, n_phi)

    @staticmethod
    def build(n_mu: int, n_phi: int) -> "GridS2":
        mu, w_mu = leggauss(n_mu)  # mu in [-1,1]
        theta = np.arccos(mu)
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
        dphi = 2.0 * np.pi / n_phi
        TH = theta[:, None]
        PH = phi[None, :]
        return GridS2(
            n_mu=n_mu,
            n_phi=n_phi,
            mu=mu,
            w_mu=w_mu,
            theta=theta,
            phi=phi,
            dphi=dphi,
            TH=TH,
            PH=PH,
        )

    def dOmega_weights(self) -> np.ndarray:
        """Return quadrature weights for the 2D grid, shape (n_mu, n_phi)."""
        # dΩ = dφ dμ, with Gauss weights for μ and uniform dφ
        return (self.w_mu[:, None]) * self.dphi

    def unit_vectors(self) -> np.ndarray:
        """Return unit vectors n(θ,φ) for all grid points, shape (N,3)."""
        sinT = np.sin(self.TH)
        cosT = np.cos(self.TH)
        cosP = np.cos(self.PH)
        sinP = np.sin(self.PH)
        x = (sinT * cosP).reshape(-1)
        y = (sinT * sinP).reshape(-1)
        z = (cosT * np.ones_like(self.PH)).reshape(-1)
        return np.stack([x, y, z], axis=1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ads_toy_kernel_f_l(r: float, l: int, Rb: float, L: float, Delta: float) -> float:
    """AdS-like toy radial kernel f_l(r), normalized so f_l(Rb)=1."""
    # (r/Rb)^l factor + conformal damping
    a = (r / Rb) ** l
    b = ((1.0 + (r * r) / (L * L)) / (1.0 + (Rb * Rb) / (L * L))) ** (-0.5 * Delta)
    return float(a * b)


def precompute_Ylm(grid: GridS2, l_max: int) -> Dict[Tuple[int, int], np.ndarray]:
    """Precompute complex spherical harmonics Y_lm on the grid for all l<=l_max.

    Returns a dict keyed by (l,m) -> array shape (n_mu, n_phi).
    """
    Y: Dict[Tuple[int, int], np.ndarray] = {}
    # sph_harm uses (m,l,phi,theta)
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            Y[(l, m)] = sph_harm(m, l, grid.PH, grid.TH)
    return Y


def project_to_clm(grid: GridS2, psi: np.ndarray, Y: Dict[Tuple[int, int], np.ndarray], l_max: int) -> Dict[Tuple[int, int], complex]:
    """Project psi(θ,φ) onto Y_lm up to l_max, returning c_{lm}."""
    w = grid.dOmega_weights()
    c: Dict[Tuple[int, int], complex] = {}
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            # integral Y* psi dΩ
            integrand = np.conjugate(Y[(l, m)]) * psi
            c[(l, m)] = complex(np.sum(integrand * w))
    return c


def reconstruct_from_clm(grid: GridS2, c: Dict[Tuple[int, int], complex], Y: Dict[Tuple[int, int], np.ndarray], l_max: int, radial_factors: Dict[int, float] | None = None) -> np.ndarray:
    """Reconstruct psi from coefficients up to l_max.

    radial_factors: optional dict l -> f_l(r) scaling applied to each l-shell.
    """
    out = np.zeros((grid.n_mu, grid.n_phi), dtype=np.complex128)
    for l in range(l_max + 1):
        fl = 1.0
        if radial_factors is not None:
            fl = radial_factors.get(l, 1.0)
        for m in range(-l, l + 1):
            out += c[(l, m)] * fl * Y[(l, m)]
    return out


def rel_L2(grid: GridS2, a: np.ndarray, b: np.ndarray) -> float:
    """Relative L2 error ||a-b||/||b|| on S^2."""
    w = grid.dOmega_weights()
    num = np.sum(np.abs(a - b) ** 2 * w)
    den = np.sum(np.abs(b) ** 2 * w)
    if den == 0.0:
        return 0.0 if num == 0.0 else float('inf')
    return float(np.sqrt(num / den))


def coverage_energy(c: Dict[Tuple[int, int], complex], l_max: int, l_ref: int) -> float:
    """Energy coverage up to l_max relative to l_ref: sum|c_lm|^2."""
    def shell_sum(L: int) -> float:
        s = 0.0
        for l in range(L + 1):
            for m in range(-l, l + 1):
                s += abs(c[(l, m)]) ** 2
        return float(s)

    num = shell_sum(l_max)
    den = shell_sum(l_ref)
    if den == 0.0:
        return 0.0
    return float(num / den)


def blob_on_s2(grid: GridS2, theta0: float, phi0: float, sigma: float) -> np.ndarray:
    """Gaussian-like blob exp(-(gamma/sigma)^2) on the sphere."""
    # cos(gamma) = cosθ cosθ0 + sinθ sinθ0 cos(φ-φ0)
    cosg = np.cos(grid.TH) * math.cos(theta0) + np.sin(grid.TH) * math.sin(theta0) * np.cos(grid.PH - phi0)
    cosg = np.clip(cosg, -1.0, 1.0)
    gamma = np.arccos(cosg)
    return np.exp(- (gamma / sigma) ** 2).astype(np.complex128)


def build_legendre_kernel(cos_gamma: np.ndarray, f_l: np.ndarray, l_max: int) -> np.ndarray:
    """Compute K(cosγ) = sum_{l<=lmax} f_l(r) (2l+1)/(4π) P_l(cosγ).

    Parameters
    ----------
    cos_gamma : array
        cosγ values for all source points.
    f_l : array
        radial factors f_l(r) for l=0..l_max (shape (l_max+1,)).
    """
    K = np.zeros_like(cos_gamma, dtype=np.float64)
    for l in range(l_max + 1):
        K += f_l[l] * (2 * l + 1) / (4.0 * np.pi) * eval_legendre(l, cos_gamma)
    return K
