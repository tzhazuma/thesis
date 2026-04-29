from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SOGKernel:
    alphas: np.ndarray
    weights: np.ndarray
    r_min: float
    r_max: float

    def evaluate_inv_r(self, r: np.ndarray) -> np.ndarray:
        r_sq = r**2
        return self.evaluate_from_r_sq(r_sq)

    def evaluate_from_r_sq(self, r_sq: np.ndarray) -> np.ndarray:
        vals = np.exp(-np.outer(r_sq, self.alphas)) @ self.weights
        return vals

    def force_prefactor(self, r_sq: np.ndarray) -> np.ndarray:
        weighted = 2.0 * self.alphas * self.weights
        return np.exp(-np.outer(r_sq, self.alphas)) @ weighted

    def fit_report(self, n_samples: int = 512) -> dict[str, float]:
        r = np.geomspace(self.r_min, self.r_max, n_samples)
        approx = self.evaluate_inv_r(r)
        target = 1.0 / r
        abs_err = np.abs(approx - target)
        rel_err = abs_err / target
        return {
            "rmse": float(np.sqrt(np.mean((approx - target) ** 2))),
            "max_abs_error": float(np.max(abs_err)),
            "mean_relative_error": float(np.mean(rel_err)),
            "max_relative_error": float(np.max(rel_err)),
        }


def fit_sog_kernel(
    r_min: float,
    r_max: float,
    n_terms: int = 12,
    n_samples: int = 4096,
) -> SOGKernel:
    if r_min <= 0.0:
        raise ValueError("r_min must be positive")
    if r_max <= r_min:
        raise ValueError("r_max must be larger than r_min")

    alpha_min = 0.5 / (r_max * r_max)
    alpha_max = 8.0 / (r_min * r_min)
    alphas = np.geomspace(alpha_min, alpha_max, n_terms)

    r = np.geomspace(r_min, r_max, n_samples)
    design = np.exp(-np.outer(r * r, alphas))
    target = 1.0 / r

    weights, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    weights = np.clip(weights, 0.0, None)

    if np.all(weights == 0.0):
        raise RuntimeError("SOG fitting failed: all weights are zero")

    return SOGKernel(alphas=alphas, weights=weights, r_min=r_min, r_max=r_max)
