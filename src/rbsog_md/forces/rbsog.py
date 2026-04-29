from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbsog_md.forces.base import ForceResult
from rbsog_md.forces.neighbor import NeighborListCoulombSolver
from rbsog_md.forces.sog import SOGKernel, fit_sog_kernel
from rbsog_md.system import ParticleSystem
from rbsog_md.utils import minimum_image

try:
    from numba import njit  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    njit = None


_NUMBA_AVAILABLE = njit is not None


if njit is not None:

    @njit(cache=True)
    def _accumulate_sog_long_range_numba(
        n_particles: int,
        i_idx: np.ndarray,
        j_idx: np.ndarray,
        disp: np.ndarray,
        r_sq: np.ndarray,
        sample_weights: np.ndarray,
        charges: np.ndarray,
        alphas: np.ndarray,
        kernel_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        forces = np.zeros((n_particles, 3), dtype=np.float64)
        potential = 0.0
        virial = 0.0

        for k in range(i_idx.shape[0]):
            i = i_idx[k]
            j = j_idx[k]
            r2 = r_sq[k]

            qij = charges[i] * charges[j]

            value = 0.0
            prefactor = 0.0
            for t in range(alphas.shape[0]):
                alpha = alphas[t]
                w = kernel_weights[t]
                exp_term = np.exp(-alpha * r2)
                value += w * exp_term
                prefactor += 2.0 * alpha * w * exp_term

            weighted_pref = sample_weights[k] * qij * prefactor
            dx = disp[k, 0]
            dy = disp[k, 1]
            dz = disp[k, 2]

            fx = weighted_pref * dx
            fy = weighted_pref * dy
            fz = weighted_pref * dz

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz

            weighted_pair = 0.5 * sample_weights[k] * qij
            potential += weighted_pair * value
            virial += weighted_pair * prefactor * r2

        return forces, potential, virial

else:

    def _accumulate_sog_long_range_numba(
        n_particles: int,
        i_idx: np.ndarray,
        j_idx: np.ndarray,
        disp: np.ndarray,
        r_sq: np.ndarray,
        sample_weights: np.ndarray,
        charges: np.ndarray,
        alphas: np.ndarray,
        kernel_weights: np.ndarray,
    ) -> tuple[np.ndarray, float, float]:
        del n_particles, i_idx, j_idx, disp, r_sq, sample_weights, charges, alphas, kernel_weights
        raise RuntimeError("Numba is not available")


@dataclass
class RBSOGConfig:
    batch_size: int = 100
    cutoff_short: float = 1.2
    sog_terms: int = 12
    r_min: float = 0.15
    importance_exponent: float = 1.0
    neighbor_skin: float = 0.3
    neighbor_rebuild_interval: int = 10
    use_numba: bool = True


class RBSOGSolver:
    name = "rbsog"

    def __init__(self, config: RBSOGConfig) -> None:
        self.config = config
        self.short_range_solver = NeighborListCoulombSolver(
            cutoff=config.cutoff_short,
            skin=config.neighbor_skin,
            rebuild_interval=config.neighbor_rebuild_interval,
            use_numba=config.use_numba,
        )
        self.long_range_jit_enabled = bool(config.use_numba and _NUMBA_AVAILABLE)
        self.kernel: SOGKernel | None = None
        self.kernel_box: np.ndarray | None = None
        self._last_diagnostics: dict[str, float] = {}

    def _ensure_kernel(self, box: np.ndarray) -> None:
        if self.kernel is not None and self.kernel_box is not None and np.allclose(box, self.kernel_box):
            return
        r_max = 0.5 * float(np.min(box))
        self.kernel = fit_sog_kernel(
            r_min=self.config.r_min,
            r_max=r_max,
            n_terms=self.config.sog_terms,
        )
        self.kernel_box = box.copy()

    def compute(
        self,
        system: ParticleSystem,
        rng: np.random.Generator | None = None,
    ) -> ForceResult:
        if rng is None:
            rng = np.random.default_rng()

        self._ensure_kernel(system.box)
        assert self.kernel is not None

        short_result = self.short_range_solver.compute(system)
        forces = short_result.forces.copy()
        potential = short_result.potential
        virial = short_result.virial

        positions = system.positions
        charges = system.charges
        n_particles = system.n_particles
        if n_particles < 2:
            return ForceResult(forces=forces, potential=potential, virial=virial)

        m_samples = max(self.config.batch_size * n_particles, 1)
        cutoff_sq = self.config.cutoff_short * self.config.cutoff_short
        r_min_sq = self.config.r_min * self.config.r_min

        q_abs = np.abs(charges) + 1e-6
        proposal = q_abs**self.config.importance_exponent
        proposal /= np.sum(proposal)

        i_all = rng.choice(n_particles, size=m_samples, p=proposal)
        j_all = rng.choice(n_particles, size=m_samples, p=proposal)

        mask_not_self = i_all != j_all
        if not np.any(mask_not_self):
            self._last_diagnostics = {
                "accepted_samples": 0.0,
                "requested_samples": float(m_samples),
                "acceptance_ratio": 0.0,
                "jit_enabled": float(self.short_range_solver.jit_enabled),
            }
            return ForceResult(forces=forces, potential=potential, virial=virial)

        i_idx = i_all[mask_not_self]
        j_idx = j_all[mask_not_self]

        displacement = minimum_image(positions[j_idx] - positions[i_idx], system.box)
        r_sq = np.einsum("ij,ij->i", displacement, displacement)
        mask_long = (r_sq > cutoff_sq) & (r_sq > r_min_sq)
        if not np.any(mask_long):
            self._last_diagnostics = {
                "accepted_samples": 0.0,
                "requested_samples": float(m_samples),
                "acceptance_ratio": 0.0,
                "jit_enabled": float(self.short_range_solver.jit_enabled),
            }
            return ForceResult(forces=forces, potential=potential, virial=virial)

        i_valid = i_idx[mask_long]
        j_valid = j_idx[mask_long]
        disp_valid = displacement[mask_long]
        r_sq_valid = r_sq[mask_long]

        pair_prob = proposal[i_valid] * proposal[j_valid]
        sample_weights = 1.0 / (m_samples * pair_prob)

        if self.long_range_jit_enabled:
            long_forces, long_potential, long_virial = _accumulate_sog_long_range_numba(
                n_particles=int(n_particles),
                i_idx=np.ascontiguousarray(i_valid, dtype=np.int64),
                j_idx=np.ascontiguousarray(j_valid, dtype=np.int64),
                disp=np.ascontiguousarray(disp_valid, dtype=np.float64),
                r_sq=np.ascontiguousarray(r_sq_valid, dtype=np.float64),
                sample_weights=np.ascontiguousarray(sample_weights, dtype=np.float64),
                charges=np.ascontiguousarray(charges, dtype=np.float64),
                alphas=np.ascontiguousarray(self.kernel.alphas, dtype=np.float64),
                kernel_weights=np.ascontiguousarray(self.kernel.weights, dtype=np.float64),
            )
            forces += long_forces
            potential += float(long_potential)
            virial += float(long_virial)
        else:
            pref = self.kernel.force_prefactor(r_sq_valid)
            qij = charges[i_valid] * charges[j_valid]
            pair_force = (qij * pref)[:, None] * disp_valid

            scaled_force = sample_weights[:, None] * pair_force
            np.add.at(forces, i_valid, scaled_force)
            np.add.at(forces, j_valid, -scaled_force)

            pair_potential = qij * self.kernel.evaluate_from_r_sq(r_sq_valid)
            potential += float(np.sum(0.5 * sample_weights * pair_potential))
            virial += float(np.sum(0.5 * sample_weights * np.einsum("ij,ij->i", disp_valid, pair_force)))

        self._last_diagnostics = {
            "accepted_samples": float(i_valid.size),
            "requested_samples": float(m_samples),
            "acceptance_ratio": float(i_valid.size / m_samples),
            "jit_enabled": float(self.short_range_solver.jit_enabled),
            "long_range_jit_enabled": float(self.long_range_jit_enabled),
        }

        return ForceResult(forces=forces, potential=potential, virial=virial)

    def kernel_report(self) -> dict[str, float]:
        if self.kernel is None:
            return {}
        report = self.kernel.fit_report()
        report.update(self._last_diagnostics)
        report.update(
            {
                "neighbor_skin": float(self.config.neighbor_skin),
                "neighbor_rebuild_interval": float(self.config.neighbor_rebuild_interval),
            }
        )
        return report
