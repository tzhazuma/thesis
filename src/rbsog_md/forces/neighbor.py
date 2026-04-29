from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbsog_md.forces.base import ForceResult
from rbsog_md.system import ParticleSystem
from rbsog_md.utils import minimum_image

try:
    from numba import njit  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - optional dependency
    njit = None


_NUMBA_AVAILABLE = njit is not None


if njit is not None:

    @njit(cache=True)
    def _compute_short_range_numba(
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
        pairs: np.ndarray,
        cutoff_sq: float,
        softening_sq: float,
    ) -> tuple[np.ndarray, float, float]:
        n_particles = positions.shape[0]
        forces = np.zeros((n_particles, 3), dtype=np.float64)
        potential = 0.0
        virial = 0.0

        box_x = box[0]
        box_y = box[1]
        box_z = box[2]

        for k in range(pairs.shape[0]):
            i = pairs[k, 0]
            j = pairs[k, 1]

            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]

            dx -= box_x * np.round(dx / box_x)
            dy -= box_y * np.round(dy / box_y)
            dz -= box_z * np.round(dz / box_z)

            r_sq = dx * dx + dy * dy + dz * dz
            if r_sq <= softening_sq or r_sq > cutoff_sq:
                continue

            inv_r = 1.0 / np.sqrt(r_sq)
            inv_r3 = inv_r * inv_r * inv_r
            qij = charges[i] * charges[j]

            fx = qij * inv_r3 * dx
            fy = qij * inv_r3 * dy
            fz = qij * inv_r3 * dz

            forces[i, 0] += fx
            forces[i, 1] += fy
            forces[i, 2] += fz
            forces[j, 0] -= fx
            forces[j, 1] -= fy
            forces[j, 2] -= fz

            potential += qij * inv_r
            virial += dx * fx + dy * fy + dz * fz

        return forces, potential, virial

else:
    def _compute_short_range_numba(
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
        pairs: np.ndarray,
        cutoff_sq: float,
        softening_sq: float,
    ) -> tuple[np.ndarray, float, float]:
        del positions, charges, box, pairs, cutoff_sq, softening_sq
        raise RuntimeError("Numba is not available")


@dataclass
class VerletNeighborList:
    cutoff: float
    skin: float = 0.3
    rebuild_interval: int = 10

    def __post_init__(self) -> None:
        if self.cutoff <= 0.0:
            raise ValueError("cutoff must be positive")
        if self.skin < 0.0:
            raise ValueError("skin must be non-negative")
        if self.rebuild_interval < 1:
            raise ValueError("rebuild_interval must be >= 1")

        self._pairs: np.ndarray | None = None
        self._ref_positions: np.ndarray | None = None
        self._ref_box: np.ndarray | None = None
        self._steps_since_build: int = 0

    @property
    def pair_cutoff(self) -> float:
        return self.cutoff + self.skin

    def _needs_rebuild(self, system: ParticleSystem) -> bool:
        if self._pairs is None or self._ref_positions is None or self._ref_box is None:
            return True

        if not np.allclose(system.box, self._ref_box):
            return True

        if self._steps_since_build >= self.rebuild_interval:
            return True

        if self.skin <= 0.0:
            return False

        displacement = minimum_image(system.positions - self._ref_positions, system.box)
        max_disp_sq = float(np.max(np.einsum("ij,ij->i", displacement, displacement)))
        return max_disp_sq > (0.5 * self.skin) ** 2

    def _build_pairs(self, system: ParticleSystem) -> np.ndarray:
        positions = system.positions
        n_particles = system.n_particles
        if n_particles < 2:
            return np.zeros((0, 2), dtype=np.int64)

        displacement = minimum_image(
            positions[:, None, :] - positions[None, :, :],
            system.box,
        )
        r_sq = np.einsum("ijk,ijk->ij", displacement, displacement)

        rc_sq = self.pair_cutoff * self.pair_cutoff
        upper = np.triu(np.ones((n_particles, n_particles), dtype=bool), k=1)
        mask = upper & (r_sq <= rc_sq)
        i_idx, j_idx = np.where(mask)

        if i_idx.size == 0:
            return np.zeros((0, 2), dtype=np.int64)

        return np.column_stack([i_idx, j_idx]).astype(np.int64, copy=False)

    def get_pairs(self, system: ParticleSystem) -> np.ndarray:
        if self._needs_rebuild(system):
            self._pairs = self._build_pairs(system)
            self._ref_positions = system.positions.copy()
            self._ref_box = system.box.copy()
            self._steps_since_build = 0
        else:
            self._steps_since_build += 1

        if self._pairs is None:
            return np.zeros((0, 2), dtype=np.int64)
        return self._pairs


class NeighborListCoulombSolver:
    name = "neighbor"

    def __init__(
        self,
        cutoff: float,
        skin: float = 0.3,
        rebuild_interval: int = 10,
        softening: float = 1e-12,
        use_numba: bool = True,
    ) -> None:
        self.cutoff = float(cutoff)
        self.softening = float(softening)
        self.neighbor_list = VerletNeighborList(
            cutoff=self.cutoff,
            skin=skin,
            rebuild_interval=rebuild_interval,
        )
        self.jit_enabled = bool(use_numba and _NUMBA_AVAILABLE)

    def compute(
        self,
        system: ParticleSystem,
        rng: np.random.Generator | None = None,
    ) -> ForceResult:
        del rng

        pairs = self.neighbor_list.get_pairs(system)
        if pairs.size == 0:
            return ForceResult(
                forces=np.zeros_like(system.positions),
                potential=0.0,
                virial=0.0,
            )

        cutoff_sq = self.cutoff * self.cutoff
        softening_sq = self.softening * self.softening

        if self.jit_enabled:
            forces, potential, virial = _compute_short_range_numba(
                positions=np.ascontiguousarray(system.positions, dtype=np.float64),
                charges=np.ascontiguousarray(system.charges, dtype=np.float64),
                box=np.ascontiguousarray(system.box, dtype=np.float64),
                pairs=np.ascontiguousarray(pairs, dtype=np.int64),
                cutoff_sq=float(cutoff_sq),
                softening_sq=float(softening_sq),
            )
            return ForceResult(
                forces=forces,
                potential=float(potential),
                virial=float(virial),
            )

        return self._compute_vectorized(
            positions=system.positions,
            charges=system.charges,
            box=system.box,
            pairs=pairs,
            cutoff_sq=cutoff_sq,
            softening_sq=softening_sq,
        )

    def _compute_vectorized(
        self,
        positions: np.ndarray,
        charges: np.ndarray,
        box: np.ndarray,
        pairs: np.ndarray,
        cutoff_sq: float,
        softening_sq: float,
    ) -> ForceResult:
        forces = np.zeros_like(positions)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]

        displacement = minimum_image(positions[j_idx] - positions[i_idx], box)
        r_sq = np.einsum("ij,ij->i", displacement, displacement)
        valid = (r_sq > softening_sq) & (r_sq <= cutoff_sq)
        if not np.any(valid):
            return ForceResult(forces=forces, potential=0.0, virial=0.0)

        i_valid = i_idx[valid]
        j_valid = j_idx[valid]
        disp_valid = displacement[valid]
        r_sq_valid = r_sq[valid]

        inv_r = 1.0 / np.sqrt(r_sq_valid)
        inv_r3 = inv_r * inv_r * inv_r
        qij = charges[i_valid] * charges[j_valid]

        pair_force = (qij * inv_r3)[:, None] * disp_valid
        np.add.at(forces, i_valid, pair_force)
        np.add.at(forces, j_valid, -pair_force)

        potential = float(np.sum(qij * inv_r))
        virial = float(np.sum(np.einsum("ij,ij->i", disp_valid, pair_force)))

        return ForceResult(forces=forces, potential=potential, virial=virial)
