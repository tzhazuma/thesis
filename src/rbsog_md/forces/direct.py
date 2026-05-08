from __future__ import annotations

import time

import numpy as np

from rbsog_md.forces.base import ForceResult
from rbsog_md.system import ParticleSystem
from rbsog_md.utils import minimum_image


class DirectCoulombSolver:
    name = "direct"

    def __init__(self, cutoff: float | None = None, softening: float = 1e-12, profile: bool = False) -> None:
        self.cutoff = cutoff
        self.softening = softening
        self.profile = bool(profile)

    def compute(
        self,
        system: ParticleSystem,
        rng: np.random.Generator | None = None,
    ) -> ForceResult:
        del rng
        t0 = time.perf_counter() if self.profile else 0.0
        positions = system.positions
        charges = system.charges
        box = system.box
        n_particles = system.n_particles

        forces = np.zeros_like(positions)
        potential = 0.0
        virial = 0.0
        evaluated_pairs = 0

        cutoff_sq = None if self.cutoff is None else self.cutoff * self.cutoff
        softening_sq = self.softening * self.softening

        for i in range(n_particles - 1):
            displacement = minimum_image(positions[i + 1 :] - positions[i], box)
            r_sq = np.einsum("ij,ij->i", displacement, displacement)
            mask = r_sq > softening_sq
            if cutoff_sq is not None:
                mask &= r_sq <= cutoff_sq
            if not np.any(mask):
                continue

            disp_valid = displacement[mask]
            j_indices = np.arange(i + 1, n_particles)[mask]
            r_sq_valid = r_sq[mask]

            inv_r = 1.0 / np.sqrt(r_sq_valid)
            inv_r3 = inv_r**3
            qij = charges[i] * charges[j_indices]

            pair_force = (qij * inv_r3)[:, None] * disp_valid
            forces[i] += np.sum(pair_force, axis=0)
            np.add.at(forces, j_indices, -pair_force)

            potential += float(np.sum(qij * inv_r))
            virial += float(np.sum(np.einsum("ij,ij->i", disp_valid, pair_force)))
            evaluated_pairs += int(r_sq_valid.size)

        diagnostics = None
        if self.profile:
            diagnostics = {
                "direct_total_time": float(time.perf_counter() - t0),
                "direct_pair_count": float(evaluated_pairs),
            }

        return ForceResult(forces=forces, potential=potential, virial=virial, diagnostics=diagnostics)
