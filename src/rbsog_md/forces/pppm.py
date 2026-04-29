from __future__ import annotations

import numpy as np

from rbsog_md.forces.base import ForceResult
from rbsog_md.system import ParticleSystem


class PPPMSolver:
    name = "pppm"

    def __init__(self, grid_shape: tuple[int, int, int] = (32, 32, 32)) -> None:
        self.grid_shape = grid_shape

    def _grid_indices(self, positions: np.ndarray, box: np.ndarray) -> np.ndarray:
        fractional = positions / box
        indices = np.floor(fractional * np.array(self.grid_shape)).astype(int)
        for axis in range(3):
            indices[:, axis] %= self.grid_shape[axis]
        return indices

    def compute(
        self,
        system: ParticleSystem,
        rng: np.random.Generator | None = None,
    ) -> ForceResult:
        del rng
        positions = system.positions
        charges = system.charges
        box = system.box

        nx, ny, nz = self.grid_shape
        cell_volume = system.volume / float(nx * ny * nz)

        rho = np.zeros(self.grid_shape, dtype=float)
        idx = self._grid_indices(positions, box)
        np.add.at(rho, (idx[:, 0], idx[:, 1], idx[:, 2]), charges / cell_volume)

        rho_k = np.fft.fftn(rho)

        kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=box[0] / nx)
        ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=box[1] / ny)
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=box[2] / nz)

        kx3 = kx[:, None, None]
        ky3 = ky[None, :, None]
        kz3 = kz[None, None, :]
        k_sq = kx3**2 + ky3**2 + kz3**2

        phi_k = np.zeros_like(rho_k, dtype=complex)
        mask = k_sq > 0.0
        phi_k[mask] = (4.0 * np.pi) * rho_k[mask] / k_sq[mask]

        ex = np.fft.ifftn(-1j * kx3 * phi_k).real
        ey = np.fft.ifftn(-1j * ky3 * phi_k).real
        ez = np.fft.ifftn(-1j * kz3 * phi_k).real
        phi = np.fft.ifftn(phi_k).real

        e_particle = np.column_stack(
            [
                ex[idx[:, 0], idx[:, 1], idx[:, 2]],
                ey[idx[:, 0], idx[:, 1], idx[:, 2]],
                ez[idx[:, 0], idx[:, 1], idx[:, 2]],
            ]
        )
        phi_particle = phi[idx[:, 0], idx[:, 1], idx[:, 2]]

        forces = charges[:, None] * e_particle
        potential = float(0.5 * np.sum(charges * phi_particle))
        virial = float(np.sum(np.einsum("ij,ij->i", positions, forces)))

        return ForceResult(forces=forces, potential=potential, virial=virial)
