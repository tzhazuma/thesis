from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rbsog_md.utils import kinetic_temperature, wrap_positions


@dataclass
class ParticleSystem:
    positions: np.ndarray
    velocities: np.ndarray
    charges: np.ndarray
    masses: np.ndarray
    box: np.ndarray
    n_lipids: int = 0

    def copy(self) -> "ParticleSystem":
        return ParticleSystem(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            charges=self.charges.copy(),
            masses=self.masses.copy(),
            box=self.box.copy(),
            n_lipids=self.n_lipids,
        )

    @property
    def n_particles(self) -> int:
        return int(self.positions.shape[0])

    @property
    def volume(self) -> float:
        return float(np.prod(self.box))

    @property
    def temperature(self) -> float:
        return kinetic_temperature(self.velocities, self.masses)

    def wrap(self) -> None:
        self.positions = wrap_positions(self.positions, self.box)

    def area_per_lipid(self) -> float:
        if self.n_lipids < 2:
            return float("nan")
        return float((self.box[0] * self.box[1]) / (self.n_lipids / 2.0))

    def membrane_thickness_proxy(self) -> float:
        if self.n_lipids < 4:
            return float("nan")
        z = self.positions[: self.n_lipids, 2]
        mid = 0.5 * self.box[2]
        top = z[z >= mid]
        bottom = z[z < mid]
        if top.size == 0 or bottom.size == 0:
            return float("nan")
        return float(np.mean(top) - np.mean(bottom))


def initialize_velocities(
    masses: np.ndarray,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    scales = np.sqrt(temperature / masses)
    velocities = rng.normal(0.0, 1.0, size=(masses.shape[0], 3)) * scales[:, None]
    center_of_mass_velocity = np.average(velocities, axis=0, weights=masses)
    velocities -= center_of_mass_velocity
    return velocities


def _leaflet_grid_positions(
    n_leaflet: int,
    box: np.ndarray,
    z_center: float,
    rng: np.random.Generator,
    jitter_xy: float,
    jitter_z: float,
) -> np.ndarray:
    grid_n = int(np.ceil(np.sqrt(n_leaflet)))
    xs = np.linspace(0.5, box[0] - 0.5, grid_n)
    ys = np.linspace(0.5, box[1] - 0.5, grid_n)
    points = []
    for x in xs:
        for y in ys:
            points.append((x, y))
    points = np.array(points[:n_leaflet], dtype=float)
    points[:, 0] += rng.normal(0.0, jitter_xy, size=n_leaflet)
    points[:, 1] += rng.normal(0.0, jitter_xy, size=n_leaflet)
    z = z_center + rng.normal(0.0, jitter_z, size=n_leaflet)
    positions = np.column_stack([points[:, 0], points[:, 1], z])
    return wrap_positions(positions, box)


def build_membrane_proxy_system(
    n_lipids: int = 128,
    n_solvent: int = 512,
    box: tuple[float, float, float] = (16.0, 16.0, 20.0),
    temperature: float = 1.0,
    seed: int = 42,
) -> ParticleSystem:
    """Build a membrane-inspired coarse-grained benchmark system.

    The first n_lipids particles are lipid proxy beads split into two leaflets.
    Remaining particles are solvent proxy beads.
    """
    if n_lipids % 2 != 0:
        raise ValueError("n_lipids must be even")

    rng = np.random.default_rng(seed)
    box_arr = np.array(box, dtype=float)
    n_leaflet = n_lipids // 2

    top = _leaflet_grid_positions(
        n_leaflet=n_leaflet,
        box=box_arr,
        z_center=0.66 * box_arr[2],
        rng=rng,
        jitter_xy=0.15,
        jitter_z=0.08,
    )
    bottom = _leaflet_grid_positions(
        n_leaflet=n_leaflet,
        box=box_arr,
        z_center=0.34 * box_arr[2],
        rng=rng,
        jitter_xy=0.15,
        jitter_z=0.08,
    )

    solvent = rng.uniform(low=(0.0, 0.0, 0.0), high=box_arr, size=(n_solvent, 3))

    positions = np.vstack([top, bottom, solvent])

    lipid_charges = np.concatenate([np.ones(n_leaflet), -np.ones(n_leaflet)])

    solvent_charges = np.zeros(n_solvent, dtype=float)
    if n_solvent > 0:
        half = n_solvent // 2
        solvent_charges[:half] = 0.2
        solvent_charges[half : 2 * half] = -0.2
        rng.shuffle(solvent_charges)

    charges = np.concatenate([lipid_charges, solvent_charges])
    charges -= np.mean(charges)
    charges[-1] -= np.sum(charges)

    masses = np.ones(positions.shape[0], dtype=float)
    velocities = initialize_velocities(masses=masses, temperature=temperature, rng=rng)

    system = ParticleSystem(
        positions=positions,
        velocities=velocities,
        charges=charges,
        masses=masses,
        box=box_arr,
        n_lipids=n_lipids,
    )
    system.wrap()
    return system
