from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def minimum_image(displacement: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Apply minimum image convention for periodic boundary conditions."""
    return displacement - box * np.round(displacement / box)


def wrap_positions(positions: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Wrap particle positions into [0, box) for each axis."""
    return np.mod(positions, box)


def kinetic_energy(velocities: np.ndarray, masses: np.ndarray) -> float:
    return float(0.5 * np.sum(masses[:, None] * velocities**2))


def kinetic_temperature(velocities: np.ndarray, masses: np.ndarray) -> float:
    dof = velocities.size - 3
    if dof <= 0:
        return 0.0
    return float((2.0 * kinetic_energy(velocities, masses)) / dof)


def pressure_from_virial(
    n_particles: int,
    volume: float,
    temperature: float,
    virial: float,
    k_b: float = 1.0,
) -> float:
    """Reduced-unit instantaneous pressure estimate."""
    return float((n_particles * k_b * temperature + virial / 3.0) / volume)


def _to_serializable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_serializable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_serializable(v) for v in value]
    return value


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = _to_serializable(payload)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)


def summarize_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
