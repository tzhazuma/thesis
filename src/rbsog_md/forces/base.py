from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rbsog_md.system import ParticleSystem


@dataclass
class ForceResult:
    forces: np.ndarray
    potential: float
    virial: float
    diagnostics: dict[str, float] | None = None


class ForceSolver(Protocol):
    name: str

    def compute(
        self,
        system: ParticleSystem,
        rng: np.random.Generator | None = None,
    ) -> ForceResult:
        ...
