"""Turns decoded commands into droplet positions.

Two droplets are held in the field. The user steers droplet A with LEFT /
RIGHT / FORWARD / BACK. MERGE glides A toward B one step at a time until
they touch.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import ControlConfig

DIRECTIONS = {
    "LEFT": np.array([-1.0, 0.0, 0.0]),
    "RIGHT": np.array([1.0, 0.0, 0.0]),
    "FORWARD": np.array([0.0, 1.0, 0.0]),
    "BACK": np.array([0.0, -1.0, 0.0]),
}


@dataclass
class DropletState:
    a: np.ndarray = field(default_factory=lambda: np.array([-0.01, 0.0, 0.05]))
    b: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.0, 0.05]))
    merged: bool = False

    def targets(self):
        return [self.a] if self.merged else [self.a, self.b]


class DropletController:
    def __init__(self, cfg: ControlConfig = ControlConfig(), state: DropletState = None):
        self.cfg = cfg
        self.state = state or DropletState()

    def _clamp(self, p: np.ndarray) -> np.ndarray:
        lo = np.array([b[0] for b in self.cfg.bounds_m])
        hi = np.array([b[1] for b in self.cfg.bounds_m])
        return np.clip(p, lo, hi)

    def apply(self, command: str) -> DropletState:
        s = self.state
        if s.merged:
            return s
        if command in DIRECTIONS:
            s.a = self._clamp(s.a + DIRECTIONS[command] * self.cfg.step_m)
        elif command == "MERGE":
            gap = s.b - s.a
            dist = float(np.linalg.norm(gap))
            if dist > 0:
                s.a = s.a + gap / dist * min(self.cfg.step_m, dist)
        else:
            raise ValueError(f"Unknown command: {command}")

        if np.linalg.norm(s.b - s.a) <= self.cfg.merge_distance_m:
            s.merged = True
            s.a = (s.a + s.b) / 2
        return s
