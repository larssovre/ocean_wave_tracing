from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Spectrum2D:
    freq_hz: np.ndarray
    dirs_sorted_rad: np.ndarray
    E_sorted: np.ndarray
    dirs_math_unsorted_rad: np.ndarray
    order: np.ndarray
    inv_order: np.ndarray

@dataclass(frozen=True)
class Box:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @staticmethod
    def from_any(xmin: float, xmax: float, ymin: float, ymax: float) -> "Box":
        return Box(min(xmin, xmax), max(xmin, xmax), min(ymin, ymax), max(ymin, ymax))
