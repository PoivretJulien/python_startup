"""A Vec2 compatible with experimental_b's drawing API.

Adapted from ``python_startup/lib/vec2.py``. The original class was a plain
``@dataclass`` with no sequence protocol, so it could not be unpacked or passed
to ``np.asarray``. This version adds ``__len__``, ``__getitem__`` and
``__iter__`` (via ``Sequence``), which makes it usable with:

    system.draw_line(Vec2(0, 0), Vec2(1, 2), color)   # 2D: only for your own math
    Vec3(v.x, v.y, z)                                # promote to the 3D API

The 3D renderers still expect three coordinates, so a Vec2 is mainly useful for
2D calculations (clock hands, directions, screen-space work) that are then
promoted to Vec3.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload


@dataclass(slots=True)
class Vec2(Sequence[float]):
    """A minimal 2D vector with the usual arithmetic and vector helpers."""

    x: float
    y: float

    # -- sequence protocol: lets tuples/lists/NumPy accept a Vec2 ----------
    def __len__(self) -> int:
        return 2

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return (self.x, self.y)[index]

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.y * k)

    def __rmul__(self, k: float) -> Vec2:
        return self * k

    def __truediv__(self, k: float) -> Vec2:
        if k == 0.0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vec2(self.x / k, self.y / k)

    # -- helpers -----------------------------------------------------------
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def determinant(self, other: Vec2) -> float:
        return self.x * other.y - self.y * other.x

    def normalized(self) -> Vec2:
        n = self.length()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalize a zero-length vector")
        return self * (1.0 / n)

    def to_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def to_string(self) -> str:
        return f"Vec2(x:{self.x:3.3f},y:{self.y:3.3f})"
