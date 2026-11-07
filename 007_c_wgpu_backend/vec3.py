"""A Vec3 compatible with experimental_b's drawing API.

Adapted from ``python_startup/lib/vec3.py`` with three changes:

* it implements the sequence protocol (``__len__`` / ``__getitem__`` /
  ``__iter__``), so a Vec3 can be passed **directly** to ``draw_arrow``,
  ``draw_line``, ``draw_circle`` and ``draw_text`` — the geometry unpacks it or
  gives it to ``np.asarray(..., dtype=float)``;
* ``__mul__`` returns all three components (the original forgot ``z``);
* ``cross`` uses ``other`` for the second operand (the original always
  returned the zero vector).

The arithmetic stays plain Python floats, and the class remains a dataclass.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import overload


@dataclass(slots=True)
class Vec3(Sequence[float]):
    """A minimal 3D vector with the arithmetic and vector helpers used here."""

    x: float
    y: float
    z: float

    # -- sequence protocol: lets tuples/lists/NumPy accept a Vec3 ----------
    def __len__(self) -> int:
        return 3

    @overload
    def __getitem__(self, index: int) -> float: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[float, ...]: ...

    def __getitem__(self, index: int | slice) -> float | tuple[float, ...]:
        return (self.x, self.y, self.z)[index]

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    # -- arithmetic --------------------------------------------------------
    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> Vec3:
        return Vec3(self.x * k, self.y * k, self.z * k)

    def __rmul__(self, k: float) -> Vec3:
        return self * k

    def __truediv__(self, k: float) -> Vec3:
        if k == 0.0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vec3(self.x / k, self.y / k, self.z / k)

    # -- helpers -----------------------------------------------------------
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def normalized(self) -> Vec3:
        n = self.length()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalize a zero-length vector")
        return self * (1.0 / n)

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_string(self) -> str:
        return f"Vec3(x:{self.x:3.3f},y:{self.y:3.3f},z:{self.z:3.3f})"
