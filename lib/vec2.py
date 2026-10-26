"""
Simple python object example.

Objective: demonstrate the python object class definition.
Subtracting two Vec2 gives a delta Vec2 where the length() method
can be called directly in the chain.

This should give the basic mechanism of modern python to build solid
scripts, for the Blender API or in a standalone application.
"""

import math
import sys
from dataclasses import dataclass


@dataclass(slots=True)
class Vec2:
    """
            tool box for a classic Vec2.
    """
    x: float
    y: float

    # arithmetic operator + - * / ...
    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    def __rmul__(self, k: float) -> "Vec2":
        return self * k

    def __truediv__(self, k: float) -> "Vec2":
        if k == 0.0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vec2(self.x / k, self.y / k)

    # compute hypotenuse (the length)  
    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y
    
    def determinant(self, other: "Vec2") -> float:
        return self.x * other.y - self.y * other.x

    # convert the vector at unit circle scale
    def normalized(self) -> "Vec2":
        n = self.length()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalize a zero-length vector")
        return self * (1.0 / n)

    # format an entity to str.
    def to_string(self)->str:
        return f"Vec2(x:{self.x:3.3f},y:{self.y:3.3f})"


