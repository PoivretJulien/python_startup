import math
from dataclasses import dataclass


@dataclass(slots=True)
class Vec3:
    """
    tool box for a classic Vec3.
    """

    x: float
    y: float
    z: float

    # arithmetic operator + - * / ...
    def __add__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vec3") -> "Vec3":
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, k: float) -> "Vec3":
        return Vec3(self.x * k, self.y * k)

    def __rmul__(self, k: float) -> "Vec3":
        return self * k

    def __truediv__(self, k: float) -> "Vec3":
        if k == 0.0:
            raise ZeroDivisionError("cannot divide a vector by zero")
        return Vec3(self.x / k, self.y / k, self.z / k)

    # compute hypotenuse (the length)
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, other: "Vec3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vec3") -> "Vec3":
        return Vec3(
            self.y * self.z - self.z * self.y,
            self.z * self.x - self.x * self.z,
            self.x * self.y - self.y * self.x,
        )

    # convert the vector at unit circle scale
    def normalized(self) -> "Vec3":
        n = self.length()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalize a zero-length vector")
        return self * (1.0 / n)

    # format an entity to str.
    def to_string(self) -> str:
        return f"Vec3(x:{self.x:3.3f},y:{self.y:3.3f},z:{self.z:3.3f})"
