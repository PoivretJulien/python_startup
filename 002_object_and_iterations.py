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

# main body entry.
def main() -> int:

    # Define some ascii terminal colors
    yellow = "\x1b[33m"
    rst = "\x1b[0m"  # reset
    green = "\x1b[32m"

    # Simple 2 instance of the object Vec2
    va = Vec2(1.0, 1.0)
    vb = Vec2(-1.0, -1.0)

    # Compute the length of the resulting object.
    length = (va - vb).length()

    # Expected value to compare.
    expected = math.sqrt(2) * 2

    # Token decoration for result confirmation.
    if math.isclose(length, expected):
        token = "\u2705"
    else:
        token = "\u274c"
        
    # Print the result.
    print(
        f"Vector length: {yellow}{length:3.3f}{rst} "
        f"(must be equal to sqrt(2)*2 {yellow}({expected:3.3f}){rst}) -> {token}"
    )

    # Determinant of zero mean that the "scalar factor of change" collapse into
    # one point or a single line the input vectors are configured in an opposite
    # diagonal in that case resulting to a length of zero in that case.
    print(
        f"Determinant of {green}{va}{rst} and {green}{vb}{rst} = {yellow}{va.determinant(vb)}{rst}"
    )

    # Create a list and sum the points this will cumulate 
    # negatives and positives values of each directions, giving us
    # the average direction.
    data: list[Vec2] = []

    # Increase value to test computing machine and python capability
    compute_quantity = 50000
   
    # Populate vector on x an y respectively 
    for i in range(1, compute_quantity):
        data.append(Vec2(i, 0))
        data.append(Vec2(0, i * (-1)))
    # print(f"{data}")

    # Sum the vectors.
    total = Vec2(0.0, 0.0)
    for v in data:
        total = total + v

    # Normalizing the average give the unit circle scale  
    # representation of the average direction.
    print(f"Total sum: {yellow}{total}{rst} of {green}{compute_quantity}{rst} "
          f"average direction: {yellow}{total.normalized().to_string()}{rst}")

    return 1

if __name__ == "__main__":
    sys.exit(main())
