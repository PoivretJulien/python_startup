"""
The early fundamental of computing:
-> transformation of a vector by a simple 2x3 matrix
Simple and minimal python script that "crystallize
this time less technique, of course compiled language
for such computation would be an optimal choice
for a direct application case,
but with far more technical frictions constraints,
we are focusing on theory there, not effective mass 
computing applications.

quotes:"a clear mind worth more than hours of works for nothing more."
"""

from lib.vec2 import Vec2
import sys


# notes:
# there are no significant differences from a python or c code
# on such basic fundamental array operations syntax is pretty similar,
# only the runtime differ, so from a theory point of view
# that's all good for simple models.

# Define some ascii terminal colors
yellow = "\x1b[33m"
red = "\x1b[31m"
blue = "\x1b[34m"
rst = "\x1b[0m"  # reset
green = "\x1b[32m"


def main() -> int:

    # A simple 2d square
    square_2d = [Vec2(1, 1), Vec2(-1, 1), Vec2(-1, -1), Vec2(1, -1)]

    # Identity matrix.
    identity = [
        [1, 0, 0],
        [0, 1, 0],
    ]

    # 45 deg (anticlockwise) + ~30 % scale.
    rotationA = [
        [1, 1, 0],
        [1, -1, 0],
    ]

    # Inverse of square root of 2 -> (1/math.sqrt(2))
    inv = 1.1412 * 0.5

    # 45 deg (anticlockwise) + zero strech.
    rotationB = [
        [inv, -inv, 0],
        [inv, inv, 0],
    ]

    # Translation of 3 units on X axis.
    translationA = [
        [1, 0, 3],
        [0, 1, 0],
    ]

    # Create a new square memory list
    rotated_square = []

    # Print the square the before transformation.
    print("Square initial state")
    print_the_square(square_2d, yellow)

    # Apply the rotation.
    for i in range(4):
        rotated_square.append(transform_point(square_2d[i], rotationB))

    # Print the transformed square
    print("Rotated square 45 deg AntiClockWise")
    print_the_square(rotated_square, green)
    translated_square = []

    # Apply the translation.
    for i in range(4):
        translated_square.append(transform_point(rotated_square[i], translationA))

    # Print the transformed square
    print("Translated square on (3 units on x axis)")
    print_the_square(translated_square, blue)
    return 1


def transform_point(pt: Vec2, m: list[list[float]]) -> Vec2:
    """
    transform point with a 2x3 matrix
    """
    x = pt.x * m[0][0] + pt.y * m[0][1] + m[0][2]
    y = pt.x * m[1][0] + pt.y * m[1][1] + m[1][2]
    return Vec2(x, y)


def print_the_square(square: list[Vec2], color: str) -> None:
    """
    display the numerical state of the square on the os/application terminal console.
    """
    print(f"2D square: [{color}")
    for point in square:
        print(f"{point.to_string()}")
    print(f"{rst}]")


if __name__ == "__main__":
    sys.exit(main())
