"""
The early fundamental of computing:
-> transformation of a vector by a simple 2x3 matrix.

Simple and minimal python script that "crystallize
this 'timeless' technique. 
Of course compiled language for such computation would be 
an optimal choice for a concreate application case,
but with far, far more technical frictions constraints.
there we are focusing on theory, not effective mass
computing applications in a nested context.

quotes:"A clear mind worth more than hours of works for nothing more."
"""

from lib.vec2 import Vec2
from lib.vec3 import Vec3
import sys


# notes:
# there are no significant differences between a python or c code
# on such basic fundamental array operations. it's pretty mush similar,
# only the runtime differ, so from a theory point of view
# that's what we want for a clear mental model.

# Define some ascii terminal colors
yellow = "\x1b[33m"
red = "\x1b[31m"
blue = "\x1b[34m"
rst = "\x1b[0m"  # reset
green = "\x1b[32m"


def main() -> int:
    """
    In this example a 3x3 matrix will be used for learning 
    2D transformations.
     R U T = (Right, Up, Translation) (columns meaning)
    [1,0,0]   R(x axis) become x=1 y=0 
    [0,1,0]   U(y axis) Become x=0 Y=1
    [0,0,0]   this is the 2x2 identity matrix (no change)
    """

    # A simple 2d square
    square_2d = [Vec2(1, 1), Vec2(-1, 1), Vec2(-1, -1), Vec2(1, -1)]

    # 3x3 Identity matrix (neutral no transformation).
    # simply a diagonal of 1.
    identity = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
     
    # - 45 deg (anticlockwise) + ~41 % scale ( Eigen vector streching ).
    # the square of 1, after transformation (rotation) would have an half edge of 1.41 sqrt{2}
    # so each side will strech by ~41,42% 
    # - the last collumn is dedicated to the translation (no translation in that case.)
    # [2][2] = 1, it's the last 1 of the diagonal of the identity matrix.
    rotationA = [
        [1, 1, 0],
        [1, -1, 0],
        [0, 0, 1],  
    ] 
    

    # Inverse of square root of 2 -> (1/math.sqrt(2))
    inv = 1.4142 * 0.5

    # 45 deg (anticlockwise) + zero strech.
    rotationB = [
        [inv, -inv, 0],
        [inv, inv, 0],
        [0, 0, 1],
    ]

    # Translation of 3 units on X axis.
    translationA = [
        [1, 0, 3],
        [0, 1, 0],
        [0, 0, 1],
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

    # combine transformations together 
    # (reverse order must be apply to recreate the same transformation as the
    # non combined one).
    combined_transform = multiply_matrices(identity, translationA)
    combined_transform = multiply_matrices(combined_transform, rotationB)

    rotated_translated_2d_square = []
    # Apply the combined rotation + transformation.
    for i in range(4):
        rotated_translated_2d_square.append(
            transform_point(square_2d[i], combined_transform)
        )

    print_the_square(square_2d, yellow)
    print_the_square(rotated_translated_2d_square, red)

    return 1


def multiply_matrices(
    m_a: list[list[float]], m_b: list[list[float]]
) -> list[list[float]]:
    """
    Combine two 3x3 matrices transformations.
    -> "the dot product of each m_a row by each m_b column"
    """
    # Format the result memory.
    result = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    # extract row from matrix A
    a0 = Vec3(m_a[0][0],m_a[0][1],m_a[0][2])
    a1 = Vec3(m_a[1][0],m_a[1][1],m_a[1][2])
    a2 = Vec3(m_a[2][0],m_a[2][1],m_a[2][2])

    # extract column from matrix B
    b0 = Vec3(m_b[0][0],m_b[1][0],m_b[2][0])
    b1 = Vec3(m_b[0][1],m_b[1][1],m_b[2][1])
    b2 = Vec3(m_b[0][2],m_b[1][2],m_b[2][2])

    # Multiply both matrices together.
    # first line.
    result[0][0] = a0.dot(b0)
    result[0][1] = a0.dot(b1) 
    result[0][2] = a0.dot(b2)
    # second line.
    result[1][0] = a1.dot(b0) 
    result[1][1] = a1.dot(b1)
    result[1][2] = a1.dot(b2)
    # third line.
    result[2][0] = a2.dot(b0)
    result[2][1] = a2.dot(b1)
    result[2][2] = a2.dot(b2)
    # return the result.
    return result


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
