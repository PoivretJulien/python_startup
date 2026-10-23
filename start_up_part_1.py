import math
import sys
import time
import tkinter as tk
from tkinter import messagebox


def main() -> int:
    """
    -> Python script lab initialization.
    Objective:
        cleanly shape prototype and bring quick interactivity at sample scale
        then leverage IA/heavy workload with the right technology for the final
        application beyond the conceptual stage.
    """

    # the classic formatting.
    # f" ---> {variable_to_print}"

    print(f"\x1b[33mHello lab \x1b[34m{time.ctime()}\x1b[33m.\x1b[0m")

    # A simple tuple.
    position = (3, 4)
    length_from_origin = math.sqrt(
        position[0] * position[0] + position[1] * position[1]
    )

    # Function call with boolean evaluation precomputed evaluation (predicate).
    check(length_from_origin == 5)

    # A false predicate
    check(1 - (-1) == 0)

    # basic for loop
    for i in range(6):
        print(f"{i},", end=" ")  # print on a single line end=" "

    tk.Tk().withdraw()
    messagebox.showinfo(
        title="my message", message="this first demo may be interesting for some..."
    )

    # the function must return an int
    return 1


def check(predicate: bool):
    # Basic conditional evaluation.
    if predicate:
        print("\u2705 ok !")
    else:
        print("\u274c error !")


# when typing python startup.py this will call main() and pass an int to sys.exit()
if __name__ == "__main__":
    sys.exit(main())
