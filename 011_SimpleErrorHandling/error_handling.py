import sys
from dataclasses import dataclass

"""
    the minimalism of effective error handling in python
    simple and clear.
"""

def square(num: float) -> float | None:
    """this function is design to return a float or None"""
    if num >= 0:
        return num
    else:
        return None

# Custom error that inherit from Exception 
# ( Exception is a system built-in base class 
# that all error inherit from ).
class MyError(Exception):
    """Exception dedicated to my api"""

@dataclass(slots=True)
class MyApi:
    """ A minimal api that handle custom error for bug tracing """
    val: bool
    def method_a(self) -> bool:
        if self.val:
            return self.val
        else:
            raise MyError("Error: Method a fail")

# Simple and modern minimal python script.
def main() -> int:
    # return a float or None.
    result_a = square(34)
    result_b = square(-32.2)
    print(f"Hello result: {result_a}, {result_b}")

    # True or False for different error handling behaviour.
    api_instance = MyApi(True)

    # Try and catch the error on Exception.
    try:
        r = api_instance.method_a()
        print(f"Ok job done. {r}")
    except MyError as e:
        # action code block to trigger on action.
        print(f"{type(e).__name__} {e}")
    finally:
        # acknowledge the end of the program.
        print("program finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
