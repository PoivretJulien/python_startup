import time
import numpy as np
import sys
import math
from lib.vec3 import Vec3
import random


# Define some ascii terminal colors
yellow = "\x1b[33m"
red = "\x1b[31m"
blue = "\x1b[34m"
rst = "\x1b[0m"  # reset
green = "\x1b[32m"
purple = "\x1b[35m"


def main() -> int:
    """
    Python is slow... well, but how much ? ( that's the 'fast thinking' state of mind )
    Objective:
            measure with simplicity the scripting performance, and evaluate when
            a compiled technology is needed.
    Quote: "a good quality of life, keep your mind sharp."
    """

    # First, establish a simple timer system.
    ITERATIONS = 100
    measured_perf = 0
    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        time.sleep(0.005)  # 5 ms
        now = (time.perf_counter_ns() - start) / 1000000  # (ns to ms)
        measured_perf += now

    # The target measured is 5 ms
    result = measured_perf / ITERATIONS
    granularity = math.fabs(5 - result)
    print(f"{green}for a 5 ms target:{rst}")
    print(
        f"{purple}->{rst} "
        + f"measured timing: {yellow}{result:1.3f}ms{rst} system granularity: "
        + f"{yellow}(+-){granularity:3.3f}ms{rst}"
    )

    # the system granularity is the foot print of
    # python + os scheduling process execution hierarchy
    # now with this technical approach we can mesure OS + python runtime
    # foot print deviation from a 5 ms theoretical target.
    # ( on my low perf computer im about 146 us of granularity, the blink of an
    # eye is about 100 to 400 millisecond the granularity is about 700 time
    # faster than a blink of eyes )

    # now let measure real computing performances
    # -> create 2 list of 1000 vectors each.
    list_a = []
    list_b = []
    for _ in range(1000):
        list_a.append(Vec3(random.random(), random.random(), random.random()))
        list_b.append(Vec3(random.random(), random.random(), random.random()))

    print(f"{green}1000 dot product test{rst}")
    # then compute the dot product across those 2000 vectors
    measured_perf = 0
    ITERATIONS = 10000
    for _ in range(ITERATIONS):
        dot_list = []
        start = time.perf_counter_ns()
        for i in range(1000):
            dot_list.append(list_a[i].dot(list_b[i]))
        now = (time.perf_counter_ns() - start) / 1000000  # (ns to ms)
        measured_perf += now
    result = measured_perf / ITERATIONS
    print(
        "-> 1000 dot products of 2000 vectors computed in an average of: "
        + f"{yellow}{result:3.3f}ms{rst} (list length of the "
        + f"result:{len(dot_list)})"
    )

    # -------------------------------------------------------------
    # Now with numpy a lib that involve pre-compiled runtime loops
    # for fast computing directly driven by the cpython runtime.
    # the syntax is a bit technical but the performance gain is there
    # -------------------------------------------------------------
    # but that the price to pay for a better performances directly driven from python.
    # ( but i'm there for clarity not performance. even if the demonstration worth it 
    # it's another layer of technicality over the math and geometry theory)
    # pragmatically, better to focus on real math theory and let ia make the optimal 
    # choice it's where ia driven by python may be the optimal choice.
    # -------------------------------------------------------------
    rng = np.random.default_rng(0)                                                                                    
                                                                                                                     
    # 1000 pairs of 3-D vectors
    list_a = rng.random((1000, 3))                                                                                    
    list_b = rng.random((1000, 3))                                                                                    
                                                                                                                     
    # Batched views for np.matmul:                                                                                    
    # (1000, 1, 3) @ (1000, 3, 1) -> (1000, 1, 1) -> (1000,)                                                          
    a = list_a[:, None, :]                                                                                            
    b = list_b[:, :, None]                                                                                            
                                                                                                                     
    print(f"{green}1000 dot product test{rst}")                                                                       
                                                                                                                     
    # Warm-up: keep first-call setup out of the average.                                                              
    for _ in range(100):                                                                                              
        dot_list = np.matmul(a, b)[:, 0, 0]                                                                           
    measured_perf = 0.0                                                                                               
    ITERATIONS = 10000                                                                                                
    for _ in range(ITERATIONS):                                                                                       
        start = time.perf_counter_ns()                                                                                
        dot_list = np.matmul(a, b)[:, 0, 0]                                                                           
        now = (time.perf_counter_ns() - start) / 1_000_000  # ns -> ms                                                
        measured_perf += now                                                                                          
    result_b = measured_perf / ITERATIONS                                                                               
    print(f"{yellow}With Numpy (np.matmul){rst}")                                                                     
    print(                                                                                                            
       "-> 1000 dot products of 2000 vectors computed in an average of: "                                            
       + f"{yellow}{result_b:3.3f}ms{rst} (list length of the "                                                        
       + f"result:{len(dot_list)})" 
    )                                                                                                                 
    print(f"{green}numpy is {yellow}{result/result_b:3.3f} {green}time faster than pure python{rst}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
