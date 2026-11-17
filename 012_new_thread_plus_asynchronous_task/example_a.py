"""
Mind map: calling async code from a normal (sync) main.

    sync main()                      (main thread, plain function)
      |
      +-- threading.Thread(worker)   (one spawned thread)
            |
            +-- asyncio.run(gather_all())   (new event loop lives in this thread)
                  |
                  +-- asyncio.gather(task_a .. task_e)   (run concurrently)
                        each task: await asyncio.sleep() per step, then
                        update its own ProgressBar on its own terminal row

Why a thread? asyncio.run() blocks until its coroutine finishes, and it must
not be called from inside a running loop. Putting it in a thread lets the sync
main keep control (here it just waits with join(); in a real app it could draw
a GUI, read input, ...).

Standalone variant: it does not modify main.py and imports only progress_bar.
Run from this folder:  python3 example_a.py
"""

import asyncio
import threading

from progress_bar import ProgressBar

FIRST_ROW = 1  # bars occupy rows FIRST_ROW .. FIRST_ROW + 4
RESULT_ROW = FIRST_ROW + 6  # results are printed below the bars

# Color
COLORS = ("\x1b[0m", "\x1b[33m", "\x1b[31m", "\x1b[34m", "\x1b[32m", "\x1b[35m")

# (colors index (above), name, steps, seconds per step): different speeds so the bars move at different rates
WORK_ITEMS = [
    (1,"task_a", 10, 0.10),
    (3,"task_b", 8, 0.15),
    (1,"task_c", 12, 0.03),
    (3,"task_d", 6, 0.20),
    (1,"task_e", 9, 0.12),
]

async def simulated_work(color:int, name: str, row: int, steps: int, delay: float) -> str:
    """Virtual workload: each step waits `delay` seconds (stands in for real I/O)."""
    bar = ProgressBar(0, total=steps)
    anchor = (1, row)
    for step in range(1, steps + 1):
        await asyncio.sleep(delay)
        bar.progress = step
        bar.display(anchor,COLORS[color])
    return f"{name}: finished {steps} steps"


async def gather_all() -> list:
    """Run the five simulated tasks concurrently and collect their results."""
    tasks = []
    for i, (color,name, steps, delay) in enumerate(WORK_ITEMS):
        tasks.append(simulated_work(color,name, FIRST_ROW + i, steps, delay))
    return await asyncio.gather(*tasks, return_exceptions=True)


def run_in_thread() -> list | None:
    """Run the event loop in a spawned thread.
    Returns the results, or None if the worker raised. An exception in a thread
    is printed with its traceback by Python, so it isn't caught here.
    """
    outcome: dict = {}

    def worker() -> None:
        outcome["results"] = asyncio.run(gather_all())

    thread = threading.Thread(target=worker, name="async-worker")
    thread.start()
    thread.join()  # sync main waits here; it could do other work before this
    return outcome.get("results")


def main() -> int:
    print("\x1b[2J", end="", flush=True)
    results = run_in_thread()
    print(f"\x1b[{RESULT_ROW};1H", end="")
    if results is None:
        print("worker failed, see the traceback above")
        return 1
    for r in results:
        print(r if not isinstance(r, BaseException) else f"failed: {r!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
