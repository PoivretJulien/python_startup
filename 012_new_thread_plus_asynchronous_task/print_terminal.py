import threading as th
import asyncio

"""
    a 'light weight' python demo code that throw some arrows across the terminal
    at different speeds, each arrows are awaited asynchronously from a 
    simple spawned single thread.
"""
# Some terminal escape sequences (with special actions)
yellow = "\x1b[33m"
rst = "\x1b[0m"
hide_cursor = "\x1b[?25l"
show_cursor = "\x1b[?25h"
clear_line = "\x1b[2K"
clear_screen = "\x1b[2J\x1b[1;1H"

async def print_on_terminal(y: int, delay: float, stop: int) -> None:
    """Move an arrow along row `y`, one column per step, `delay` seconds apart."""
    for x in range(1, stop + 1):
        print(f"\x1b[{y};1H{clear_line}\x1b[{y};{x}H{yellow}-->{rst}", end="", flush=True)
        await asyncio.sleep(delay)

async def throw_arrow() -> None:
    delay_list = [0.03, 0.05, 0.01, 0.1, 0.15]
    stop = 60 # the length in x terminal characters of the arrow course
    # create all tasks first so they run concurrently, then wait for all of them
    tasks = [
        asyncio.create_task(print_on_terminal(row + 1, delay_list[row], stop))
        for row in range(len(delay_list))
    ]
    await asyncio.gather(*tasks)

def async_run(txt:str) -> None:
    "run the asynchronous loop/runtime"
    print(f"\x1b[6;1H{txt}")
    asyncio.run(throw_arrow())

# main is chosen to not be asynchronous.
def main() -> int:
    print(f"{clear_screen}{hide_cursor}", end="")
    try:
        # spawn a thread with asynchronous loop inside.
        t = th.Thread(target=async_run,args=("Ok, throwing some arrows...",))
        t.start()
        t.join()
    finally:
        # always restore the cursor, even on Ctrl+C
        print(f"\x1b[{6};1H{show_cursor}\n", end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
