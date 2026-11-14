"""Where lambda meets asyncio: callbacks, and delayed calls."""
import asyncio
import functools
import time


def resize(name, width, height):        # ordinary, blocking code
    time.sleep(0.2)
    return f"{name} resized to {width}x{height}"


async def compute():
    await asyncio.sleep(0.1)
    return 42


async def main():
    loop = asyncio.get_running_loop()

    # 1. call a plain function later
    loop.call_later(0.3, lambda: print("0.3 s passed"))

    # 2. be told when a task finishes
    task = asyncio.create_task(compute())
    task.add_done_callback(lambda t: print("task result:", t.result()))

    # 3. give arguments to a function that will be called for you
    #    (run_in_executor refuses keyword arguments: wrap the call)
    print(await loop.run_in_executor(None, lambda: resize("a.png", width=64, height=64)))
    print(await loop.run_in_executor(None, functools.partial(resize, "b.png", width=32, height=32)))
    print(await asyncio.to_thread(resize, "c.png", 16, 16))     # the simplest

    # 4. a lambda cannot contain await, but it can RETURN a coroutine
    make = lambda: compute()
    print("made:", await make())

    await asyncio.sleep(0.2)            # leave time for the 0.3 s callback


asyncio.run(main())
