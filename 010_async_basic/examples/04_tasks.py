"""await runs one thing and waits. A task runs in the background."""
import asyncio
import time


async def cook(dish, seconds):
    print("  start", dish)
    await asyncio.sleep(seconds)
    print("  ready", dish)
    return dish


async def one_by_one():
    await cook("rice", 1.0)             # wait for the rice...
    await cook("soup", 1.0)             # ...and only then start the soup


async def together():
    rice = asyncio.create_task(cook("rice", 1.0))    # starts in the background
    soup = asyncio.create_task(cook("soup", 1.0))    # starts in the background
    print("  two tasks created, none has started yet")
    await rice                          # now wait for the results
    await soup


async def main():
    for recipe in (one_by_one, together):
        start = time.perf_counter()
        await recipe()
        print(f"{recipe.__name__}: {time.perf_counter() - start:.1f} s")


asyncio.run(main())
