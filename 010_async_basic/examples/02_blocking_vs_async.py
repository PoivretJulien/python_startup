"""Three slow jobs: one after the other, then all at once."""
import asyncio
import time


def download(name):                     # the ordinary version
    time.sleep(1)                       # stands for a slow network call
    return name + " done"


async def download_async(name):         # the async version
    await asyncio.sleep(1)              # waits WITHOUT holding the program
    return name + " done"


def blocking():
    return [download(n) for n in ("a", "b", "c")]


async def concurrent():
    return await asyncio.gather(
        download_async("a"), download_async("b"), download_async("c"))


start = time.perf_counter()
print(blocking(), f"{time.perf_counter() - start:.1f} s")

start = time.perf_counter()
print(asyncio.run(concurrent()), f"{time.perf_counter() - start:.1f} s")
