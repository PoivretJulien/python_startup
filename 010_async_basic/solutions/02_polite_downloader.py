"""Practice 2: ten downloads, three at a time, 0.9 second in all."""
import asyncio
import time

start = time.perf_counter()


async def fetch(n, gate):
    async with gate:                            # three seats
        await asyncio.sleep(0.2 * (n % 3 + 1))  # 0.2, 0.4 or 0.6 s
        return f"page {n}"


async def main():
    gate = asyncio.Semaphore(3)
    tasks = [asyncio.create_task(fetch(n, gate)) for n in range(10)]
    received = 0
    try:
        async with asyncio.timeout(0.9):
            for finished in asyncio.as_completed(tasks):
                page = await finished
                received += 1
                print(f"{time.perf_counter() - start:.1f} s  {page}")
    except TimeoutError:
        for task in tasks:
            task.cancel()                       # do not leave the rest running
        print(f"time is over: {received} of {len(tasks)} pages received")


asyncio.run(main())
