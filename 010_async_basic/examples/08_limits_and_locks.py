"""Semaphore: at most N at a time. Lock: one at a time."""
import asyncio
import time

start = time.perf_counter()


def clock():
    return f"{time.perf_counter() - start:.1f} s"


async def download(n, gate):
    async with gate:                    # wait here if 2 are already inside
        print(clock(), "start", n)
        await asyncio.sleep(0.5)


async def add_money(account, lock):
    async with lock:                    # read, wait, write: nobody in between
        balance = account["balance"]
        await asyncio.sleep(0.01)
        account["balance"] = balance + 10


async def main():
    gate = asyncio.Semaphore(2)
    await asyncio.gather(*(download(n, gate) for n in range(6)))
    print(clock(), "all downloaded")

    account = {"balance": 0}
    lock = asyncio.Lock()
    await asyncio.gather(*(add_money(account, lock) for _ in range(5)))
    print("balance:", account["balance"])


asyncio.run(main())
