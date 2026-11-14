"""async with, async for, and the async generator."""
import asyncio
from contextlib import asynccontextmanager


@asynccontextmanager
async def connection(name):
    print("open ", name)                # could await a real connect here
    await asyncio.sleep(0.1)
    try:
        yield name.upper()              # the value after "as"
    finally:
        await asyncio.sleep(0.1)
        print("close", name)            # runs even if the block raises


async def countdown(n):                 # async def + yield = async generator
    while n > 0:
        await asyncio.sleep(0.1)        # each value may need a wait
        yield n
        n -= 1


async def main():
    async with connection("db") as db:
        print("using", db)

    async for n in countdown(3):
        print(n)

    squares = [n * n async for n in countdown(3)]
    print(squares)


asyncio.run(main())
