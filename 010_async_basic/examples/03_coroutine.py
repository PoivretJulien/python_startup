"""async def, await, asyncio.run: the three words."""
import asyncio


async def add(a, b):
    await asyncio.sleep(0.5)            # pretend this takes time
    return a + b


async def main():
    thing = add(1, 2)                   # calling it does NOT run it
    print(type(thing).__name__)         # it is a coroutine: a paused function
    result = await thing                # await runs it and gives the result
    print(result)

    print(await add(10, 20))            # the usual way: call and await at once


asyncio.run(main())                     # the door from normal code into async code
