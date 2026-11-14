"""One blocking call freezes everything. to_thread is the way out."""
import asyncio
import time


async def heartbeat():
    while True:                         # a sign of life every 0.25 s
        print("  tick")
        await asyncio.sleep(0.25)


def heavy():                            # ordinary, blocking code
    time.sleep(1)
    return "heavy result"


async def main():
    beat = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)              # let the heartbeat start

    print("WRONG: calling blocking code directly")
    heavy()                             # the loop is stuck here for 1 s
    await asyncio.sleep(0.1)

    print("RIGHT: sending it to a thread")
    print(" ", await asyncio.to_thread(heavy))

    beat.cancel()


asyncio.run(main())
