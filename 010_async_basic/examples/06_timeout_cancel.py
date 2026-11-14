"""Stopping work: a time limit, and a cancel by hand."""
import asyncio


async def slow_job(name):
    try:
        print(name, "started")
        await asyncio.sleep(10)
        print(name, "finished")         # never reached below
    except asyncio.CancelledError:
        print(name, "was cancelled, cleaning up")
        raise                           # always let the cancel continue


async def main():
    # 1. a time limit
    try:
        async with asyncio.timeout(0.5):
            await slow_job("job A")
    except TimeoutError:
        print("job A took too long")

    # 2. cancel by hand
    task = asyncio.create_task(slow_job("job B"))
    await asyncio.sleep(0.2)            # let it start
    task.cancel()                       # ask it to stop
    try:
        await task                      # wait until it really stopped
    except asyncio.CancelledError:
        print("job B is gone:", task.cancelled())


asyncio.run(main())
