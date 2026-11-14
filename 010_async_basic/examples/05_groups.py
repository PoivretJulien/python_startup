"""Three ways to run many coroutines and collect the results."""
import asyncio


async def fetch(name, seconds):
    await asyncio.sleep(seconds)
    return f"{name} ({seconds} s)"


async def main():
    # gather: results come back in the order you WROTE them
    results = await asyncio.gather(fetch("slow", 0.3), fetch("fast", 0.1))
    print("gather      ", results)

    # TaskGroup: the block ends when every task is finished
    async with asyncio.TaskGroup() as group:
        slow = group.create_task(fetch("slow", 0.3))
        fast = group.create_task(fetch("fast", 0.1))
    print("TaskGroup   ", [slow.result(), fast.result()])

    # as_completed: results come back in the order they FINISH
    jobs = [fetch("slow", 0.3), fetch("fast", 0.1)]
    for finished in asyncio.as_completed(jobs):
        print("as_completed", await finished)

    # one failure: gather can hand the error back as a value
    async def broken():
        raise ValueError("no network")

    results = await asyncio.gather(fetch("fine", 0.1), broken(),
                                   return_exceptions=True)
    print("with errors ", results)


asyncio.run(main())
