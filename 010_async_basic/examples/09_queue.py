"""A queue: one side puts work in, workers take it out."""
import asyncio


async def producer(queue):
    for n in range(1, 7):
        await queue.put(n)
        print("produced", n)
        await asyncio.sleep(0.05)


async def worker(name, queue):
    while True:
        n = await queue.get()           # sleeps here while the queue is empty
        await asyncio.sleep(0.2)        # the work
        print(f"   {name} finished {n}")
        queue.task_done()


async def main():
    queue = asyncio.Queue(maxsize=3)    # put() waits when 3 items are waiting
    workers = [asyncio.create_task(worker(name, queue))
               for name in ("worker A", "worker B")]

    await producer(queue)
    await queue.join()                  # until every item is task_done()
    for w in workers:
        w.cancel()                      # they loop forever: stop them
    print("all work done")


asyncio.run(main())
