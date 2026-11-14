"""Practice 3: a chat server. What one client writes, all the others read."""
import asyncio

HOST, PORT = "127.0.0.1", 8889
clients = set()                         # the writers of everyone connected


async def handle(reader, writer):
    clients.add(writer)
    try:
        async for line in reader:
            for other in clients:       # no await in this loop: nobody can
                if other is not writer: # join or leave while we walk the set
                    other.write(line)
    finally:
        clients.discard(writer)         # also when the client vanishes
        writer.close()
        await writer.wait_closed()


async def listener(name):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    line = await reader.readline()      # sleeps until somebody speaks
    print(f"{name} heard: {line.decode().strip()}")
    writer.close()
    await writer.wait_closed()


async def speaker(name):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    await asyncio.sleep(0.2)            # let the listeners arrive
    writer.write(f"{name}: hello everyone\n".encode())
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle, HOST, PORT)
    async with server:
        async with asyncio.TaskGroup() as group:
            group.create_task(listener("bob"))
            group.create_task(listener("cat"))
            group.create_task(speaker("ann"))
    print("everybody left")


asyncio.run(main())
