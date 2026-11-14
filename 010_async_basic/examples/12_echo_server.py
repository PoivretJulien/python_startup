"""A complete program: a TCP server and three clients, in one event loop."""
import asyncio

HOST, PORT = "127.0.0.1", 8888


async def handle(reader, writer):       # runs once PER client, as a task
    async for line in reader:           # one line at a time, until the client leaves
        text = line.decode().strip()
        print(f"server: got {text!r}")
        writer.write(f"echo: {text}\n".encode())
        await writer.drain()            # wait until the bytes are really sent
    writer.close()
    await writer.wait_closed()


async def client(name, delay):
    reader, writer = await asyncio.open_connection(HOST, PORT)
    for word in ("hello", "bye"):
        writer.write(f"{name} says {word}\n".encode())
        await writer.drain()
        answer = await reader.readline()
        print(f"{name}: received {answer.decode().strip()!r}")
        await asyncio.sleep(delay)
    writer.close()
    await writer.wait_closed()


async def main():
    server = await asyncio.start_server(handle, HOST, PORT)
    async with server:                  # the server listens inside this block
        async with asyncio.TaskGroup() as group:
            group.create_task(client("ann", 0.2))
            group.create_task(client("bob", 0.3))
            group.create_task(client("cat", 0.1))
    print("all clients left, server closed")


asyncio.run(main())
