"""Small helpers shared by every program in apps/.

The leading underscore in the file name means "internal": it is not a program
you run, only a module the programs import from.
"""

# Lets type hints such as `str | None` work everywhere and be checked lazily.
from __future__ import annotations

import asyncio  # the async runtime: event loop, tasks, queues, sleeping
import contextlib  # contextlib.suppress(...) = "ignore this exception here"
import logging  # timestamped log lines instead of bare print()
import signal  # names of OS signals (SIGINT = Ctrl+C, SIGTERM = `kill`)
import sys  # access to stdin (the keyboard input of the terminal)
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any


def setup_logging(verbose: bool) -> None:
    """Configure log output once, at program start.

    Every log line looks like:  16:51:26 INFO    time_server: -> time=...
    With --verbose (-v) the DEBUG lines are shown too (dropped packets, etc.).
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,  # minimum level that gets printed
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",  # time, level, logger name, text
        datefmt="%H:%M:%S",  # time only, no date
    )


async def stdin_lines() -> AsyncIterator[str]:
    """Yield each line typed in the terminal, without blocking the program.

    A normal `input()` would freeze the whole async program while it waits for
    the user. Here the event loop watches stdin instead and wakes us up only
    when a full line is ready, so sending/receiving keeps running meanwhile.

    Usage:  async for line in stdin_lines(): ...

    Only works on macOS and Linux: Windows' event loop has no add_reader().
    """
    loop = asyncio.get_running_loop()
    # The callback below puts each typed line here; the loop at the bottom
    # takes them out. None is used as an "end of input" marker.
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    fd = sys.stdin.fileno()  # the OS-level number of the stdin stream

    def on_ready() -> None:
        # Called by the event loop each time stdin has data to read.
        line = sys.stdin.readline()
        if not line:  # empty string = EOF (Ctrl+D): stop watching stdin
            loop.remove_reader(fd)
            queue.put_nowait(None)
        else:
            queue.put_nowait(line)

    loop.add_reader(fd, on_ready)  # "call on_ready whenever stdin is readable"
    try:
        # Wait for lines until the EOF marker arrives; skip empty lines.
        while (line := await queue.get()) is not None:
            if text := line.strip():  # strip() removes the trailing newline/spaces
                yield text
    finally:
        # Always stop watching stdin, even if the caller stops early.
        loop.remove_reader(fd)


def parse_kv(text: str) -> dict[str, float | str]:
    """Turn a 'key=value;key=value' string into a dict.

    'temp=21.4;hum=40'      -> {'temp': 21.4, 'hum': 40.0}
    'time=16:30:03;tick=2'  -> {'time': '16:30:03', 'tick': 2.0}

    Values that look like numbers become floats; anything else stays a string.
    Pieces without '=' (or with an empty key) are ignored.
    """
    out: dict[str, float | str] = {}
    for part in text.split(";"):  # 'temp=21.4;hum=40' -> ['temp=21.4', 'hum=40']
        key, sep, value = part.partition("=")  # 'temp=21.4' -> ('temp', '=', '21.4')
        if not sep or not key.strip():  # no '=' found, or nothing before it
            continue
        try:
            out[key.strip()] = float(value)  # numeric value
        except ValueError:
            out[key.strip()] = value.strip()  # not a number: keep the text
    return out


def parse_fields(text: str) -> dict[str, str]:
    """Like parse_kv(), but every value stays text: 'id=7;value=5' -> {'id': '7', 'value': '5'}.

    Use it when exact values matter (ids, commands, user input): parse_kv()
    would turn '5' into 5.0 and '007' into 7.0.
    """
    out: dict[str, str] = {}
    for part in text.split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip():
            out[key.strip()] = value.strip()
    return out


def run(main: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Start an async `main()` and shut it down cleanly on Ctrl+C or SIGTERM.

    Plain asyncio.run() handles Ctrl+C, but not SIGTERM, which is what `kill`
    and service managers (systemd) send. Here both signals cancel the main
    task, so every `async with` block exits properly: the client sends its
    goodbye (BYE) and sockets are closed, instead of the process just dying.
    """

    async def runner() -> None:
        task = asyncio.current_task()  # the task running this coroutine
        assert task is not None
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            # Windows doesn't support signal handlers in asyncio; there,
            # Ctrl+C is still handled by asyncio.run() itself.
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, task.cancel)  # signal -> cancel main
        # Cancellation is our normal way to stop, so don't report it as an error.
        with contextlib.suppress(asyncio.CancelledError):
            await main()

    # On Windows, Ctrl+C surfaces as KeyboardInterrupt: exit quietly too.
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(runner())
