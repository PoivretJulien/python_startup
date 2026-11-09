from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import sys
from collections.abc import AsyncIterator, Callable, Coroutine
from typing import Any


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def stdin_lines() -> AsyncIterator[str]:
    """Non-blocking async reader for typed lines (POSIX: Linux / macOS)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    fd = sys.stdin.fileno()

    def on_ready() -> None:
        line = sys.stdin.readline()
        if not line:  # EOF (Ctrl+D)
            loop.remove_reader(fd)
            queue.put_nowait(None)
        else:
            queue.put_nowait(line)

    loop.add_reader(fd, on_ready)
    try:
        while (line := await queue.get()) is not None:
            if text := line.strip():
                yield text
    finally:
        loop.remove_reader(fd)


def parse_kv(text: str) -> dict[str, float | str]:
    """'temp=21.4;hum=40' -> {'temp': 21.4, 'hum': 40.0}. Non-numbers stay str."""
    out: dict[str, float | str] = {}
    for part in text.split(";"):
        key, sep, value = part.partition("=")
        if not sep or not key.strip():
            continue
        try:
            out[key.strip()] = float(value)
        except ValueError:
            out[key.strip()] = value.strip()
    return out


def run(main: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """asyncio.run() with graceful shutdown on Ctrl+C *and* SIGTERM (systemd, kill)."""

    async def runner() -> None:
        task = asyncio.current_task()
        assert task is not None
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):  # Windows
                loop.add_signal_handler(sig, task.cancel)
        with contextlib.suppress(asyncio.CancelledError):
            await main()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(runner())
