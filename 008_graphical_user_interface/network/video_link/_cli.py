from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
from collections.abc import Callable, Coroutine
from typing import Any

from udp_link import LinkConfig, add_cli_args, config_from_cli

from .config import VideoConfig


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def add_video_args(parser: argparse.ArgumentParser) -> None:
    add_cli_args(parser)  # --config --server-ip --bind-ip --port (shared with udp_link)
    parser.add_argument("--video-port", type=int, help="UDP port for video (default 50556)")
    parser.add_argument("-v", "--verbose", action="store_true")


def configs_from_cli(args: argparse.Namespace) -> tuple[LinkConfig, VideoConfig]:
    return config_from_cli(args), VideoConfig.load(args.config, port=args.video_port)


def run(main: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """asyncio.run() with graceful shutdown on Ctrl+C and SIGTERM."""

    async def runner() -> None:
        task = asyncio.current_task()
        assert task is not None
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, task.cancel)
        with contextlib.suppress(asyncio.CancelledError):
            await main()

    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(runner())
