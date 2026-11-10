"""VIDEO SERVER program -- run this on server_side.

Captures a camera (or a test pattern), encodes JPEG and streams it to every
subscribed client. While nobody is subscribed the camera is closed and the
program sleeps: no capture, no encoding, no timer.

    python -m video_link.run_server --config link.toml --source 0 --fps 15
    python -m video_link.run_server --config link.toml --source test   # no camera
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging

from .camera import capture
from .server import VideoServer
from ._cli import add_video_args, configs_from_cli, run, setup_logging

log = logging.getLogger("video_server")


async def report(server: VideoServer, every: float = 5.0) -> None:
    """Log the stream statistics every `every` seconds, only while someone watches."""
    last_frames = last_bytes = 0
    while True:
        await server.wait_for_subscriber()  # sleeps while nobody is subscribed
        await asyncio.sleep(every)
        s = server.stats
        frames, sent = s.frames_sent - last_frames, s.bytes_sent - last_bytes
        last_frames, last_bytes = s.frames_sent, s.bytes_sent
        log.info(
            "%d subscriber(s) | %.1f fps | %.2f Mbit/s | skipped %d | rejected %d",
            len(server.subscribers), frames / every, sent * 8 / every / 1e6, s.frames_skipped, s.rejected,
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_video_args(parser)
    parser.add_argument("--source", default="0", help='camera index, /dev/videoN, file/URL, or "test"')
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--quality", type=int, default=70, help="JPEG quality 1-100")
    args = parser.parse_args()
    setup_logging(args.verbose)
    link, video = configs_from_cli(args)

    async with VideoServer(link, video) as server, asyncio.TaskGroup() as tg:
        tg.create_task(report(server))
        while True:
            # Camera off until a client subscribes.
            await server.wait_for_subscriber()
            log.info("streaming: camera on")
            frames = capture(args.source, width=args.width, height=args.height, fps=args.fps, quality=args.quality)
            async with contextlib.aclosing(frames):  # closing it releases the camera
                async for jpeg, captured_at in frames:
                    await server.publish(jpeg, captured_at)
                    if not server.subscribers:  # last viewer left (BYE) or timed out
                        break
            log.info("no subscriber: camera off, waiting")


if __name__ == "__main__":
    run(main)
