"""VIDEO CLIENT program -- run this on the PC.

Subscribes to the stream, shows it in a window (--display) and logs fps,
bandwidth, latency and lost frames.

    python -m video_link.run_client --config link.toml --display
    python -m video_link.run_client --config link.toml --save-dir snapshots
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from statistics import fmean

from .camera import require_cv2
from .client import VideoClient
from ._cli import add_video_args, configs_from_cli, run, setup_logging

log = logging.getLogger("video_client")


async def report(client: VideoClient, every: float = 5.0) -> None:
    """Log the stream statistics every `every` seconds, only while video arrives."""
    last_frames = last_bytes = 0
    while True:
        await client.wait_for_frame()  # sleeps while no video arrives
        await asyncio.sleep(every)
        s = client.stats
        frames, received = s.frames - last_frames, s.bytes - last_bytes
        last_frames, last_bytes = s.frames, s.bytes
        recent = list(client.frames)[-frames:] if frames else []
        latency = f"{fmean(f.latency for f in recent) * 1000:.0f} ms" if recent else "-"
        log.info(
            "%.1f fps | %.2f Mbit/s | latency %s | lost frames %d | rejected %d",
            frames / every, received * 8 / every / 1e6, latency, s.incomplete, s.rejected,
        )


async def display(client: VideoClient) -> None:
    cv2 = require_cv2()
    import numpy as np

    print("Video window open: press 'q' in the window to quit.")
    try:
        async for frame in client.stream():
            img = await asyncio.to_thread(cv2.imdecode, np.frombuffer(frame.data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            cv2.imshow("video_link", img)  # GUI calls stay on the main thread
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return
    finally:
        cv2.destroyAllWindows()


async def save_snapshots(client: VideoClient, directory: Path, every: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    while True:
        frame = await client.next_frame()
        path = directory / f"frame_{frame.frame_id:08d}.jpg"
        await asyncio.to_thread(path.write_bytes, frame.data)
        log.info("saved %s", path)
        await asyncio.sleep(every)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_video_args(parser)
    parser.add_argument("--display", action="store_true", help="show the video (needs opencv-python)")
    parser.add_argument("--save-dir", type=Path, help="save a JPEG snapshot here periodically")
    parser.add_argument("--save-every", type=float, default=5.0, help="seconds between snapshots")
    args = parser.parse_args()
    setup_logging(args.verbose)
    link, video = configs_from_cli(args)

    async with VideoClient(link, video) as client:
        reporter = asyncio.create_task(report(client))
        saver = asyncio.create_task(save_snapshots(client, args.save_dir, args.save_every)) if args.save_dir else None
        try:
            if args.display:
                await display(client)  # returns when 'q' is pressed
            else:
                await asyncio.Event().wait()  # run until Ctrl+C / SIGTERM
        finally:
            for task in (reporter, saver):
                if task:
                    task.cancel()


if __name__ == "__main__":
    run(main)
