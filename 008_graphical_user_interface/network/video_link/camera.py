"""Frame sources (need OpenCV): a real camera/video file, or a test pattern.

    server_side:  pip install opencv-python-headless
    PC + window:  pip install opencv-python

All OpenCV calls run on ONE dedicated worker thread, so capture and JPEG
encoding never block the asyncio loop, and the camera is never touched by
two threads at once (safe release on shutdown).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any


def require_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for camera/display: "
            "pip install opencv-python-headless  (server_side)  or  opencv-python  (PC with a window)"
        ) from exc
    return cv2


def _open_camera(cv2: Any, source: str, width: int, height: int, fps: float) -> tuple[Callable[[], Any], Callable[[], None]]:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low where supported

    def grab() -> Any:
        ok, img = cap.read()
        if not ok and not source.isdigit():  # video file: loop it
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = cap.read()
        return img if ok else None

    return grab, cap.release


def _test_pattern(cv2: Any, width: int, height: int) -> tuple[Callable[[], Any], Callable[[], None]]:
    import numpy as np

    xs = np.linspace(0, 255, width, dtype=np.float32)
    counter = 0

    def grab() -> Any:
        nonlocal counter
        counter += 1
        shift = (counter * 4) % 256
        img = np.empty((height, width, 3), dtype=np.uint8)
        img[:, :, 0] = ((xs + shift) % 256).astype(np.uint8)
        img[:, :, 1] = 96
        img[:, :, 2] = ((255 - xs + shift) % 256).astype(np.uint8)
        x = int((counter * 6) % width)
        cv2.rectangle(img, (x, 0), (x + 20, height), (255, 255, 255), -1)
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        cv2.putText(img, f"#{counter} {stamp}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(img, f"#{counter} {stamp}", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return img

    return grab, lambda: None


async def capture(
    source: str = "0",
    *,
    width: int = 640,
    height: int = 480,
    fps: float = 15.0,
    quality: int = 70,
) -> AsyncIterator[tuple[bytes, float]]:
    """Yield (jpeg_bytes, capture_time) at most `fps` times per second.

    source: camera index ("0"), device path ("/dev/video0"), file/URL, or "test".
    """
    cv2 = require_cv2()
    loop = asyncio.get_running_loop()
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera")
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]

    def open_source() -> tuple[Callable[[], Any], Callable[[], None]]:
        if source == "test":
            return _test_pattern(cv2, width, height)
        return _open_camera(cv2, source, width, height, fps)

    grab, release = await loop.run_in_executor(worker, open_source)

    def grab_jpeg() -> tuple[bytes, float] | None:
        img = grab()
        if img is None:
            return None
        stamp = time.time()
        ok, buf = cv2.imencode(".jpg", img, params)
        return (buf.tobytes(), stamp) if ok else None

    period = 1.0 / fps
    next_tick = loop.time()
    failures = 0
    try:
        while True:
            result = await loop.run_in_executor(worker, grab_jpeg)
            if result is None:
                failures += 1
                if failures >= 10:
                    raise RuntimeError(f"video source {source!r} stopped delivering frames")
                await asyncio.sleep(0.1)
                continue
            failures = 0
            yield result
            next_tick += period
            delay = next_tick - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                next_tick = loop.time()  # running behind: don't burst to catch up
    finally:
        # Queued after any in-flight read on the same single thread -> safe.
        await asyncio.shield(loop.run_in_executor(worker, release))
        worker.shutdown(wait=False)
