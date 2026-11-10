"""video_link for programs that are not async: a window, a game loop, a script.

The twin of ``udp_link.bridge.LinkThread``, for pictures instead of strings.
``VideoServer`` / ``VideoClient`` live inside an asyncio loop; ``VideoThread``
runs that loop in a background thread and hands you ready-to-draw pictures::

    from video_link.bridge import VideoThread

    def on_frame(picture):               # an (H, W, 3) RGB uint8 NumPy array
        ...                              # (called from the video thread)

    video = VideoThread("server", source="test", on_frame=on_frame).start()
    video = VideoThread("client", on_frame=on_frame).start()
    ...
    video.close()

* ``"server"`` owns the camera: it captures ``source``, encodes each picture as
  JPEG and streams it to every subscribed client. ``on_frame`` gets the same
  pictures, as a local preview.
* ``"client"`` subscribes to ``server_ip`` and gets each picture that arrives.

Without a callback, read ``video.latest`` (the newest picture, or ``None``) and
``video.frames`` (how many arrived) whenever you like.

Video is the opposite of ``udp_link``: nothing is acknowledged or sent again. A
late picture is worthless, so a lost one is simply skipped and the next one
shown — the stream never falls behind.

The configuration is the usual ``link.toml`` (address + secret), plus an
optional ``[video]`` table (``port = 50556`` by default). Needs OpenCV
(``opencv-python-headless`` is enough: nothing here opens an OpenCV window).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import threading
from collections.abc import Callable
from typing import Any

from udp_link.config import LinkConfig

from .camera import capture, require_cv2
from .client import Frame, VideoClient
from .config import VideoConfig
from .server import VideoServer

log = logging.getLogger(__name__)

FrameHandler = Callable[[Any], None]  # receives an (H, W, 3) RGB uint8 array


class VideoThread:
    """A ``VideoServer`` (camera) or ``VideoClient`` (viewer) in its own thread."""

    def __init__(
        self,
        role: str,
        config: str | os.PathLike[str] | None = None,
        *,
        on_frame: FrameHandler | None = None,
        source: str = "test",
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        quality: int = 70,
    ) -> None:
        """``role`` is ``"server"`` or ``"client"``; ``config`` the link.toml path.

        The remaining options matter on the server only: ``source`` is
        ``"test"`` (a moving pattern, no camera needed), a camera index
        (``"0"``), a device path or a video file; ``width`` / ``height`` /
        ``fps`` are requested from the camera; ``quality`` is the JPEG quality
        (1-100: lower = smaller datagrams, blockier picture).
        """
        if role not in ("server", "client"):
            raise ValueError(f"role must be 'server' or 'client', not {role!r}")
        self.role = role
        self.link = LinkConfig.load(config)
        self.video = VideoConfig.load(config)
        self.on_frame = on_frame
        self.source = str(source)
        self.size = (int(width), int(height))
        self.fps = float(fps)
        self.quality = int(quality)

        self.latest: Any = None  # newest picture: (H, W, 3) RGB uint8, or None
        self.frames = 0  # pictures captured (server) / received (client)
        self.latency = 0.0  # client: capture-to-arrival delay of the newest, seconds
        self.viewers = 0  # server: clients currently subscribed
        self.connected = False  # server: listening / client: subscribed
        self.error: BaseException | None = None  # why the stream stopped, if it did

        self._cv2 = require_cv2()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task[None] | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"video-link-{role}", daemon=True)

    # ------------------------------------------------------------ any thread
    def start(self) -> VideoThread:
        """Start the video thread and return at once."""
        self._thread.start()
        self._ready.wait()
        return self

    def close(self, timeout: float = 3.0) -> None:
        """Stop streaming, release the camera and wait for the thread to end."""
        loop, task = self._loop, self._task
        if loop is not None and task is not None and not loop.is_closed():
            with contextlib.suppress(RuntimeError):  # the loop closed meanwhile
                loop.call_soon_threadsafe(task.cancel)
        if self._thread.is_alive():
            self._thread.join(timeout)

    def __enter__(self) -> VideoThread:
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---------------------------------------------------------- video thread
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # port in use, camera refused, ...
            self.error = exc
            log.error("video link (%s) stopped: %s", self.role, exc)
        finally:
            self.connected = False
            self._ready.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.current_task()
        self._ready.set()
        try:
            if self.role == "server":
                await self._serve()
            else:
                await self._watch()
        except asyncio.CancelledError:
            pass  # close(): the normal way out

    async def _serve(self) -> None:
        width, height = self.size
        async with VideoServer(self.link, self.video) as server:
            self.connected = True
            frames = capture(
                self.source, width=width, height=height, fps=self.fps, quality=self.quality
            )
            async with contextlib.aclosing(frames):  # closing it releases the camera
                async for jpeg, captured_at in frames:
                    await server.publish(jpeg, captured_at)  # skipped when nobody watches
                    self.viewers = len(server.subscribers)
                    self._show(jpeg)

    async def _watch(self) -> None:
        def on_frame(frame: Frame) -> None:
            self.latency = frame.latency
            self._show(frame.data)

        async with VideoClient(self.link, self.video, on_frame=on_frame) as client:
            while True:  # nothing to do here: frames arrive through on_frame
                self.connected = client.subscribed
                await asyncio.sleep(0.5)

    def _show(self, jpeg: bytes) -> None:
        """JPEG bytes -> RGB picture -> ``latest`` / ``on_frame``."""
        import numpy as np

        cv2 = self._cv2
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return  # a damaged picture: wait for the next one
        # OpenCV orders the channels blue-green-red; everyone else wants RGB.
        self.latest = np.ascontiguousarray(bgr[:, :, ::-1])
        self.frames += 1
        if self.on_frame is not None:
            self.on_frame(self.latest)
