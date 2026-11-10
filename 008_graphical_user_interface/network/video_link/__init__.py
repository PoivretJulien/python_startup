"""video_link: real-time, authenticated JPEG video streaming over UDP.

Separate from udp_link on purpose: sensor strings need reliable delivery
(ACK + retransmit), video needs the opposite (drop late frames). The two
run side by side and share the same link.toml (IP + secret).
"""

from .client import Frame, VideoClient, VideoClientStats
from .config import DEFAULT_VIDEO_PORT, VideoConfig
from .protocol import VideoProtocolError
from .server import VideoServer, VideoServerStats

__all__ = [
    "DEFAULT_VIDEO_PORT",
    "Frame",
    "VideoClient",
    "VideoClientStats",
    "VideoConfig",
    "VideoProtocolError",
    "VideoServer",
    "VideoServerStats",
]
