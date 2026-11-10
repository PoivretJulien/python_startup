"""udp_link: async, acknowledged, authenticated string exchange over UDP."""

from .client import UDPClient
from .config import DEFAULT_PORT, LinkConfig, add_cli_args, config_from_cli
from .endpoint import Addr, DeliveryError, LinkStats, Message
from .protocol import AuthenticationError, PayloadTooLargeError, ProtocolError
from .server import UDPServer

__all__ = [
    "DEFAULT_PORT",
    "Addr",
    "AuthenticationError",
    "DeliveryError",
    "LinkConfig",
    "LinkStats",
    "Message",
    "PayloadTooLargeError",
    "ProtocolError",
    "UDPClient",
    "UDPServer",
    "add_cli_args",
    "config_from_cli",
]
__version__ = "1.0.0"
