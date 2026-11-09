"""Run: python -m unittest -v   (from the project root; stdlib only)"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import unittest

from udp_link import (
    AuthenticationError,
    DeliveryError,
    LinkConfig,
    PayloadTooLargeError,
    UDPClient,
    UDPServer,
)
from udp_link.protocol import Kind, Packet, StalePacketError, decode, encode

SECRET = "test-secret-0123456789"
BASE = LinkConfig(
    server_ip="127.0.0.1", bind_ip="127.0.0.1", port=0, secret=SECRET,
    ack_timeout=0.05, max_retries=4, max_backoff=0.2, heartbeat_interval=0.2,
)


class ProtocolTests(unittest.TestCase):
    def test_roundtrip_unicode(self) -> None:
        pkt = Packet(Kind.DATA, "abcd", 42, time.time(), "temp=21.5°C;hum=40")
        out = decode(encode(pkt, BASE.key), BASE.key, max_clock_skew=60)
        self.assertEqual((out.kind, out.seq, out.payload), (Kind.DATA, 42, pkt.payload))

    def test_tampered_packet_rejected(self) -> None:
        raw = bytearray(encode(Packet(Kind.DATA, "abcd", 1, time.time(), "x=1"), BASE.key))
        raw[-3] ^= 0x01
        with self.assertRaises(AuthenticationError):
            decode(bytes(raw), BASE.key, max_clock_skew=60)

    def test_wrong_key_rejected(self) -> None:
        other = dataclasses.replace(BASE, secret="another-secret-987654321")
        raw = encode(Packet(Kind.DATA, "abcd", 1, time.time(), "x"), BASE.key)
        with self.assertRaises(AuthenticationError):
            decode(raw, other.key, max_clock_skew=60)

    def test_stale_packet_rejected(self) -> None:
        raw = encode(Packet(Kind.DATA, "abcd", 1, time.time() - 600, "x"), BASE.key)
        with self.assertRaises(StalePacketError):
            decode(raw, BASE.key, max_clock_skew=60)

    def test_oversized_payload(self) -> None:
        with self.assertRaises(PayloadTooLargeError):
            encode(Packet(Kind.DATA, "abcd", 1, time.time(), "x" * 2000), BASE.key)

    def test_short_secret_refused(self) -> None:
        with self.assertRaises(ValueError):
            LinkConfig(secret="short")


class LinkTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = UDPServer(BASE)
        _, port = await self.server.start()
        self.client_cfg = dataclasses.replace(BASE, port=port)
        self.client = UDPClient(self.client_cfg)
        await self.client.connect()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.server.close()

    async def test_client_to_server_stacked(self) -> None:
        for i in range(5):
            await self.client.send(f"msg={i}")
        self.assertEqual([m.text for m in self.server.received], [f"msg={i}" for i in range(5)])
        self.assertEqual(len(self.server.drain()), 5)
        self.assertEqual(self.server.received, [])

    async def test_server_broadcast_to_client_stacked(self) -> None:
        self.assertEqual(len(self.server.peers), 1)
        result = await self.server.broadcast("temp=21.0")
        self.assertEqual(list(result.values()), [True])
        msg = await self.client.recv(timeout=1)
        self.assertEqual(msg.text, "temp=21.0")

    async def test_concurrent_sends(self) -> None:
        await asyncio.gather(*(self.client.send(f"n={i}") for i in range(100)))
        self.assertEqual(sorted(m.text for m in self.server.received), sorted(f"n={i}" for i in range(100)))

    async def test_retransmission_survives_packet_loss(self) -> None:
        ep = self.client._endpoint()
        real_send, dropped = ep._send_raw, []

        def lossy(pkt, addr, transport=None):  # drop the first 2 DATA transmissions
            if pkt.kind is Kind.DATA and len(dropped) < 2:
                dropped.append(pkt.seq)
                return
            real_send(pkt, addr, transport)

        ep._send_raw = lossy
        await self.client.send("important")
        self.assertEqual([m.text for m in self.server.received], ["important"])
        self.assertGreaterEqual(self.client.stats.retransmits, 2)

    async def test_lost_ack_does_not_duplicate(self) -> None:
        sep = self.server._endpoint()
        real_send, dropped = sep._send_raw, []

        def drop_first_ack(pkt, addr, transport=None):
            if pkt.kind is Kind.ACK and not dropped:
                dropped.append(pkt.seq)
                return
            real_send(pkt, addr, transport)

        sep._send_raw = drop_first_ack
        await self.client.send("once")
        self.assertEqual([m.text for m in self.server.received], ["once"])
        self.assertEqual(self.server.stats.duplicates, 1)

    async def test_wrong_secret_client_cannot_connect(self) -> None:
        bad = dataclasses.replace(self.client_cfg, secret="wrong-secret-0000000000")
        with self.assertRaises(DeliveryError):
            await UDPClient(bad, wait_for_server=False).connect()
        self.assertEqual(len(self.server.peers), 1)  # intruder never registered
        self.assertGreater(self.server.stats.rejected, 0)

    async def test_send_fails_when_server_gone(self) -> None:
        await self.server.close()
        with self.assertRaises(DeliveryError):
            await self.client.send("anyone?")

    async def test_bye_removes_peer(self) -> None:
        await self.client.close()
        await asyncio.sleep(0.05)
        self.assertEqual(self.server.peers, [])

    async def test_inbox_limit_drops_oldest(self) -> None:
        await self.client.close()
        await self.server.close()
        self.server = UDPServer(dataclasses.replace(BASE, inbox_limit=3))
        _, port = await self.server.start()
        self.client = UDPClient(dataclasses.replace(BASE, port=port))
        await self.client.connect()
        for i in range(5):
            await self.client.send(str(i))
        self.assertEqual([m.text for m in self.server.received], ["2", "3", "4"])

    async def test_async_handler_called(self) -> None:
        seen: list[str] = []

        async def handler(msg):
            seen.append(msg.text)

        self.server._endpoint()._on_message = handler
        await self.client.send("cb")
        await asyncio.sleep(0.01)
        self.assertEqual(seen, ["cb"])


if __name__ == "__main__":
    unittest.main()
