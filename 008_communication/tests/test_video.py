"""video_link tests (no OpenCV needed: frames are random bytes)."""

from __future__ import annotations

import asyncio
import dataclasses
import os
import socket
import time
import unittest

from udp_link import LinkConfig
from video_link import VideoClient, VideoConfig, VideoServer
from video_link.protocol import (
    COOKIE_SIZE,
    MAX_CHUNK_PAYLOAD,
    Chunk,
    Kind,
    VideoProtocolError,
    decode,
    derive_video_key,
    encode,
    split_frame,
)

LINK = LinkConfig(server_ip="127.0.0.1", bind_ip="127.0.0.1", port=0, secret="video-test-secret-0123")
KEY = derive_video_key(LINK.key)
VIDEO = VideoConfig(port=0, subscribe_interval=0.05, subscriber_ttl=1.0)


async def until(predicate, timeout: float = 2.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


class ProtocolTests(unittest.TestCase):
    def test_roundtrip(self) -> None:
        c = Chunk(Kind.FRAME, b"abcd", 7, 2, 3, time.time(), b"jpegdata")
        self.assertEqual(decode(encode(c, KEY), KEY, max_clock_skew=60).payload, b"jpegdata")

    def test_tamper_detected(self) -> None:
        raw = bytearray(encode(Chunk(Kind.FRAME, b"abcd", 1, 0, 1, time.time(), b"x" * 50), KEY))
        raw[30] ^= 1
        with self.assertRaises(VideoProtocolError):
            decode(bytes(raw), KEY, max_clock_skew=60)

    def test_split_and_size(self) -> None:
        data = os.urandom(MAX_CHUNK_PAYLOAD * 3 + 5)
        chunks = split_frame(data, session=b"abcd", frame_id=1, ts=time.time())
        self.assertEqual(len(chunks), 4)
        self.assertEqual(b"".join(c.payload for c in chunks), data)
        self.assertTrue(all(len(encode(c, KEY)) <= 1200 for c in chunks))


class StreamTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.server = VideoServer(LINK, VIDEO)
        _, port = await self.server.start()
        self.client_video = dataclasses.replace(VIDEO, port=port)
        self.client = VideoClient(LINK, self.client_video)
        await self.client.connect()
        await until(lambda: self.server.subscribers)

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.server.close()

    async def test_frames_arrive_intact(self) -> None:
        for _ in range(3):
            data = os.urandom(50_000)
            await self.server.publish(data)
            frame = await self.client.next_frame(timeout=1)
            self.assertEqual(frame.data, data)
        self.assertEqual(len(self.client.frames), 3)

    async def test_lost_chunk_drops_only_that_frame(self) -> None:
        real, seen = self.client._on_datagram, []

        def lossy(data, addr):
            seen.append(data)
            if len(seen) == 2:  # lose the 2nd chunk of the first frame
                return
            real(data, addr)

        self.client._on_datagram = lossy
        await self.server.publish(os.urandom(10_000))
        second = os.urandom(10_000)
        await self.server.publish(second)
        frame = await self.client.next_frame(timeout=1)
        self.assertEqual(frame.data, second)
        self.assertEqual(self.client.stats.frames, 1)
        self.assertEqual(self.client.stats.incomplete, 1)

    async def test_replayed_old_frame_ignored(self) -> None:
        real, captured = self.client._on_datagram, []

        def recorder(data, addr):
            captured.append((data, addr))
            real(data, addr)

        self.client._on_datagram = recorder
        await self.server.publish(b"frame-one")
        await self.client.next_frame(timeout=1)
        old = list(captured)
        await self.server.publish(b"frame-two")
        await self.client.next_frame(timeout=1)
        for data, addr in old:
            real(data, addr)
        self.assertEqual(self.client.stats.frames, 2)
        self.assertEqual(self.client.latest.data, b"frame-two")
        self.assertGreaterEqual(self.client.stats.late, 1)

    async def test_bye_unsubscribes(self) -> None:
        await self.client.close()
        await until(lambda: not self.server.subscribers)

    async def test_frame_size_limit(self) -> None:
        with self.assertRaises(ValueError):
            await self.server.publish(b"x" * (VIDEO.max_frame_bytes + 1))

    async def test_wrong_secret_never_subscribes(self) -> None:
        bad_link = dataclasses.replace(LINK, secret="not-the-right-secret-000")
        intruder = VideoClient(bad_link, self.client_video)
        await intruder.connect()
        await asyncio.sleep(0.2)
        await intruder.close()
        self.assertEqual(len(self.server.subscribers), 1)
        self.assertGreater(self.server.stats.rejected, 0)


class CookieTests(unittest.IsolatedAsyncioTestCase):
    async def test_sub_without_cookie_only_gets_same_size_challenge(self) -> None:
        async with VideoServer(LINK, VIDEO) as server:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setblocking(False)
            sock.bind(("127.0.0.1", 0))
            sub = encode(Chunk(Kind.SUB, bytes(4), ts=time.time(), payload=bytes(COOKIE_SIZE)), KEY)
            loop = asyncio.get_running_loop()
            await loop.sock_sendto(sock, sub, server.address)
            reply = await asyncio.wait_for(loop.sock_recv(sock, 2048), 1)
            sock.close()
            self.assertEqual(server.subscribers, {})  # not subscribed yet
            self.assertEqual(len(reply), len(sub))  # no amplification
            self.assertIs(decode(reply, KEY, max_clock_skew=60).kind, Kind.CHALLENGE)


if __name__ == "__main__":
    unittest.main()
