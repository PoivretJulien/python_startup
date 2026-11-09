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


IDLE_VIDEO = dataclasses.replace(VIDEO, subscribe_interval=0, subscriber_ttl=0.3)


class VideoIdleConfigTests(unittest.TestCase):
    def test_zero_keepalive_allowed_negative_refused(self) -> None:
        self.assertEqual(VideoConfig(subscribe_interval=0).subscribe_interval, 0)
        with self.assertRaises(ValueError):
            VideoConfig(subscribe_interval=-1)


class VideoIdleModeTests(unittest.IsolatedAsyncioTestCase):
    """subscribe_interval = 0: the client subscribes once, then is silent while no video flows."""

    async def asyncSetUp(self) -> None:
        self.server = VideoServer(LINK, IDLE_VIDEO)
        _, port = await self.server.start()
        self.client = VideoClient(LINK, dataclasses.replace(IDLE_VIDEO, port=port))
        await self.client.connect()
        await until(lambda: self.client.subscribed)

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.server.close()

    async def test_subscribe_loop_ends_after_suback(self) -> None:
        await until(lambda: self.client._sub_task.done())
        self.assertEqual(len(self.server.subscribers), 1)
        # Nothing is left running: no timer will wake the process.
        self.assertEqual(asyncio.all_tasks(), {asyncio.current_task()})

    async def test_pause_longer_than_ttl_keeps_subscriber(self) -> None:
        await asyncio.sleep(IDLE_VIDEO.subscriber_ttl * 3)  # no video, no SUB
        self.assertEqual(await self.server.publish(b"after-pause"), 1)
        self.assertEqual((await self.client.next_frame(timeout=1)).data, b"after-pause")

    async def test_streaming_refreshes_subscription(self) -> None:
        # Stream for 4x the TTL: the client re-subscribes from the frames alone.
        loop = asyncio.get_running_loop()
        end = loop.time() + IDLE_VIDEO.subscriber_ttl * 4
        while loop.time() < end:
            await self.server.publish(os.urandom(2000))
            await asyncio.sleep(0.02)
        self.assertEqual(len(self.server.subscribers), 1)
        self.assertGreater(self.client.stats.frames, 20)

    async def test_dead_client_dropped_after_ttl_of_streaming(self) -> None:
        self.client._transport.close()  # crash: no BYE, no more SUBs
        loop = asyncio.get_running_loop()
        end = loop.time() + IDLE_VIDEO.subscriber_ttl * 3
        while loop.time() < end and self.server.subscribers:
            await self.server.publish(b"x")
            await asyncio.sleep(0.02)
        self.assertEqual(self.server.subscribers, {})

    async def test_wait_for_frame_does_not_consume(self) -> None:
        waiter = asyncio.create_task(self.client.wait_for_frame(timeout=1))
        await asyncio.sleep(0)
        await self.server.publish(b"one")
        await waiter
        self.assertEqual((await self.client.next_frame(timeout=1)).data, b"one")

    async def test_wait_for_subscriber(self) -> None:
        await asyncio.wait_for(self.server.wait_for_subscriber(), 0.5)  # already one
        await self.client.close()
        await until(lambda: not self.server.subscribers)
        waiter = asyncio.create_task(self.server.wait_for_subscriber())
        await asyncio.sleep(0.05)
        self.assertFalse(waiter.done())
        self.client = VideoClient(LINK, dataclasses.replace(IDLE_VIDEO, port=self.server.address[1]))
        await self.client.connect()
        await asyncio.wait_for(waiter, 1)


if __name__ == "__main__":
    unittest.main()
