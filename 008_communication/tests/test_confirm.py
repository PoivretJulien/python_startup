"""Tests for the application-level confirmation example (apps/confirm_*.py)."""

from __future__ import annotations

import dataclasses
import unittest

from udp_link import DeliveryError, LinkConfig, UDPClient, UDPServer

from apps.confirm_client import Requester, RequestFailed
from apps.confirm_server import Device, handle_request

CFG = LinkConfig(
    server_ip="127.0.0.1", bind_ip="127.0.0.1", port=0, secret="confirm-test-secret-0123",
    ack_timeout=0.05, max_retries=3, max_backoff=0.2, heartbeat_interval=0.5,
)


class ConfirmTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.device = Device()
        self.server = UDPServer(CFG, on_message=lambda msg: handle_request(self.server, self.device, msg))
        _, port = await self.server.start()
        self.requester = Requester()
        self.client = UDPClient(dataclasses.replace(CFG, port=port), on_message=self.requester.on_message)
        await self.client.connect()
        self.requester.client = self.client

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.server.close()

    async def test_ok_reply_after_processing(self) -> None:
        result, ack_ms, done_ms = await self.requester.request("led", "on")
        self.assertEqual(result, "led is on")
        self.assertEqual(self.device.led, "on")  # the work really happened
        self.assertLess(ack_ms, done_ms)  # received first, processed later

    async def test_error_reply(self) -> None:
        with self.assertRaises(RequestFailed):
            await self.requester.request("led", "blue")
        with self.assertRaises(RequestFailed):
            await self.requester.request("reboot")

    async def test_concurrent_requests_matched_by_id(self) -> None:
        import asyncio
        results = await asyncio.gather(*(self.requester.request("add", "1") for _ in range(10)))
        self.assertEqual(self.device.counter, 10)
        self.assertEqual(sorted(r[0] for r in results), sorted(f"counter is {i}" for i in range(1, 11)))

    async def test_no_reply_times_out(self) -> None:
        async def slow(cmd: str, value: str) -> str:
            import asyncio
            await asyncio.sleep(1)
            return "late"
        self.device.execute = slow
        with self.assertRaises(TimeoutError):
            await self.requester.request("status", timeout=0.2)
        self.assertEqual(self.requester._pending, {})

    async def test_server_down_not_received(self) -> None:
        await self.server.close()
        with self.assertRaises(DeliveryError):
            await self.requester.request("status")
