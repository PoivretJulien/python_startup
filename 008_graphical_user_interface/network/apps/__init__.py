"""Runnable example programs built on the udp_link library.

This file marks `apps/` as a Python package, which is what lets you start the
programs with `python3 -m apps.<name>` (Windows: `py -m apps.<name>`) and lets
them import shared helpers with `from apps._common import ...`.

Programs (each one runs on its own, in its own terminal):

    time_server.py    test server: sends the OS time every 3 s (macOS, Linux, Windows)
    time_client.py    test client: prints the time messages it receives
    sensor_server.py  sensor example, server side: publishes sensor readings
    pc_client.py      sensor example, client side: stacks and batch-processes them

_common.py holds the small helpers they share (logging, Ctrl+C handling,
reading typed lines, parsing 'key=value' strings).
"""
