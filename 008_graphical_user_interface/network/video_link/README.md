# video_link

This module streams live JPEG video (MJPEG) over UDP on the same LAN, from a camera to one or more viewers. It runs next to [`udp_link`](../udp_link/README.md), which carries the strings, and reuses its `link.toml`: the same server IP and the same secret, on its own port (default `50556`). Typical setup:

* **server_side**: runs the **video server**, captures a camera (or a test pattern), encodes JPEG and sends each frame to every subscribed client.
* **Client side**: runs the **video client**, subscribes, reassembles the frames and shows them, saves them or hands them to your code.

The protocol needs only the standard library. OpenCV is needed only to capture and display images.

## What runs where

| | server_side | Client side |
|---|---|---|
| Quick test (no camera) | `python3 -m video_link.run_server --source test` | `python3 -m video_link.run_client` |
| Camera | `python3 -m video_link.run_server --source 0` | `python3 -m video_link.run_client --display` |
| OpenCV | `pip install opencv-python-headless` | `pip install opencv-python` |
| Firewall | allow inbound UDP 50556, see [Open the UDP port](#open-the-udp-port-firewall) | nothing to open |
| `link.toml` | the udp_link file, optional `[video]` table | copy of the server's file |
| Your code goes in | `publish(jpeg)` in your loop (see [Library API](#library-api)) | `client.stream()` / `on_frame` |

Add `--config link.toml` to every command. On Windows, type `py` instead of `python3`.

## Project layout

```
video_link/
├── protocol.py    # chunk format, HMAC tag, frame splitting, cookie messages
├── server.py      # VideoServer: subscribers, publish(jpeg), wait_for_subscriber()
├── client.py      # VideoClient: subscribe, reassemble, frames buffer
├── camera.py      # OpenCV capture / test pattern -> JPEG (worker thread)
├── config.py      # VideoConfig ([video] table in link.toml)
├── _cli.py        # shared command-line options and clean shutdown
├── run_server.py  # program for server_side
└── run_client.py  # program for the client side
```

## Protocol and security

| Concern | Technique |
|---|---|
| Size | Each JPEG is split into chunks of up to 1160 bytes, so datagrams stay at 1200 bytes or less and IP never fragments them. Each chunk carries `frame_id`, `index` and `count` |
| Losses | No ACK and no retransmission: a late frame is worthless. A frame with a missing chunk is dropped and the next one replaces it |
| Latency | **Latest frame wins.** Older or duplicate frames are ignored, so your code always gets the newest complete frame, never a backlog |
| Backpressure | If the server's socket can't keep up, `publish()` skips the frame instead of queuing it |
| Authenticity | Every chunk carries a 128-bit HMAC-SHA256 tag. The key is derived from the udp_link key with a separate label for video |
| Address validation | A stateless cookie, the same idea QUIC and DTLS use. The first `SUB` from an address only gets a `CHALLENGE` of the same size; streaming starts when the client sends the cookie back. A spoofed `SUB` can't aim the video at another host |
| Liveness | The client sends `SUB` every second and the server confirms with `SUBACK`. The server stops streaming to a subscriber silent for 4 s while it streams. `BYE` unsubscribes at once. The keep-alive can be turned off: see [Idle mode](#idle-mode-low-power) |
| Replay | Timestamp window (`max_clock_skew` from `[link]`), frame ids that only move forward, session-change checks |

Frames are **authenticated but not encrypted**. Anyone on the LAN can watch them, but nobody without the secret can inject frames. If the video is private, run the link over WireGuard.

Why a separate module from udp_link:

| | udp_link (strings) | video_link (frames) |
|---|---|---|
| Lost packet | resent until ACKed | frame dropped, the next frame replaces it |
| Ordering | every message delivered | only the newest frame matters |
| Size | one datagram per string | about 40 datagrams per 640×480 JPEG |

## Setup

Set up **udp_link first** on both machines ([Setup](../udp_link/README.md#setup)): the project folder, Python 3.11+ and the shared `link.toml`. video_link reads the same file. Then, on each side:

<!-- sides -->
<!-- server -->
**On server_side**

1. Install OpenCV without GUI: `pip install opencv-python-headless`

<details><summary>In depth: which OpenCV package, and why only one</summary>

OpenCV is published on PyPI in four builds that all install the same `cv2` module: `opencv-python` (with windows), `opencv-python-headless` (no GUI), and two `-contrib` variants with extra algorithms. Installed together they overwrite each other's files, and `import cv2` breaks in confusing ways. So install **exactly one** per machine.

The headless build doesn't depend on Qt or GTK, so it's smaller and works on boards with no screen, like the Linux side of an Arduino UNO Q. The server only needs `VideoCapture` (read the camera) and `imencode` (compress to JPEG), which every build includes. OpenCV runs on its own worker thread, so a slow camera never blocks the event loop.

</details>

2. Optional: tune the stream in a `[video]` table of `link.toml`.

```toml
[video]
port = 50556
max_subscribers = 4
```

<details><summary>In depth: what the [video] table controls</summary>

Every key is optional; the [`VideoConfig`](#videoconfig) table lists them all with their defaults. The lookup order is the same as udp_link: the TOML file, then the `UDP_LINK_VIDEO_PORT` environment variable, then `--video-port` on the command line. Unknown keys stop the program with an error, so a typo can't be silently ignored.

`max_subscribers` matters most on a small board: there is no multicast, so each viewer receives its own copy of every frame and costs the full upload bandwidth. `max_write_buffer` sets how much unsent video may pile up in the OS before `publish()` starts skipping frames. `socket_buffer` asks the OS for a bigger send buffer (`SO_SNDBUF`), so short Wi-Fi hiccups don't drop frames.

The secret and the server IP come from the `[link]` table. The video key is derived from the udp_link key with HMAC-SHA256 and the label `video_link/v1`. This **key separation** means a video packet can never be mistaken for a udp_link packet, even though both come from one secret.

</details>

3. Allow inbound UDP **50556**: see [Open the UDP port](#open-the-udp-port-firewall).

<details><summary>In depth: why only the server needs a rule</summary>

As with udp_link, the client always sends first: its `SUB` packets create the flow in its own stateful firewall, so the frames coming back are accepted as replies. The server receives packets it didn't ask for, so it needs an inbound rule.

Packets without a valid HMAC are dropped without any reply. A valid but unvalidated `SUB` gets only a `CHALLENGE` exactly as big as the request, so the port can't be used to amplify an attack.

</details>
<!-- client -->
**On the client side**

1. Install OpenCV with GUI: `pip install opencv-python`

<details><summary>In depth: what the client uses OpenCV for</summary>

Only the `run_client` program uses it, and only with `--display`. It decodes each JPEG with `cv2.imdecode` on a worker thread and shows it with `cv2.imshow`, which needs the GUI build. Without `--display`, and in your own code using `VideoClient`, OpenCV isn't needed: `frame.data` is a complete JPEG file you can save, decode with Pillow or send elsewhere.

</details>

2. Use **the same file** as the server: `link.toml`, copied unchanged.

<details><summary>In depth: what the client reads from it</summary>

From `[link]`: `server_ip` (where to subscribe), `secret` (to derive the same video key) and `client_bind_ip`. From `[video]`: `port` and the client-side settings `subscribe_interval`, `buffer_frames` and `max_partial_frames`. Like udp_link, the client binds to port `0`, so the OS picks a free ephemeral port, and it accepts packets **only** from the server's exact address and port.

</details>

3. Linux only, if many frames are lost: `sudo sysctl -w net.core.rmem_max=8388608`

<details><summary>In depth: the receive buffer</summary>

A 640×480 frame arrives as about 40 datagrams in a burst. If the program is busy for a moment, the kernel queues them in the socket's receive buffer; when it's full, the next datagrams are dropped and the frame is lost. The client asks for a 4 MB buffer (`socket_buffer`, `SO_RCVBUF`), but Linux silently caps the request at `net.core.rmem_max`, often only 208 KB. The `sysctl` command raises the cap until the next reboot; add `net.core.rmem_max=8388608` to `/etc/sysctl.conf` to keep it. macOS and Windows grant the request without this step.

</details>
<!-- /sides -->

## Run

### 1. Quick test: test pattern (no camera)

A moving test pattern checks the network, the secret and the firewall before you plug a camera in. It still needs OpenCV on the server to draw and encode the pattern.

<!-- sides -->
<!-- server -->
**On server_side**

```bash
python3 -m video_link.run_server --config link.toml --source test
```

* The camera (here, the pattern) stays **off until a client subscribes**, and goes off again when the last one leaves
* While streaming, it logs subscribers, fps, Mbit/s and skipped frames every 5 s
<!-- client -->
**On the client side**

```bash
python3 -m video_link.run_client --config link.toml
```

* Runs headless and logs fps, Mbit/s, latency and lost frames every 5 s while video arrives
* Add `--display` to see the pattern in a window
<!-- /sides -->

### 2. Camera

<!-- sides -->
<!-- server -->
**On server_side**

```bash
python3 -m video_link.run_server --config link.toml --source 0 --fps 15 --width 640 --height 480 --quality 70
```

* `--source`: a camera index (`0`), a device such as `/dev/video0`, a video file or a URL
* `--quality`: JPEG quality from 1 to 100
<!-- client -->
**On the client side**

```bash
python3 -m video_link.run_client --config link.toml --display
```

* `--display`: opens a window; press `q` in it to quit
* `--save-dir snaps`: saves a JPEG every `--save-every` seconds (default 5)
<!-- /sides -->

Rough bandwidth: 640×480 at quality 70 is about 30–50 KB per frame, so about 4–6 Mbit/s at 15 fps, per viewer. Lower `--quality`, `--fps` or the resolution on slow Wi-Fi. Latency is only meaningful when both clocks are synced with NTP.

## Idle mode (low power)

The programs already save power when nobody watches: `run_server` closes the camera and stops encoding until a client subscribes, and both programs log statistics only while video flows. What remains is the client's keep-alive: a `SUB` every second, answered by a `SUBACK`, even when no video is sent. You can turn it off in the `[video]` table. Use the **same file on both machines**.

| Profile | `subscribe_interval` | When no video is sent |
|---|---|---|
| Default | `1.0` | SUB/SUBACK every second. A restarted server is found again within a second. |
| Slow keep-alive | `50.0` | One SUB/SUBACK every 50 s. Survives a server restart and keeps the client's firewall open. |
| Idle | `0` | Nothing at all once subscribed: no packet, no timer. |

```toml
[video]
subscribe_interval = 0     # client: subscribe once, then only while video flows
```

`subscriber_ttl` stays at `4.0` in every profile. The server counts a subscriber's silence only **while it streams**: after a pause, every subscriber gets a fresh `subscriber_ttl`. While frames arrive, the client re-subscribes on its own every `subscriber_ttl / 3` seconds, whatever the keep-alive setting. So a live client is never dropped, and a crashed one is dropped 4 s after streaming resumes.

What idle mode gives up:

* **A server restart forgets the client.** The new server has no subscriber, so it never starts the camera, and the idle client never subscribes again. Restart the client, or use the slow keep-alive.
* **The client's firewall may forget the flow** after a long pause, and block the first frames when streaming resumes. The slow keep-alive avoids this; the 50 s value is explained in the udp_link [Idle mode](../udp_link/README.md#idle-mode-low-power) section.
* **Subscribing is confirmed once.** Until the `SUBACK` arrives, the client retries every 2 s, for example while the server is still down.

For the whole project to be idle, also set `heartbeat_interval = 0` and `peer_ttl = 0` in `[link]` if you use udp_link: see its [Idle mode](../udp_link/README.md#idle-mode-low-power).

## Open the UDP port (firewall)

Only **server_side** needs a rule: allow inbound UDP **50556**, exactly like port 50555 for udp_link. The udp_link page has the commands for macOS, Linux (ufw and firewalld) and Windows, with the steps to revert them: [Open the UDP ports](../udp_link/README.md#open-the-udp-ports-firewall). Replace `50555` with `50556`, or open both ports if you run both modules.

| OS | Open UDP 50556 | Revert |
|---|---|---|
| Linux, ufw | `sudo ufw allow 50556/udp comment 'video_link'` | `sudo ufw delete allow 50556/udp` |
| Linux, firewalld | `sudo firewall-cmd --permanent --add-port=50556/udp` then `sudo firewall-cmd --reload` | same with `--remove-port=50556/udp` |
| Windows (admin PowerShell) | `New-NetFirewallRule -DisplayName "video_link 50556" -Direction Inbound -Protocol UDP -LocalPort 50556 -RemoteAddress LocalSubnet -Profile Private -Action Allow` | `Remove-NetFirewallRule -DisplayName "video_link 50556"` |
| macOS | nothing more: the firewall filters per application, so the Python rule covers both ports | see udp_link |

## Library API

Both sides load the two configs from the same file: `link = LinkConfig.load("link.toml")` and `video = VideoConfig.load("link.toml")`. Every class and method is listed in the [API reference](#api-reference) below.

<!-- sides -->
<!-- server -->
**server_side: `VideoServer`**

```python
from contextlib import aclosing
from udp_link import LinkConfig
from video_link import VideoConfig, VideoServer
from video_link.camera import capture

link = LinkConfig.load("link.toml")
video = VideoConfig.load("link.toml")
async with VideoServer(link, video) as server:
    # yields (jpeg_bytes, capture_time)
    frames = capture("0", fps=15)
    async with aclosing(frames):
        async for jpeg, ts in frames:
            await server.publish(jpeg, ts)
```
<!-- client -->
**Client side: `VideoClient`**

```python
from udp_link import LinkConfig
from video_link import VideoClient, VideoConfig

link = LinkConfig.load("link.toml")
video = VideoConfig.load("link.toml")
async with VideoClient(link, video) as client:
    # always the newest complete frame
    async for frame in client.stream():
        process(frame.data)  # JPEG bytes
    client.frames  # the 30 most recent frames
    client.latest  # newest Frame or None
```
<!-- /sides -->

## API reference

Everything below is importable from `video_link` (and `capture` from `video_link.camera`). Methods marked `async` must be awaited. Members marked `@property` are read without parentheses. The network settings (`server_ip`, `bind_ip`, `secret`, `max_clock_skew`) come from the `udp_link` [`LinkConfig`](../udp_link/README.md#linkconfig); `VideoConfig` only adds the video-specific ones.

### `VideoConfig`

```python
from video_link import VideoConfig

video = VideoConfig.load("link.toml")            # the optional [video] table
video = VideoConfig(max_subscribers=2)          # or build it directly
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `@classmethod load(path=None, **overrides)` | `VideoConfig` | Reads the `[video]` table of the TOML file (all keys optional), then the `UDP_LINK_VIDEO_PORT` env var, then `overrides`. Unknown keys raise `ValueError`. |

<!-- api -->
| Field | Default | Meaning |
|---|---|---|
| `port: int` | `50556` | UDP port of the video server (separate from the string link). |
| `subscribe_interval: float` | `1.0` | Seconds between the client's SUB keep-alives. `0` = subscribe once, then only while video flows ([idle mode](#idle-mode-low-power)). |
| `subscriber_ttl: float` | `4.0` | The server stops streaming to a client that stayed silent this long **while it was streaming**; a pause gives everyone a fresh TTL. |
| `max_subscribers: int` | `4` | Maximum simultaneous viewers; each one costs the full upload bandwidth. |
| `max_frame_bytes: int` | `1000000` | Larger JPEGs are refused by `publish()` and by the client. |
| `max_partial_frames: int` | `4` | Frames the client reassembles at the same time; older ones are abandoned. |
| `buffer_frames: int` | `30` | How many recent frames `VideoClient.frames` keeps. |
| `socket_buffer: int` | `4194304` | Requested OS send/receive buffer (Linux caps it at `net.core.*mem_max`). |
| `max_write_buffer: int` | `2097152` | If the server's unsent backlog is above this, `publish()` skips the frame. |

### `VideoServer`

The server_side publisher: sends each frame to every subscribed client, without retransmission.

```python
async with VideoServer(link, video) as server:  # start() ... close()
    sent_to = await server.publish(jpeg_bytes)
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `VideoServer(link: LinkConfig, video: VideoConfig or None = None)` | `VideoServer` | Creates the server; `video=None` uses the defaults. A random session id is chosen so clients can tell a restart from a replay. |
| `async start()` | `Addr` | Opens the UDP socket on `bind_ip:port` and starts answering subscriptions. |
| `async close()` | `None` | Closes the socket. |
| `async with server:` | `VideoServer` | `start()` on entry, `close()` on exit. |
| `async publish(jpeg: bytes, captured_at: float or None = None)` | `int` | Splits the JPEG into signed chunks and sends them to every subscriber. Returns how many received it: `0` when nobody is subscribed or the frame was skipped because the network is behind. Never waits for the network. Raises `ValueError` above `max_frame_bytes`. `captured_at` defaults to now. |
| `async wait_for_subscriber()` | `None` | Sleeps until at least one client is subscribed (returns at once if one is). No polling: `run_server` uses it to keep the camera off while nobody watches. |
| `@property address` | `Addr` | The local `(ip, port)` of the socket. |
| `subscribers` | `dict[Addr, float]` | Clients currently subscribed, with the time of their last SUB. |
| `stats` | `VideoServerStats` | Live counters, see below. |

### `VideoClient`

The client viewer: subscribes to the server and reassembles frames. It always gives you the newest complete frame and drops older ones, so latency never builds up.

```python
async with VideoClient(link, video, on_frame=show) as client:  # connect() ... close()
    frame = await client.next_frame(timeout=5)
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `VideoClient(link: LinkConfig, video=None, *, on_frame=None)` | `VideoClient` | Creates the client. `on_frame(frame)` is called (plain function) for every complete frame. |
| `async connect()` | `None` | Opens a socket and starts the subscribe loop: SUB every `subscribe_interval`, answering the server's anti-spoofing cookie. With `subscribe_interval = 0` the loop ends once the server confirms. Doesn't wait for the server. |
| `async close()` | `None` | Stops subscribing, sends BYE, closes the socket. |
| `async with client:` | `VideoClient` | `connect()` on entry, `close()` on exit. |
| `async next_frame(timeout: float or None = None)` | `Frame` | Waits for a frame newer than the last one it returned, skipping any backlog. Raises `TimeoutError` after `timeout` seconds. |
| `async stream()` | `AsyncIterator[Frame]` | `async for frame in client.stream():` calls `next_frame()` forever. |
| `async wait_for_frame(timeout: float or None = None)` | `None` | Waits until a new frame arrives **without consuming it**: the next `next_frame()` still returns it. Useful beside a viewer, e.g. to log stats only while video flows. |
| `@property subscribed` | `bool` | `True` once the server confirmed the subscription (`SUBACK`) or video arrived. |
| `@property latest` | `Frame or None` | The newest complete frame, without waiting. `None` before the first one. |
| `frames` | `deque[Frame]` | The `buffer_frames` most recent frames, oldest first. |
| `stats` | `VideoClientStats` | Live counters, see below. |

### Data types

`Frame` is one complete received image:

<!-- api -->
| Field | Type | Meaning |
|---|---|---|
| `frame_id` | `int` | Increasing number given by the server; gaps mean frames were lost or skipped. |
| `data` | `bytes` | The JPEG file. Decode it with `cv2.imdecode`, or save it as `.jpg`. |
| `captured_at` | `float` | Server wall clock at capture (seconds since 1970). |
| `received_at` | `float` | Local wall clock when the last chunk arrived. |
| `@property latency` | `float` | `received_at - captured_at` in seconds. Only meaningful if both clocks are NTP-synced. |

<!-- api -->
| Counter | Meaning |
|---|---|
| `VideoServerStats.frames_sent` | Frames sent to at least one subscriber. |
| `VideoServerStats.frames_skipped` | Frames dropped because the send backlog was above `max_write_buffer`. |
| `VideoServerStats.bytes_sent` | Bytes sent, all subscribers together. |
| `VideoServerStats.rejected` | Bad or unauthenticated packets received. |
| `VideoClientStats.frames` | Complete frames received. |
| `VideoClientStats.bytes` | JPEG bytes received in complete frames. |
| `VideoClientStats.incomplete` | Frames abandoned because a chunk never arrived. |
| `VideoClientStats.late` | Chunks of frames older than one already shown, ignored. |
| `VideoClientStats.rejected` | Packets dropped: wrong sender, bad HMAC, stale or malformed. |

### Camera and errors

<!-- api -->
| Function / exception | Returns | What it does |
|---|---|---|
| `async capture(source="0", *, width=640, height=480, fps=15.0, quality=70)` | `AsyncIterator[tuple[bytes, float]]` | From `video_link.camera`. Yields `(jpeg_bytes, capture_time)` at most `fps` times per second. `source` is a camera index (`"0"`), a device path (`"/dev/video0"`), a file or URL, or `"test"` for a generated pattern. OpenCV runs on its own thread so the event loop never blocks. Needs `opencv-python`. |
| `VideoProtocolError` | exception | A video datagram is malformed, unauthenticated or stale. Handled internally and counted in `rejected`. |
| `DEFAULT_VIDEO_PORT` | `int` | `50556`. |

## Tests

```bash
python3 -m unittest tests.test_video -v
```

The tests cover: intact frames, idle mode (subscribe once, no task left running, pauses longer than the TTL, re-subscribing from frames, dropping a crashed client), a lost chunk dropping only its own frame, replayed frames being ignored, the cookie challenge (with no amplification), a wrong secret, BYE, and the frame size limit.
