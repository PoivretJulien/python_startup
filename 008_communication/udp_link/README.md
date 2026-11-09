# udp_link

This package exchanges plain `str` data over UDP between two machines on the same LAN. It uses only the standard library, needs Python 3.11 or newer, and runs on macOS, Linux and Windows. Typical setup:

* **server_side**: runs the **server** and pushes strings to its clients, for example the OS time or sensor readings such as `temp=21.4;hum=45.0`.
* **Client side**: runs the **client**, stacks every received string in a list and processes it.

Data flows both ways. Each side keeps the strings it receives in a list (`.received`).

## What runs where

| | server_side | Client side |
|---|---|---|
| Time test (any OS) | `python3 -m apps.time_server` | `python3 -m apps.time_client` |
| Sensor strings | `python3 -m apps.sensor_server` | `python3 -m apps.pc_client` |
| Confirmed requests | `python3 -m apps.confirm_server` | `python3 -m apps.confirm_client` |
| Video (optional) | `python3 -m video_link.run_server` | `python3 -m video_link.run_client` |
| OpenCV (video only) | `pip install opencv-python-headless` | `pip install opencv-python` |
| Firewall | allow inbound UDP 50555 (and 50556 for video), see [Open the UDP ports](#open-the-udp-ports-firewall) | nothing to open |
| `link.toml` | created here | copy of the server's file |
| Your code goes in | `read_sensor()` in `apps/sensor_server.py` | `process_batch()` in `apps/pc_client.py` |

On Windows, type `py` instead of `python3` in every command.

## Project layout

```
udp_link_project/
├── udp_link/            # the library (copy it to both machines)
│   ├── protocol.py      # wire format, HMAC, replay checks
│   ├── endpoint.py      # reliability: seq/ACK/retransmit/dedup, received stack
│   ├── server.py        # UDPServer  (bind, peers, send, broadcast)
│   ├── client.py        # UDPClient  (HELLO handshake, heartbeat, send)
│   └── config.py        # LinkConfig (TOML / env / CLI)
├── apps/
│   ├── time_server.py   # test: sends the OS time every 3 s (any OS)
│   ├── time_client.py   # test: prints the time messages (any OS)
│   ├── sensor_server.py # sensor example, server_side
│   ├── pc_client.py     # sensor example, client side
│   ├── confirm_server.py # confirmed requests, server_side
│   └── confirm_client.py # confirmed requests, client side
├── video_link/          # separate module: camera video streaming (see video_link/README.md)
├── tests/               # test_link.py, test_video.py, test_confirm.py
└── link.example.toml
```

## Protocol and security

| Concern | Technique |
|---|---|
| Delivery | Sequence numbers, ACKs, retransmission with exponential backoff and jitter. `send()` raises `DeliveryError` if no ACK arrives after the last retry |
| Duplicates | Per-peer, per-session dedup window. A duplicate is ACKed again but never delivered twice |
| Liveness | The client opens with a `HELLO` handshake, then sends a `PING` heartbeat every 5 s. The server forgets clients that stay silent past the TTL. The client reconnects on its own. Both timers can be turned off: see [Idle mode](#idle-mode-low-power) |
| Authenticity | HMAC-SHA256 over every datagram, using a key derived (PBKDF2) from a shared secret. Unauthenticated packets are dropped without a reply and never create state |
| Replay | Timestamp window (`max_clock_skew`) plus the dedup window |
| Resource limits | Datagrams of 1200 bytes or less (no fragmentation), capped peer count, capped stack size, capped number of in-flight sends |
| Concurrency | Pure `asyncio`; many sends can run at once |

Payloads are **authenticated but not encrypted**. Anyone on the LAN can read them, but nobody without the secret can forge them. If the data is confidential, run the link over WireGuard.

Because the payload must fit in one datagram, the practical limit is about 1100 bytes per string. Longer strings raise `PayloadTooLargeError`.

## Setup

On **both machines**:

1. Copy the whole `udp_link_project/` folder, for example: `scp -r udp_link_project user@<server_side-ip>:~` (`scp` is built into macOS, Linux and Windows 10+).
2. Check Python with `python3 --version` (Windows: `py --version`). It must be 3.11 or newer.
3. Optional: run `pip install -e .` in the project folder. The programs then run from any folder, and `--config link.toml` is also looked up in the project folder.

Then, on each side:

<!-- sides -->
<!-- server -->
**On server_side**

1. Find its IP: `python3 -m udp_link ip`

<details><summary>In depth: how the IP is found, and where it goes</summary>

The command opens a UDP socket and calls `connect()` toward `10.255.255.255`. For UDP, `connect()` **sends nothing**: it only asks the operating system which route, and so which **source address**, it would use to reach that destination. `getsockname()` then returns that address: the IPv4 address of the interface on the default route (Wi-Fi or Ethernet). With no network at all it prints `127.0.0.1`.

LAN addresses come from the private ranges of **RFC 1918** (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`). They are usually handed out by the router via **DHCP**, so they can change after a reboot. Reserve a fixed address for the server in the router's DHCP settings, or `link.toml` has to be updated on both machines.

Where it goes: into `server_ip` in `link.toml`. Only the **client** uses it, as the destination of every packet. The server doesn't bind to it; it listens on `bind_ip` (default `0.0.0.0` = all IPv4 interfaces). The client resolves `server_ip` with `getaddrinfo`, so a hostname such as `myserver.local` (mDNS) also works.

With several interfaces (Ethernet + Wi-Fi, a VPN), the command picks the default-route one, which may not be the LAN you want. Check by hand: macOS `ipconfig getifaddr en0`, Linux `hostname -I` or `ip -4 addr`, Windows `ipconfig`.

</details>

2. Generate a secret: `python3 -m udp_link keygen`

<details><summary>In depth: what the secret is, the standards behind it, where it lives</summary>

`keygen` calls `secrets.token_urlsafe(32)`: **32 random bytes (256 bits)** from the operating system's cryptographic random generator (`getrandom` on Linux, `arc4random` on macOS, `BCryptGenRandom` on Windows), written as 43 URL-safe Base64 characters.

At startup, both sides turn the secret into a 32-byte key with **PBKDF2-HMAC-SHA256** (RFC 8018; fixed salt `udp-link/v1`, 100 000 iterations). The fixed salt is what lets both machines derive the **same key** without any handshake; the iterations slow down guessing if someone uses a weak passphrase instead of a generated secret.

Every datagram then carries a **HMAC-SHA256** tag (RFC 2104, FIPS 198-1, with SHA-256 from FIPS 180-4) computed over its content. The receiver recomputes it and compares in constant time (`hmac.compare_digest`). A wrong tag means a wrong secret or a modified packet: the packet is dropped silently, with no reply. video_link derives its own separate key from this one (label `video_link/v1`) and uses 128-bit tags.

This is **authentication and integrity, not encryption**: anyone on the LAN can read the strings, but no one can forge or alter them without the secret. The secret itself **never travels over the network**.

Where it lives: only in `link.toml` (or the `UDP_LINK_SECRET` environment variable) on each machine. In memory, the program keeps the derived key, and both are hidden from logs and error messages (`repr=False`). It is deliberately not a command-line option, because other users can see command lines with `ps`. To change it: generate a new one, update `link.toml` on both machines, restart both programs.

</details>

3. Create the config: `cp link.example.toml link.toml` (Windows: `copy link.example.toml link.toml`), then fill in `server_ip` (this machine's IP from step 1), `port` and `secret`.

<details><summary>In depth: the file format, how it's found, and what each field does</summary>

`link.toml` uses **TOML 1.0**, read with Python's built-in `tomllib`. Settings live in a `[link]` table; an optional `[video]` table holds video_link settings.

Each program looks for the file in this order: the `--config` path, then the `UDP_LINK_CONFIG` environment variable, then `link.toml` in the current folder, then in the project folder. Values are then overridden by environment variables (`UDP_LINK_SERVER_IP`, `UDP_LINK_PORT`, `UDP_LINK_SECRET`, `UDP_LINK_BIND_IP`), then by command-line flags, so the precedence is **flags > environment > file > defaults**.

* `server_ip`: where the client sends (see step 1).
* `bind_ip`: which interface the server listens on. `0.0.0.0` = all; a specific IP = only that interface.
* `port`: `50555` by default, inside the dynamic/private range 49152–65535 (RFC 6335), so it won't collide with registered services. Both sides must use the same value.
* `secret`: from step 2, at least 16 characters.

The file is validated at every start: unknown keys are refused (a typo fails loudly instead of being ignored), the port must be 0–65535, the secret long enough. Also note `max_clock_skew` (default 120 s): packets whose timestamp is further than that from the receiver's clock are rejected as replays, so keep both clocks in sync (NTP), or set it to `0` on a network without time sync.

</details>

4. Protect it: `chmod 600 link.toml` (Windows: `icacls link.toml /inheritance:r /grant:r "${env:USERNAME}:F"`)

<details><summary>In depth: what these permissions mean</summary>

`chmod 600` sets the POSIX permission bits to `rw-------`: the owner can read and write, the group and everyone else can do nothing. Without it, any other account on the machine could read the secret. If the file holds a secret and is readable by others, the program prints a warning at startup.

On Windows, `icacls` edits the file's access control list (ACL): `/inheritance:r` removes the permissions inherited from the folder, and `/grant:r "${env:USERNAME}:F"` replaces them with Full control for your account only.

Administrators (root on macOS/Linux) can still read the file: permissions protect against other users, not against the machine's owner. Using the `UDP_LINK_SECRET` environment variable instead keeps the secret out of files, but it is then visible to your own processes and shell history, so the protected file is usually the better choice.

</details>

5. Allow inbound UDP on its port: see [Open the UDP ports](#open-the-udp-ports-firewall).

<details><summary>In depth: why only the server needs a rule</summary>

UDP has no connections, but modern firewalls are **stateful**: when a machine sends a UDP packet, the firewall remembers the flow (address and port pairs) and lets the replies back in for a while. On Linux this is `conntrack`, which keeps an idle UDP flow for about 30 seconds to a few minutes.

The client always speaks first (`HELLO`, then `PING` every 5 s), so its firewall sees outgoing flows and accepts the server's ACKs and pushed strings as replies. The server receives packets it didn't ask for, so its firewall needs an explicit inbound rule for UDP `50555` (and `50556` for video). The 5 s heartbeat also keeps the client's flow fresh, which is why a much slower heartbeat could let the client's firewall forget it. [Idle mode](#idle-mode-low-power) explains how to choose a slow heartbeat that stays under these timeouts.

Packets without a valid HMAC are dropped without any reply, so a port scanner sees the port as "open|filtered" and learns nothing.

</details>
<!-- client -->
**On the client side**

1. Copy the server's `link.toml` into the project folder. It must be **the same file** on both machines: `scp user@<server_side-ip>:~/udp_link_project/link.toml .`

<details><summary>In depth: why the same file, and why scp</summary>

The two sides share no handshake and negotiate nothing: they agree only because they read the same values. `server_ip` and `port` tell the client where to send; the `secret` must match **exactly**, character for character, so both derive the same PBKDF2 key and the HMAC tags verify. `bind_ip` is ignored on the client.

`scp` copies over **SSH**, which encrypts the transfer, so the secret never crosses the network in the clear. Email, chat or a shared folder are weaker places to put it.

If the secrets differ, the server silently drops every packet, so the client keeps retrying and logs `server ... not answering (down, wrong IP/port or wrong secret)`. That message is the usual sign of a copy mistake.

</details>

2. Protect it, as on the server.

<details><summary>In depth: why the client needs this too</summary>

The client's copy holds the same secret, and with it anyone could send forged strings that the server would accept as genuine. Same command, same reason: `chmod 600 link.toml` on macOS/Linux, the `icacls` line on Windows.

</details>

3. Nothing else: the client uses a random local port and the server only answers it, so no firewall rule is needed.

<details><summary>In depth: the random port, and how the server finds the client</summary>

The client binds to port `0`, which tells the operating system to pick a free **ephemeral port**: 49152–65535 on macOS and Windows, 32768–60999 by default on Linux. It changes at every start.

The server never needs to know it in advance. It reads the source address and port of the client's `HELLO`, stores it in its peer table, and sends every ACK and pushed string back to exactly that address. The client, in turn, accepts packets **only** from the server's exact address and port; anything else is ignored before it is even decoded.

Because the client always sends first and keeps sending (heartbeat), its firewall treats the server's packets as replies, as explained in server step 5.

</details>
<!-- /sides -->

To try everything on **one computer** first, set `server_ip = "127.0.0.1"` and run both sides in two terminals. No firewall rule is needed for that.

## Run

### 1. Quick test: OS time every 3 s (macOS, Linux, Windows)

The simplest check that the link works. Every 3 seconds the server sends its local OS time and the time elapsed since it started, for example `time=2026-11-09T16:30:03+01:00;elapsed=63.0;tick=21`. The elapsed time uses a monotonic clock, so changing the system clock doesn't affect it.

<!-- sides -->
<!-- server -->
**On server_side**

macOS / Linux:

```bash
python3 -m apps.time_server --config link.toml
```

Windows (PowerShell or cmd):

```powershell
py -m apps.time_server --config link.toml
```

Logs each message it sends and how many clients ACKed it. `--interval 1` changes the period.
<!-- client -->
**On the client side**

macOS / Linux:

```bash
python3 -m apps.time_client --config link.toml
```

Windows (PowerShell or cmd):

```powershell
py -m apps.time_client --config link.toml
```

Prints each received message, for example `server time 2026-11-09T16:30:03+01:00 | elapsed 63.0 s | 21 message(s) received`.
<!-- /sides -->

Stop either side with `Ctrl+C`. You can start either side first; the client waits for the server.

### 2. Sensor example

<!-- sides -->
<!-- server -->
**On server_side**

```bash
python3 -m apps.sensor_server --config link.toml --interval 1 --interactive
```

Sends a sensor reading to every connected client each `--interval` seconds, and stacks the strings clients send back.

* Typed line + Enter: broadcast it to the clients
* `/stack`: show the received strings
* `/peers`: list the connected clients
* Your sensors: replace `read_sensor()` in `apps/sensor_server.py`
<!-- client -->
**On the client side**

```bash
python3 -m apps.pc_client --config link.toml --batch-interval 5 --interactive
```

Connects to the server, stacks every received reading and processes the stack in batches: when a reading arrives, it waits `--batch-interval` seconds for more, then processes them all. With no readings, it sleeps.

* Typed line + Enter: send it to the server (ACKed)
* `/stack`: show the readings waiting to be processed
* Your processing: edit `process_batch()` in `apps/pc_client.py`
<!-- /sides -->

`--interactive` reads typed lines without blocking, which only works on macOS and Linux. On Windows, run the sensor example without `--interactive`.

### Sensor context: server_side on an Arduino UNO Q

If server_side is an Arduino UNO Q, `sensor_server.py` runs on the board's Linux side. The sensors are usually wired to the MCU, and the Linux side reaches the sketch through the Arduino Bridge (RPC), for example `Bridge.call("read_temp")` inside App Lab. Make `read_sensor()` call the Bridge and format what it returns as a `key=value;key=value` string.

### 3. Confirmed requests: received, then processed

When `await client.send(...)` returns, the server's **transport ACK** has arrived: the message is in the server's stack. That doesn't prove your server code has handled it. This example adds an **application reply**: the server does the work, then answers `done=<id>` with the outcome. The client therefore sees both confirmations for every request.

<!-- sides -->
<!-- server -->
**On server_side**

```bash
python3 -m apps.confirm_server --config link.toml
```

* Carries out commands on a pretend device: `led on|off`, `add <number>`, `status` (0.2 s each)
* Replies `done=<id>;status=ok;result=...` or `done=<id>;status=error;reason=...`
* Your code goes in: `Device.execute()` in `apps/confirm_server.py`
<!-- client -->
**On the client side**

```bash
python3 -m apps.confirm_client --config link.toml
```

* Without arguments, runs a demo of 5 requests, two of them invalid
* Or send your own: `... confirm_client --config link.toml "led off" "add 5" status`
* `--timeout`: seconds to wait for each reply (default 5)
<!-- /sides -->

The client prints both steps for every request:

```
request 1  led on
  sent        transport ACK: the server received it      (0 ms)
  done        status=ok, led is on: the server processed it  (204 ms)
request 3  led blue
  sent        transport ACK: the server received it
  FAILED      the server processed it and refused: led value must be on or off, got 'blue'
```

| What the client sees | Meaning |
|---|---|
| `done` | received **and** processed successfully |
| `FAILED` | received and processed, but the server refused it (bad command or value) |
| `NO REPLY` | received (ACKed), but no answer within `--timeout`: the work may or may not have happened |
| `NOT RECEIVED` | `DeliveryError`: no ACK after all retries; the server never had the message |

<details><summary>In depth: how replies are matched to requests</summary>

Every request carries a number, `id=7;cmd=led;value=on`, taken from a counter in the client. Before sending, the client stores an `asyncio.Future` under that id: a placeholder that a later event fills in. The server echoes the id in its reply (`done=7;...`). The client's `on_message` handler looks the id up, fills the Future, and the waiting `request()` call wakes up. Because of the ids, several requests can be in flight at the same time and their replies can arrive in any order.

The server handler is an `async def`, which udp_link runs as its own task for each message. A slow command therefore never delays the reception of the next one. Every outcome gets a reply, including errors and unexpected exceptions, so the client never waits for an answer that won't come.

Both sides parse with `parse_fields()` from `apps/_common.py`, which keeps every value as text. `parse_kv()`, used by the sensor example, would turn `5` into `5.0` and break ids and exact values.

`NO REPLY` is the one uncertain case: the reply may be lost, the server may be slow, or it may have crashed mid-work. If running a command twice is harmful, make commands **idempotent** ("set led on" rather than "toggle led"), or have the server remember recent ids and answer a repeated id with the stored result instead of doing the work again.

</details>

## Idle mode (low power)

By default the link checks itself even when no data is exchanged. The client sends a `PING` every 5 s, the server answers with an ACK, and a server timer forgets silent clients. That costs very little (about 30 wake-ups and a few milliseconds of CPU per 20 s), but it keeps both machines and their Wi-Fi radios waking up. On a battery device you can turn this off in `link.toml`. Use the **same values on both machines**.

| Profile | `heartbeat_interval` | `peer_ttl` | When nothing is sent |
|---|---|---|---|
| Default | `5.0` | `20.0` | PING/ACK every 5 s. Lost links are noticed within seconds. |
| Slow heartbeat (listen-only clients) | `50.0` | `180.0` | One PING/ACK every 50 s. The client still hears from the server after long pauses. |
| Idle | `0` | `0` | Nothing at all: no packet, no timer. Each side wakes only when a packet arrives. |

```toml
[link]
# ... server_ip, port, secret as usual ...
heartbeat_interval = 0     # client: no PING
peer_ttl           = 0     # server: never forget clients on a timer
```

`0` turns that timer off. When both are on, `peer_ttl` must be longer than `heartbeat_interval`; 3 to 4 times longer leaves room for a lost PING.

What idle mode gives up, and how the link copes:

* **The client notices a lost server only when it sends.** `send()` raises `DeliveryError` and sets `client.connected` to `False`. The next successful send sets it back to `True`.
* **A server restart forgets every client.** A client gets back into the list the next time it sends something, because every DATA packet re-registers it. A client that only listens stays unknown until then. Use the slow heartbeat for those clients.
* **Crashed clients stay in the peer list.** Clients that exit normally send BYE and are removed at once. A client that crashed, or lost power, is removed on the first `broadcast()` it doesn't ACK.
* **The client's firewall may forget the flow.** After a long pause, the server's next push can be blocked by the client's firewall (see below). Use the slow heartbeat for those clients, or allow inbound UDP on the client too.

<details><summary>In depth: why 50 s for the slow heartbeat, and what still wakes up</summary>

Stateful firewalls keep a UDP flow open for a limited time after its last packet. On Linux, `conntrack` keeps an answered flow for 120 s (`nf_conntrack_udp_timeout_stream`) and an unanswered one for 30 s. Windows Defender Firewall drops idle UDP state after about 60 s, and home routers often use 30 to 180 s. A PING every 50 s keeps the flow open under all of these, so the server's pushes keep arriving as replies. A `peer_ttl` of 180 s lets the server miss two PINGs in a row before it forgets the client.

Any packet refreshes the flow, in either direction. So a client that receives data regularly never needs the heartbeat to keep its firewall open; it matters only after long pauses.

In idle mode, the library itself starts no task and no timer. Retransmission timers exist only while a send is waiting for its ACK. The operating system still wakes the process when a packet arrives, which is exactly the point.

Your own program decides the rest. `pc_client` sleeps until the first message of a batch arrives, so it is idle too. `time_server` and `sensor_server` send on a fixed rhythm by design; on a battery device, send only when a value changes, or at a slow interval.

</details>

## Open the UDP ports (firewall)

Only **server_side** needs a rule: it must accept inbound UDP on port **50555** (udp_link), plus **50556** if you use video_link. The client side needs nothing, because the server only replies to the client's own traffic.

Below, every "open" step has a matching "revert" step that puts the firewall back exactly as it was. The "back to default" steps wipe **all** custom rules, not just these, so use them only if you know nothing else depends on the firewall.

### macOS

The macOS firewall is **off by default**, and then there is nothing to do. It filters per **application**, not per port, so one rule for Python covers both ports.

Check whether it's on:

```bash
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
```

If it's on, the easiest way is to click **Allow** when macOS asks "Do you want the application Python to accept incoming network connections?" the first time the server starts. To do it from the terminal instead:

```bash
PYAPP="$(python3 -c 'import sys,os; p=os.path.join(sys.base_prefix,"Resources/Python.app"); print(p if os.path.exists(p) else os.path.realpath(sys.executable))')"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add "$PYAPP"
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp "$PYAPP"
```

Check: `/usr/libexec/ApplicationFirewall/socketfilterfw --listapps`. If "Block all incoming connections" is enabled (`--getblockall`), it overrides per-app rules.

**Revert:** remove Python from the list (set `PYAPP` as above first):

```bash
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --remove "$PYAPP"
```

**Back to default:** the default is firewall off. Only if you switched it on for this test, switch it off again with `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off`, or in System Settings → Network → Firewall.

### Linux

Most distributions use one of two tools: **ufw** (Debian, Ubuntu, Raspberry Pi OS) or **firewalld** (Fedora, RHEL, Rocky, openSUSE). To see which is active, run `sudo ufw status` or `sudo firewall-cmd --state`. If neither is installed or active, nothing filters the port and there is nothing to do.

For video, repeat each command with `50556`.

<!-- sides -->
<!-- col ufw -->
**ufw**

Open the port:

```bash
sudo ufw allow 50555/udp comment 'udp_link'
```

Or LAN only (safer):

```bash
sudo ufw allow from 192.168.1.0/24 to any port 50555 proto udp comment 'udp_link'
```

Check:

```bash
sudo ufw status numbered
```

**Revert** (same rule, with `delete`):

```bash
sudo ufw delete allow 50555/udp
```

```bash
sudo ufw delete allow from 192.168.1.0/24 to any port 50555 proto udp
```

**Back to default:** `sudo ufw reset` deletes all rules and disables ufw, which is the default state on Ubuntu. If you enable it again, re-allow SSH first (`sudo ufw allow OpenSSH`) or you may lock yourself out of a remote machine.
<!-- col firewalld -->
**firewalld**

Open the port (permanent, then reload):

```bash
sudo firewall-cmd --permanent --add-port=50555/udp
sudo firewall-cmd --reload
```

Or LAN only (safer):

```bash
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="50555" protocol="udp" accept'
sudo firewall-cmd --reload
```

Check:

```bash
sudo firewall-cmd --list-ports
sudo firewall-cmd --list-rich-rules
```

**Revert** (same rule, with `remove`):

```bash
sudo firewall-cmd --permanent --remove-port=50555/udp
sudo firewall-cmd --reload
```

```bash
sudo firewall-cmd --permanent --remove-rich-rule='rule family="ipv4" source address="192.168.1.0/24" port port="50555" protocol="udp" accept'
sudo firewall-cmd --reload
```

**Back to default:** reload the zone's shipped settings, which removes all custom rules in that zone:

```bash
sudo firewall-cmd --permanent --load-zone-defaults=$(firewall-cmd --get-default-zone)
sudo firewall-cmd --reload
```
<!-- /sides -->

Replace `192.168.1.0/24` with your own LAN range (the first three numbers of your IP, then `.0/24`).

### Windows

Windows Defender Firewall blocks inbound UDP by default. Run these in **PowerShell as Administrator**.

The first time the server starts, Windows may ask whether Python can communicate on private networks. Clicking **Allow** creates rules for `python.exe` (all ports). The commands below instead open only the needed port, which is tighter.

Open the port for the LAN only, on private networks:

```powershell
New-NetFirewallRule -DisplayName "udp_link 50555" -Direction Inbound -Protocol UDP -LocalPort 50555 -RemoteAddress LocalSubnet -Profile Private -Action Allow
```

For video, add the same rule with `50556` and the name `"udp_link 50556"`.

Check the rules, and that your network is **Private** (rules for `-Profile Private` don't apply on a Public network):

```powershell
Get-NetFirewallRule -DisplayName "udp_link*" | Format-Table DisplayName, Enabled, Profile, Action
Get-NetConnectionProfile
```

**Revert** (removes only these rules):

```powershell
Remove-NetFirewallRule -DisplayName "udp_link*"
```

If you clicked **Allow** on the prompt, list and remove the Python rules it created:

```powershell
Get-NetFirewallApplicationFilter | Where-Object Program -like "*python*" | Get-NetFirewallRule | Format-Table DisplayName, Direction, Action, Profile
Get-NetFirewallApplicationFilter | Where-Object Program -like "*python*" | Get-NetFirewallRule | Remove-NetFirewallRule
```

**Back to default:** `netsh advfirewall reset` restores the factory firewall policy and deletes **all** custom rules, including ones other programs created. Prefer the targeted revert above.

### Revert at a glance

| OS | Remove only these rules | Back to default (removes all custom rules) |
|---|---|---|
| macOS | `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --remove "$PYAPP"` | firewall off (only if it was off before) |
| Linux, ufw | `sudo ufw delete allow 50555/udp` | `sudo ufw reset` |
| Linux, firewalld | `sudo firewall-cmd --permanent --remove-port=50555/udp`, then `--reload` | `--load-zone-defaults=<zone>`, then `--reload` |
| Windows | `Remove-NetFirewallRule -DisplayName "udp_link*"` | `netsh advfirewall reset` |

## Library API

Both sides load the same config: `cfg = LinkConfig.load("link.toml")` (or env vars / keyword overrides).

Every class and method is listed, with its arguments and return value, in the [API reference](#api-reference) below.

<!-- sides -->
<!-- server -->
**server_side: `UDPServer`**

```python
from udp_link import LinkConfig, UDPServer

cfg = LinkConfig.load("link.toml")
server = UDPServer(cfg, on_message=print)
async with server:
    # send to all clients -> {peer: delivered}
    await server.broadcast("temp=21.4")
    await server.send("hi", server.peers[0])
    msg = await server.recv(timeout=5)  # oldest
    server.received         # list[Message]
    batch = server.drain()  # take all, clear
```
<!-- client -->
**Client side: `UDPClient`**

```python
from udp_link import LinkConfig, UDPClient

cfg = LinkConfig.load("link.toml")
async with UDPClient(cfg) as client:
    await client.send("led=on")  # waits for ACK
    # stream them (or batch: client.drain())
    async for msg in client.messages():
        print(msg.text, msg.received_at)
    client.received  # list[Message]
```
<!-- /sides -->

## API reference

Everything below is importable from `udp_link`. Methods marked `async` must be awaited (`await server.send(...)`). Members marked `@property` are read like attributes, without parentheses (`server.peers`). Using a server or client before it is started raises `RuntimeError`.

### `LinkConfig`

The shared settings: addresses, port, secret and protocol tuning. It is frozen (read-only once created) and both sides must agree on `server_ip`, `port` and `secret`.

```python
from udp_link import LinkConfig

cfg = LinkConfig.load("link.toml")              # file, then env vars
cfg = LinkConfig.load("link.toml", port=50600)  # keyword overrides win
cfg = LinkConfig(server_ip="192.168.1.42", secret="at-least-16-characters")
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `@classmethod load(path=None, **overrides)` | `LinkConfig` | Builds a config from, in increasing priority: defaults, the TOML file (`[link]` table or top level), the env vars `UDP_LINK_SERVER_IP` / `UDP_LINK_BIND_IP` / `UDP_LINK_PORT` / `UDP_LINK_SECRET`, then `overrides` (a `None` value is ignored). With `path=None` it looks for `link.toml` in the current folder, then the project root. Unknown keys raise `ValueError`. |
| `is_allowed(host: str)` | `bool` | `True` if `host` may talk to us: always when `allowed_hosts` is empty, otherwise only when it is listed. |
| `key` | `bytes` | The 32-byte HMAC key derived from `secret` with PBKDF2 when the config is created. Not settable, never printed. |

Fields (all keyword arguments, all optional except `secret`):

<!-- api -->
| Field | Default | Meaning |
|---|---|---|
| `server_ip: str` | `"127.0.0.1"` | Address the client sends to. |
| `bind_ip: str` | `"0.0.0.0"` | Interface the server listens on (`0.0.0.0` = all). |
| `client_bind_ip: str` | `"0.0.0.0"` | Interface the client sends from; its port is picked by the OS. |
| `port: int` | `50555` | UDP port of the server. |
| `secret: str` | (required) | Shared secret, at least 16 characters. Hidden from `repr()`. |
| `ack_timeout: float` | `0.3` | Seconds before the first retransmission. |
| `max_retries: int` | `5` | Retransmissions after the first send before `DeliveryError`. |
| `max_backoff: float` | `2.0` | Upper limit of the doubling wait between retransmissions. |
| `max_in_flight: int` | `64` | Sends that may wait for an ACK at the same time; more wait their turn. |
| `heartbeat_interval: float` | `5.0` | Seconds between client PINGs. `0` = no heartbeat ([idle mode](#idle-mode-low-power)). |
| `peer_ttl: float` | `20.0` | The server forgets a client silent for this long. `0` = never on a timer, only on BYE or a failed send. Must be longer than `heartbeat_interval` when both are on. |
| `max_clock_skew: float` | `120.0` | Replay window: older or future packets are rejected. `0` turns the check off. |
| `max_peers: int` | `32` | Maximum number of clients the server tracks. |
| `inbox_limit: int` | `10000` | Maximum stacked messages; the oldest are dropped beyond it. |
| `allowed_hosts: frozenset[str]` | `frozenset()` | Optional IP allow-list; empty = any host that knows the secret. |

### `UDPServer`

The server_side endpoint: listens on `bind_ip:port`, tracks clients and sends to them with ACKs.

```python
server = UDPServer(cfg, on_message=handler)  # handler(msg) may be a plain or async function
async with server:                           # start() ... close()
    ...
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `UDPServer(config, *, on_message=None)` | `UDPServer` | Creates the server without opening anything yet. `on_message(msg)` is called for every new message; an `async def` handler is run as a task. |
| `async start()` | `Addr` | Opens the UDP socket and starts the housekeeping task that forgets silent clients. Returns the local `(ip, port)`. |
| `async close()` | `None` | Stops housekeeping and closes the socket. Waiting `send()` calls fail. |
| `async with server:` | `UDPServer` | Calls `start()` on entry and `close()` on exit, even after an error or Ctrl+C. |
| `@property peers` | `list[Addr]` | Clients currently known: those heard from (HELLO, PING or DATA) within `peer_ttl`, or since they connected if `peer_ttl` is `0`. |
| `async send(text: str, peer: Addr)` | `None` | Reliable send to one client: retransmits until ACKed. Raises `DeliveryError` after `max_retries`, `PayloadTooLargeError` if `text` is too big. |
| `async broadcast(text: str)` | `dict[Addr, bool]` | Sends to every known client at the same time and returns who acknowledged. Never raises `DeliveryError`: a client that didn't ACK is `False` and is dropped from `peers` until its next heartbeat or send. |

The receiving methods (`received`, `recv`, `drain`, `messages`, `stats`) are listed under [Shared by server and client](#shared-by-server-and-client).

### `UDPClient`

The client endpoint: connects to `server_ip:port`, keeps the link alive with heartbeats and says goodbye on close.

```python
async with UDPClient(cfg, on_message=handler) as client:  # connect() ... close()
    await client.send("led=on")
```

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `UDPClient(config, *, on_message=None, wait_for_server=True)` | `UDPClient` | Creates the client. With `wait_for_server=True`, `connect()` keeps retrying until the server answers, so the client may start first. |
| `async connect()` | `None` | Resolves `server_ip`, opens a socket, does the HELLO handshake, then starts the PING heartbeat task (none if `heartbeat_interval` is `0`). |
| `async close()` | `None` | Stops the heartbeat, sends BYE so the server forgets us at once, closes the socket. |
| `async with client:` | `UDPClient` | Calls `connect()` on entry and `close()` on exit. |
| `@property server` | `Addr` | The server's `(ip, port)` after `connect()`. |
| `connected` | `bool` | `True` after the handshake; set to `False` when a PING or `send()` gets no ACK, and back to `True` on the next success. |
| `async send(text: str)` | `None` | Reliable send to the server. Raises `DeliveryError` if it isn't ACKed after all retries, `PayloadTooLargeError` if `text` is too big. Also re-registers the client on the server, which recovers from a server restart. |

### Shared by server and client

Both `UDPServer` and `UDPClient` inherit these. They work once the link is started.

<!-- api -->
| Method | Returns | What it does |
|---|---|---|
| `config` | `LinkConfig` | The config the object was created with. |
| `@property received` | `list[Message]` | The stack of received messages, oldest first. Read or loop over it freely; it keeps growing until you take messages out. |
| `drain()` | `list[Message]` | Takes every stacked message at once and empties the stack. Best for batch processing. |
| `async recv(timeout: float or None = None)` | `Message` | Removes and returns the oldest message, waiting if the stack is empty. Raises `TimeoutError` after `timeout` seconds, `ConnectionError` if the link closes. |
| `async messages()` | `AsyncIterator[Message]` | `async for msg in link.messages():` yields messages one by one (via `recv()`) until the link closes. |
| `@property stats` | `LinkStats` | Live counters for this endpoint. |
| `@property address` | `Addr` | The local `(ip, port)` of the socket. |

### Data types

`Message` is one received string. It is created by the library; you only read it.

<!-- api -->
| Field | Type | Meaning |
|---|---|---|
| `text` | `str` | The string that was sent. |
| `peer` | `Addr` | Sender's `(ip, port)`. On the server, pass it to `send()` to answer. |
| `seq` | `int` | Sender's sequence number, used for ACKs and duplicate detection. |
| `sent_at` | `float` | Sender's wall clock at send time (seconds since 1970). |
| `received_at` | `datetime` | Time of arrival, in UTC (timezone-aware; `.astimezone()` gives local time). |

`LinkStats` counts what happened on one endpoint since it started:

<!-- api -->
| Field | Type | Meaning |
|---|---|---|
| `sent` | `int` | Datagrams sent, of every kind: data, retransmissions, ACKs, PINGs. |
| `delivered` | `int` | New messages handed to your code. |
| `retransmits` | `int` | Extra copies sent because an ACK was late. |
| `acked` | `int` | Sends confirmed by the peer. |
| `duplicates` | `int` | Copies received again (lost ACK) and ignored after re-ACKing. |
| `rejected` | `int` | Packets dropped: bad HMAC, malformed, stale, or host not allowed. |
| `delivery_failures` | `int` | Sends that ended in `DeliveryError`. |

`Addr` is an alias for `tuple[str, int]`, an `(ip, port)` pair. `DEFAULT_PORT` is `50555`.

### Exceptions

<!-- api -->
| Exception | Raised when |
|---|---|
| `DeliveryError` | `send()` got no ACK after `max_retries` retransmissions (peer down, wrong secret, firewall). |
| `PayloadTooLargeError` | The encoded message would be larger than one 1200-byte datagram (about 1100 bytes of text). Subclass of `ProtocolError`. |
| `AuthenticationError` | A packet's HMAC doesn't match: wrong secret or tampering. The library counts and drops these itself. Subclass of `ProtocolError`. |
| `ProtocolError` | Base class for malformed or unacceptable packets. |

### Command-line helpers

<!-- api -->
| Function | Returns | What it does |
|---|---|---|
| `add_cli_args(parser: argparse.ArgumentParser)` | `None` | Adds `--config`, `--server-ip`, `--bind-ip` and `--port` to your parser. The secret is deliberately not a flag: command lines are visible to other users in `ps`. |
| `config_from_cli(args: argparse.Namespace)` | `LinkConfig` | `LinkConfig.load()` with the parsed options as overrides. |

```bash
python3 -m udp_link keygen   # print a new random secret
python3 -m udp_link ip       # print this machine's LAN IP
```

## Video streaming

Video goes through the separate [`video_link/`](../video_link/README.md) module, on its own port, and shares this `link.toml`. It has its own [idle mode](../video_link/README.md#idle-mode-low-power), set in the `[video]` table.

## Tests

```bash
python3 -m unittest -v
```

The tests cover these cases: two-way exchange, stacking, 100 concurrent sends, packet loss and retransmission, lost-ACK de-duplication, wrong-secret rejection, tampering, stale packets, oversized payloads, peer cleanup on BYE, the stack limit, idle mode, and the confirmed-request example (replies matched by id, errors, timeouts).
