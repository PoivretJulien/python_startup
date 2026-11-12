# DHT PC client

Receives the temperature / humidity readings that the Arduino UNO Q app
(`../dht-thermal-sensor`) broadcasts over the LAN with `udp_link`
(acknowledged, HMAC-authenticated UDP strings, standard library only).

```
 UNO Q MCU (sketch.ino)  --Bridge.notify-->  UNO Q Linux (main.py + sensor_server.py)
                                                   |  UDP 50555, HMAC secret
                                                   v
                                             PC (apps/pc_client.py)
```

Requires Python 3.11+ on the PC. No pip install needed.

## 1. Credentials (copied by hand over SSH, never committed)

Both sides must have the **same** `link.toml` (same `server_ip`, `port`, `secret`).

On the board (SSH in, then in the app's `python/network` folder):

```bash
ssh arduino@<board-ip>
cd ~/ArduinoApps/dht-thermal-sensor/python/network   # adjust to where App Lab stores the app
python3 -m udp_link ip        # LAN IP of the board -> server_ip
python3 -m udp_link keygen    # new shared secret   -> secret
cp link.example.toml link.toml
nano link.toml                # set server_ip = "<board-ip>" and secret = "<key>"
chmod 600 link.toml
```

Then copy that file to the PC, into this folder:

```bash
scp arduino@<board-ip>:~/ArduinoApps/dht-thermal-sensor/python/network/link.toml ./link.toml
chmod 600 link.toml
```

(Or the other way round: write `link.toml` here and `scp` it to the board.
Either way the secret only travels over SSH.)

## 2. Run

On the board, start the app in App Lab (it runs `python/main.py`, which starts
the sensor server). Then on the PC:

```bash
python3 -m apps.pc_client --config link.toml
```

```
16:51:26 INFO    pc_client: connected
16:51:26  23.4 °C   56.0 %RH   (seq 41, age 0.03 s)
16:51:28  23.5 °C   56.1 %RH   (seq 42, age 0.02 s)
16:51:56 INFO    pc_client: summary of 15 message(s): temp avg=23.4 min=23.3 max=23.5 °C | hum avg=56.0 ...
```

Options:

| flag | effect |
|---|---|
| `--batch-interval 60` | seconds per avg/min/max summary (default 30) |
| `--csv readings.csv` | append every reading to a CSV file |
| `--interactive` | type `get` for the latest reading, or any text to send to the board |
| `-v` | debug output |

Windows: use `py -m apps.pc_client --config link.toml` (`--interactive` is macOS/Linux only).

## Test without the board

From `../dht-thermal-sensor/python/network`, with a `link.toml` whose
`server_ip` is `127.0.0.1`:

```bash
python3 -m apps.sensor_server --config link.toml --simulate
```

then run the client here with `--config` pointing to the same file.

## Troubleshooting

- Client stuck on "connecting": wrong `server_ip`, port, or secret, or a firewall
  blocks UDP 50555 (on the PC and/or the board).
- `udp_link/` in this folder is a copy of the library in
  `../dht-thermal-sensor/python/network/udp_link`; keep both at the same version.
