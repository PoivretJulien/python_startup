"""PC side of the DHT thermal sensor link.

    pc_client.py   receives the DHT readings broadcast by the UNO Q
                   (dht-thermal-sensor/python/network/apps/sensor_server.py)
    _common.py     shared helpers (logging, Ctrl+C handling, 'key=value' parsing)

Start with:  python3 -m apps.pc_client --config link.toml
"""
