"""Generate assets/lan_mesh.svg: a symbolic illustration of a local network mesh.

    python3 tools/make_mesh_illustration.py          # SVG (animated in browsers)
    python3 tools/make_mesh_illustration.py --png    # + 3200x1800 PNG (macOS)

Deterministic (fixed seed): rerunning gives the same picture. Several server
devices (A, B, C, ...) and a local client sit in a mesh around a Wi-Fi capable
switch. Each server has a route to the local client through the switch, and
packets travel both ways on every route (the animation plays in browsers; the
PNG shows the packets frozen along the routes).

To add a server device, append an entry to SERVERS.
"""

from __future__ import annotations

import math
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

W, H = 1600, 900
CX, CY = 800, 470
OUT = Path(__file__).resolve().parent.parent / "assets" / "lan_mesh.svg"
MONO = "ui-monospace,Menlo,monospace"

EDGE, NODE_FILL, NODE_STROKE = "#4a6aa3", "#16223b", "#6f8fca"
CLIENT_COLOR = "#5b9bff"
OUTER_COUNT = 12  # devices on the edge of the LAN; slot k sits at angle k * 30 degrees


@dataclass(frozen=True)
class Device:
    label: str
    address: str
    color: str
    slot: int  # position on the outer ring (0 = right, 3 = bottom, 6 = left, 9 = top)
    label_above: bool = False
    icon: str = "server"


SERVERS = [
    Device("server device A", "192.168.1.10", "#f5a524", 6),
    Device("server device B", "192.168.1.11", "#34d399", 10, label_above=True),
    Device("server device C", "192.168.1.12", "#f472b6", 4),
    Device("server device D", "192.168.1.13", "#c084fc", 2),
]
CLIENT = Device("local client", "192.168.1.20", CLIENT_COLOR, 0, icon="client")


def f(v: float) -> str:
    return f"{v:.1f}"


def dist(p: tuple[float, float], q: tuple[float, float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


# --------------------------------------------------------------------------- layout
rng = random.Random(7)


def ring(n: int, radius: float, squash: float, phase: float, jitter: float) -> list[tuple[float, float]]:
    pts = []
    for k in range(n):
        a = phase + 2 * math.pi * k / n + rng.uniform(-jitter, jitter)
        r = radius * rng.uniform(0.9, 1.08)
        pts.append((CX + r * math.cos(a) * 1.55, CY + r * math.sin(a) * squash))
    return pts


HUB = (CX, CY)
INNER = ring(6, 170, 0.95, math.pi / 6, 0.12)
OUTER = ring(OUTER_COUNT, 300, 0.92, 0.0, 0.08)
for dev in (*SERVERS, CLIENT):  # pin highlighted devices exactly on their slot
    a = 2 * math.pi * dev.slot / OUTER_COUNT
    OUTER[dev.slot] = (CX + 300 * math.cos(a) * 1.55, CY + 300 * math.sin(a) * 0.92)

EDGES: set[tuple[tuple[str, int], tuple[str, int]]] = set()


def link(a: tuple[str, int], b: tuple[str, int]) -> None:
    EDGES.add((a, b) if a <= b else (b, a))


for i in range(len(INNER)):
    link(("h", 0), ("i", i))
    link(("i", i), ("i", (i + 1) % len(INNER)))
for j, p in enumerate(OUTER):
    link(("o", j), ("o", (j + 1) % len(OUTER)))
    for i in sorted(range(len(INNER)), key=lambda i: dist(p, INNER[i]))[:2]:
        link(("i", i), ("o", j))

POS = {"h": [HUB], "i": INNER, "o": OUTER}


def nearest_inner(p: tuple[float, float]) -> int:
    return min(range(len(INNER)), key=lambda i: dist(p, INNER[i]))


def route_for(dev: Device) -> list[tuple[str, int]]:
    """server -> nearest inner node -> switch -> inner node near client -> client"""
    return [("o", dev.slot), ("i", nearest_inner(OUTER[dev.slot])), ("h", 0),
            ("i", nearest_inner(OUTER[CLIENT.slot])), ("o", CLIENT.slot)]


ROUTES = [(dev, route_for(dev)) for dev in SERVERS]
ROUTE_EDGES = {tuple(sorted(pair)) for _, r in ROUTES for pair in zip(r, r[1:])}


def point_along(points: list[tuple[float, float]], t: float) -> tuple[float, float]:
    """Point at fraction t (0..1) of the polyline's length."""
    lengths = [dist(a, b) for a, b in zip(points, points[1:])]
    target = t * sum(lengths)
    for (a, b), seg in zip(zip(points, points[1:]), lengths):
        if target <= seg:
            k = target / seg if seg else 0
            return a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k
        target -= seg
    return points[-1]


# --------------------------------------------------------------------------- drawing
def build(animated: bool) -> str:
    parts: list[str] = []
    add = parts.append

    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        'aria-label="Symbolic local network mesh: several server devices and a local client exchange packets '
        'in both directions through a Wi-Fi switch, among other devices on the LAN">')
    gradients = "".join(
        f'<linearGradient id="route{k}" gradientUnits="userSpaceOnUse" x1="{f(OUTER[d.slot][0])}" y1="{f(OUTER[d.slot][1])}" '
        f'x2="{f(OUTER[CLIENT.slot][0])}" y2="{f(OUTER[CLIENT.slot][1])}">'
        f'<stop offset="0" stop-color="{d.color}"/><stop offset="1" stop-color="{CLIENT_COLOR}"/></linearGradient>'
        for k, (d, _) in enumerate(ROUTES)
    )
    add(f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0a1020"/><stop offset="1" stop-color="#121c33"/></linearGradient>
  <radialGradient id="glow" cx="50%" cy="52%" r="55%"><stop offset="0" stop-color="#2a4a8a" stop-opacity=".45"/><stop offset="1" stop-color="#2a4a8a" stop-opacity="0"/></radialGradient>
  <pattern id="dots" width="28" height="28" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1.2" fill="#8fb0ff" fill-opacity=".09"/></pattern>
  {gradients}
  <filter id="soft" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8"/></filter>
  <filter id="halo" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="14"/></filter>
</defs>""")
    add(f'<rect width="{W}" height="{H}" fill="url(#bg)"/>')
    add(f'<rect width="{W}" height="{H}" fill="url(#dots)"/>')
    add(f'<ellipse cx="{CX}" cy="{CY}" rx="700" ry="400" fill="url(#glow)"/>')

    # LAN boundary
    add(f'<ellipse cx="{CX}" cy="{CY}" rx="640" ry="345" fill="none" stroke="#8fb0ff" stroke-opacity=".28" '
        'stroke-width="1.5" stroke-dasharray="2 10" stroke-linecap="round"/>')
    add(f'<text x="190" y="842" fill="#8fb0ff" fill-opacity=".7" font-family="{MONO}" '
        'font-size="18" letter-spacing="2">LAN · 192.168.1.0/24</text>')

    # Wi-Fi coverage around the switch: fixed rings, plus waves rippling out when animated
    for r, op in ((70, 0.22), (110, 0.13), (150, 0.07)):
        add(f'<circle cx="{CX}" cy="{CY}" r="{r}" fill="none" stroke="#8fb0ff" stroke-opacity="{op}" stroke-width="1.2"/>')
    if animated:
        for k in range(3):
            add(f'<circle cx="{CX}" cy="{CY}" r="50" fill="none" stroke="#9fc1ff" stroke-width="2" opacity="0">'
                f'<animate attributeName="r" values="50;210" dur="3.6s" begin="{-k * 1.2:.1f}s" repeatCount="indefinite"/>'
                f'<animate attributeName="opacity" values=".55;0" dur="3.6s" begin="{-k * 1.2:.1f}s" repeatCount="indefinite"/>'
                '</circle>')

    # mesh links
    for a, b in sorted(EDGES):
        if (a, b) in ROUTE_EDGES:
            continue
        (x1, y1), (x2, y2) = POS[a[0]][a[1]], POS[b[0]][b[1]]
        op = 0.55 if "h" in (a[0], b[0]) else 0.38
        add(f'<line x1="{f(x1)}" y1="{f(y1)}" x2="{f(x2)}" y2="{f(y2)}" stroke="{EDGE}" stroke-opacity="{op}" stroke-width="1.4"/>')

    # routes: glow + line
    for k, (dev, route) in enumerate(ROUTES):
        d = "M " + " L ".join(f"{f(POS[n][i][0])} {f(POS[n][i][1])}" for n, i in route)
        add(f'<path d="{d}" fill="none" stroke="url(#route{k})" stroke-width="9" stroke-opacity=".28" filter="url(#soft)" stroke-linejoin="round"/>')
        add(f'<path id="routePath{k}" d="{d}" fill="none" stroke="url(#route{k})" stroke-width="2.6" '
            'stroke-linejoin="round" stroke-linecap="round" stroke-opacity=".9"/>')

    # packets, both directions on every route: server -> client in the server's
    # color, client -> server in the client's color, each route at its own pace
    for k, (dev, route) in enumerate(ROUTES):
        pts = [POS[n][i] for n, i in route]
        dur = 3.4 + 0.7 * k
        # phases chosen so frozen packets (PNG) sit on open stretches of the route,
        # not hidden under the end devices or the switch
        flows = [(dev.color, False, 0.2), (dev.color, False, 0.7), (CLIENT_COLOR, True, 0.15), (CLIENT_COLOR, True, 0.62)]
        for color, reverse, phase in flows:
            if animated:
                keys = ' keyPoints="1;0" keyTimes="0;1" calcMode="linear"' if reverse else ""
                begin = -(phase + 0.03 * k) * dur
                add(f'<circle r="5.5" fill="{color}"><animateMotion dur="{dur:.1f}s" begin="{begin:.2f}s" '
                    f'repeatCount="indefinite"{keys}><mpath href="#routePath{k}"/></animateMotion></circle>')
            else:
                t = (phase + 0.03 * k) % 1.0
                x, y = point_along(pts, 1 - t if reverse else t)
                add(f'<circle cx="{f(x)}" cy="{f(y)}" r="10" fill="{color}" opacity=".35" filter="url(#soft)"/>'
                    f'<circle cx="{f(x)}" cy="{f(y)}" r="5" fill="{color}"/>')

    # label plates go here (inserted by device()), over the lines, under the dots
    plates_at = len(parts)

    # ordinary devices
    special = {d.slot for d in (*SERVERS, CLIENT)}
    for name, pts in (("i", INNER), ("o", OUTER)):
        for idx, (x, y) in enumerate(pts):
            if name == "o" and idx in special:
                continue
            r = 10 if name == "i" else 8
            add(f'<circle cx="{f(x)}" cy="{f(y)}" r="{r + 6}" fill="#6f8fca" fill-opacity=".08"/>')
            add(f'<circle cx="{f(x)}" cy="{f(y)}" r="{r}" fill="{NODE_FILL}" stroke="{NODE_STROKE}" stroke-width="1.6"/>')
            add(f'<circle cx="{f(x)}" cy="{f(y)}" r="2.6" fill="#a9c2f0"/>')

    # switch: hexagon with a Wi-Fi symbol
    hexagon = " ".join(
        f"{f(CX + 44 * math.cos(math.pi / 6 + k * math.pi / 3))},{f(CY + 44 * math.sin(math.pi / 6 + k * math.pi / 3))}"
        for k in range(6)
    )
    add(f'<circle cx="{CX}" cy="{CY}" r="52" fill="#5b7bd6" fill-opacity=".25" filter="url(#halo)"/>')
    add(f'<polygon points="{hexagon}" fill="#1a2a4c" stroke="#9fb8ef" stroke-width="2"/>')
    wx, wy = CX, CY + 13
    for r in (9, 17, 25):
        a0, a1 = math.radians(-135), math.radians(-45)
        add(f'<path d="M {f(wx + r * math.cos(a0))} {f(wy + r * math.sin(a0))} A {r} {r} 0 0 1 '
            f'{f(wx + r * math.cos(a1))} {f(wy + r * math.sin(a1))}" fill="none" stroke="#cfe0ff" '
            'stroke-width="3.2" stroke-linecap="round"/>')
    add(f'<circle cx="{wx}" cy="{wy}" r="3.4" fill="#cfe0ff"/>')
    add(f'<rect x="{CX - 92}" y="{CY + 62}" width="184" height="28" rx="9" fill="#101a30" fill-opacity=".85"/>')
    add(f'<text x="{CX}" y="{CY + 81}" text-anchor="middle" fill="#9fb8ef" fill-opacity=".9" '
        f'font-family="{MONO}" font-size="16" letter-spacing="1.5">switch · wi-fi</text>')

    # highlighted devices
    for dev in (*SERVERS, CLIENT):
        x, y = OUTER[dev.slot]
        c = dev.color
        add(f'<circle cx="{f(x)}" cy="{f(y)}" r="44" fill="{c}" fill-opacity=".28" filter="url(#halo)"/>')
        add(f'<circle cx="{f(x)}" cy="{f(y)}" r="34" fill="#111a2e" stroke="{c}" stroke-width="2.5"/>')
        add(f'<circle cx="{f(x)}" cy="{f(y)}" r="44" fill="none" stroke="{c}" stroke-opacity=".35" stroke-width="1.2"/>')
        if dev.icon == "server":
            add("".join(
                f'<rect x="{f(x - 15)}" y="{f(y - 15 + k * 11)}" width="30" height="8" rx="2" fill="none" stroke="{c}" stroke-width="2"/>'
                f'<circle cx="{f(x + 9)}" cy="{f(y - 11 + k * 11)}" r="1.6" fill="{c}"/>'
                for k in range(3)
            ))
        else:
            add(f'<rect x="{f(x - 17)}" y="{f(y - 15)}" width="34" height="22" rx="3" fill="none" stroke="{c}" stroke-width="2"/>'
                f'<line x1="{f(x)}" y1="{f(y + 7)}" x2="{f(x)}" y2="{f(y + 13)}" stroke="{c}" stroke-width="2"/>'
                f'<line x1="{f(x - 9)}" y1="{f(y + 14)}" x2="{f(x + 9)}" y2="{f(y + 14)}" stroke="{c}" stroke-width="2" stroke-linecap="round"/>')
        top = y - 106 if dev.label_above else y + 50
        parts.insert(plates_at, f'<rect x="{f(x - 112)}" y="{f(top)}" width="224" height="56" rx="12" fill="#101a30" fill-opacity=".92"/>')
        add(f'<text x="{f(x)}" y="{f(top + 22)}" text-anchor="middle" fill="{c}" font-family="{MONO}" '
            f'font-size="19" font-weight="700" letter-spacing=".5">{dev.label}</text>')
        add(f'<text x="{f(x)}" y="{f(top + 46)}" text-anchor="middle" fill="#c4d2ee" fill-opacity=".7" '
            f'font-family="{MONO}" font-size="14">{dev.address}</text>')

    add(f'<text x="1410" y="842" text-anchor="end" fill="#8fb0ff" fill-opacity=".45" '
        f'font-family="{MONO}" font-size="14" letter-spacing="1">local network mesh</text>')
    add("</svg>")
    return "\n".join(parts)


# --------------------------------------------------------------------------- PNG export (macOS)
def export_png(svg: str, out: Path, scale: int = 2) -> None:
    """Render with Quick Look (square canvas), then crop to 16:9 with sips."""
    pad = (W - H) // 2
    square = svg.replace(
        f'viewBox="0 0 {W} {H}" width="{W}" height="{H}"',
        f'viewBox="0 {-pad} {W} {W}" width="{W // 2}" height="{W // 2}"',
        1,
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "mesh.svg"
        src.write_text(square, encoding="utf-8")
        subprocess.run(["qlmanage", "-t", "-s", str(W * scale), "-o", tmp, str(src)], check=True, capture_output=True)
        rendered = Path(tmp) / "mesh.svg.png"
        subprocess.run(["sips", "-c", str(H * scale), str(W * scale), str(rendered), "--out", str(out)],
                       check=True, capture_output=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(animated=True), encoding="utf-8")
    print(f"wrote {OUT}")
    if "--png" in sys.argv:
        export_png(build(animated=False), OUT.with_suffix(".png"))
