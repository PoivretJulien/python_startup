"""Example 8 — pan and zoom a high-resolution map inside a rounded frame.

A large picture (4096 x 2048 pixels) is shown through a rounded window in the
middle of the screen. Inside that frame:

    pinch, mouse wheel or Ctrl + scroll      zoom, around the cursor
    two-finger scroll or Shift + wheel       pan
    click and drag                           pan
    +  and  -                                zoom, around the centre
    R                                        back to the whole map

Outside the frame nothing reacts: the gestures are switched off as soon as the
cursor leaves it, and the border turns grey to say so.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/08_map_viewer/main.py                    # the built-in map
    .venv/bin/python examples/08_map_viewer/main.py --image photo.jpg  # your own picture
    .venv/bin/python examples/08_map_viewer/main.py --poll             # always redraw

The map
-------
The built-in map is *computed* when the program starts (about a second): random
terrain, coloured by height, with shading, contour lines and a grid. Nothing is
downloaded, so the example works offline and its picture belongs to nobody.
Map imagery from Google, and most online map services, may not be copied into
your own program; ``--image`` accepts any picture you do have the right to
use — a public-domain NASA "Blue Marble" image is a good high-resolution one.

How it works
------------
Three shapes, drawn in this order:

1. ``draw_screen_image`` — the whole map, at its current position and size. It
   is usually far larger than the window; the graphics card simply clips it.
2. ``draw_screen_inverted_rect`` — the background colour everywhere *except* a
   rounded box. This is the frame: it hides the map outside the box.
3. ``draw_screen_rect_outline`` — the border of that same box.

The view is two values: ``scale`` (screen pixels per map pixel) and ``offset``
(where the map's top-left corner is on the screen). Panning adds to the
offset; zooming changes the scale and moves the offset so that the point under
the cursor stays under the cursor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The drawing library is the graphics/ folder of the project.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from settings3d import BACKGROUND  # noqa: E402
from texture3d import Texture  # noqa: E402

ORANGE = (1.00, 0.55, 0.10, 1.0)
GREY = (0.35, 0.35, 0.42, 1.0)
LABEL = (0.85, 0.92, 1.00)
MUTED = (0.55, 0.55, 0.62)
FRAME_COLOR = (*BACKGROUND, 1.0)  # what covers the map outside the frame

MARGIN = 36  # between the frame and the window, left and right
TOP, BOTTOM = 64, 64  # room for the title and the status lines
RADIUS = 30  # corners of the frame
MAX_SCALE = 16.0  # closest zoom: one map pixel = 16 screen pixels
KEY_ZOOM = 1.25  # what one press of + or - does
LARGEST_SIDE = 8192  # a GPU texture cannot be larger than this


# ------------------------------------------------------------------ the map
def make_map(width: int = 4096, height: int = 2048, seed: int = 7) -> np.ndarray:
    """A made-up terrain map as an ``(height, width, 3)`` uint8 RGB array.

    Random noise at several sizes is added up (large soft shapes for the
    continents, small ones for the detail), which gives a height for every
    pixel. The height picks the colour; its slope gives the shading.
    """
    from PIL import Image

    rng = np.random.default_rng(seed)
    elevation = np.zeros((height, width), dtype=np.float32)
    weight, total = 1.0, 0.0
    cells = 3  # the first layer has 3 x 6 random values; each next one doubles
    for _octave in range(9):
        coarse = rng.random((cells, cells * 2), dtype=np.float32)
        layer = Image.fromarray(coarse, mode="F").resize((width, height), Image.BICUBIC)
        elevation += weight * np.asarray(layer, dtype=np.float32)
        total += weight
        weight *= 0.5
        cells *= 2
    elevation /= total
    low, high = float(elevation.min()), float(elevation.max())
    elevation = (elevation - low) / (high - low)  # 0 = deepest, 1 = highest

    # Height -> colour: sea, shore, plains, forest, rock, snow.
    stops = [0.00, 0.42, 0.50, 0.52, 0.62, 0.74, 0.86, 1.00]
    colors = np.array(
        [
            (0.04, 0.10, 0.32), (0.10, 0.30, 0.62), (0.25, 0.55, 0.80),
            (0.85, 0.80, 0.58), (0.38, 0.62, 0.30), (0.20, 0.42, 0.22),
            (0.50, 0.42, 0.34), (0.96, 0.96, 0.98),
        ],
        dtype=np.float32,
    )
    rgb = np.stack([np.interp(elevation, stops, colors[:, c]) for c in range(3)], axis=-1)

    land = elevation > 0.51
    # Shading: light from the top-left, so slopes facing it are brighter.
    slope_y, slope_x = np.gradient(elevation)
    shade = np.clip(1.0 - 260.0 * (slope_x + slope_y), 0.55, 1.45)
    rgb[land] *= shade[land][:, None]
    # Contour lines every 4 % of height, on land only.
    steps = np.floor(elevation * 25.0)
    contour = land & ((steps != np.roll(steps, 1, axis=0)) | (steps != np.roll(steps, 1, axis=1)))
    rgb[contour] *= 0.72
    # A grid every 256 map pixels, to feel the scale while zooming.
    rgb[::256, :] = rgb[::256, :] * 0.6 + 0.4
    rgb[:, ::256] = rgb[:, ::256] * 0.6 + 0.4
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def load_picture(path: str) -> np.ndarray:
    """Any picture file as an RGB array, shrunk if the GPU could not hold it."""
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # large maps are the point of this example
    with Image.open(path) as picture:
        picture = picture.convert("RGB")
        if max(picture.size) > LARGEST_SIDE:
            picture.thumbnail((LARGEST_SIDE, LARGEST_SIDE), Image.LANCZOS)
        return np.asarray(picture)


# ---------------------------------------------------------------- the program
def main() -> int:
    parser = argparse.ArgumentParser(description="Pan and zoom a large picture inside a frame.")
    parser.add_argument("--image", help="a picture file to show instead of the built-in map")
    app = app3d.App("example 8 — map viewer", size=(900, 640), parser=parser)
    ui = app.aa

    if app.args.image:
        pixels = load_picture(app.args.image)
    else:
        print("building the map ...")
        pixels = make_map()
    map_h, map_w = pixels.shape[:2]
    # One texture, uploaded once. Its smaller copies (mipmaps, on by default)
    # keep the map smooth when it is zoomed far out.
    texture = Texture(pixels)

    # --- the view: where the map is, and how big -----------------------------
    scale = 0.0  # screen pixels per map pixel; 0 = "not set yet"
    offset = [0.0, 0.0]  # screen position of the map's top-left corner
    dragging = False
    was_down = False
    last_cursor = (0.0, 0.0)

    def frame_box() -> tuple[tuple[float, float], tuple[float, float]]:
        """The frame follows the window: (anchor, size)."""
        width, height = app.size()
        return (MARGIN, TOP), (max(width - 2 * MARGIN, 10), max(height - TOP - BOTTOM, 10))

    def fit_scale() -> float:
        """The scale at which the whole map fits in the frame."""
        _anchor, size = frame_box()
        return min(size[0] / map_w, size[1] / map_h)

    def reset_view() -> None:
        nonlocal scale
        anchor, size = frame_box()
        scale = fit_scale()
        offset[0] = anchor[0] + (size[0] - map_w * scale) / 2  # centred
        offset[1] = anchor[1] + (size[1] - map_h * scale) / 2

    def zoom_at(point, factor: float) -> None:
        """Change the scale, keeping the map pixel under ``point`` where it is."""
        nonlocal scale
        new_scale = min(max(scale * factor, fit_scale() * 0.5), MAX_SCALE)
        factor = new_scale / scale  # what the limits left of it
        # The point stays put: its distance to the map's corner grows by `factor`.
        offset[0] = point[0] - (point[0] - offset[0]) * factor
        offset[1] = point[1] - (point[1] - offset[1]) * factor
        scale = new_scale

    def draw_frame() -> None:
        nonlocal dragging, was_down, last_cursor

        if scale == 0.0:
            reset_view()  # first frame: the window size is known now
        anchor, size = frame_box()
        cursor = app.cursor()
        down = app.mouse_down()
        # The one question that switches the gestures on and off.
        inside = ui.in_screen_rect(cursor, anchor, size, RADIUS)

        # --- 1. gestures: only when the cursor is in the frame ----------------
        zoom = app.zoom()  # read every frame, used only if inside
        pan = app.pan()
        if inside:
            if zoom != 1.0:
                zoom_at(cursor, zoom)
            offset[0] += pan[0]
            offset[1] += pan[1]

        # --- 2. click and drag: starts inside, may continue outside -------------
        if down and not was_down and inside:
            dragging = True
        if not down:
            dragging = False
        was_down = down
        if dragging:
            offset[0] += cursor[0] - last_cursor[0]
            offset[1] += cursor[1] - last_cursor[1]
        last_cursor = cursor

        # --- 3. keyboard ----------------------------------------------------------
        centre = (anchor[0] + size[0] / 2, anchor[1] + size[1] / 2)
        if any(app.key_pressed(k, repeat=True) for k in ("=", "+", "KP ADD")):
            zoom_at(centre, KEY_ZOOM)
        if any(app.key_pressed(k, repeat=True) for k in ("-", "KP SUBTRACT")):
            zoom_at(centre, 1.0 / KEY_ZOOM)
        if app.key_pressed("R"):
            reset_view()

        # Never lose the map: the centre of the frame always stays on it.
        offset[0] = min(max(offset[0], centre[0] - map_w * scale), centre[0])
        offset[1] = min(max(offset[1], centre[1] - map_h * scale), centre[1])

        # --- 4. the picture: map, frame over it, border ---------------------------
        ui.draw_screen_image(offset, (map_w * scale, map_h * scale), texture)
        ui.draw_screen_inverted_rect(anchor, size, FRAME_COLOR, RADIUS)
        active = inside or dragging
        ui.draw_screen_rect_outline(anchor, size, ORANGE if active else GREY, RADIUS, 3)

        # --- 5. read-outs -----------------------------------------------------------
        ui.draw_screen_text(MARGIN, 40, "MAP VIEWER", 16.0, LABEL)
        state = "gestures on" if active else "gestures off: move the cursor into the frame"
        ui.draw_screen_text(MARGIN + 150, 40, state, 13.0, ORANGE if active else MUTED)

        _width, height = app.size()
        status = f"zoom {scale / fit_scale() * 100:.0f} %"
        if inside:
            # Screen -> map: undo the offset, then the scale.
            map_x = (cursor[0] - offset[0]) / scale
            map_y = (cursor[1] - offset[1]) / scale
            status += f"   map pixel ({map_x:.0f}, {map_y:.0f}) of {map_w} x {map_h}"
        ui.draw_screen_text(MARGIN, height - 34, status, 13.0, LABEL)
        ui.draw_screen_text(
            MARGIN, height - 14, f"{app.gestures}   drag = pan   + / - = zoom   R = reset",
            11.0, MUTED,
        )

    try:
        app.run(draw_frame)
    finally:
        texture.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
