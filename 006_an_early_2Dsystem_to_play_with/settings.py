"""Tunable constants for the experimental 2D mid-step.

Everything you are likely to tweak lives here: window size, grid extent and
spacing, colours, arrow-head shape, the camera's zoom/pan feel, and the terminal
readout interval.
"""

from __future__ import annotations

WIDTH, HEIGHT = 1000, 800
TITLE = "mid-step 01½ — 2D grid + arrow"

HALF = 5.0  # grid spans -HALF..+HALF on both axes
STEP = 1.0  # one grid square = one world unit

BACKGROUND = (0.06, 0.06, 0.07)
GRID_COLOR = (0.28, 0.28, 0.32)
AXIS_COLOR = (0.95, 0.45, 0.10)
ARROW_COLOR = (0.20, 0.85, 0.95)

ARROW_HEAD_LENGTH = 0.35
ARROW_HEAD_ANGLE = 28.0  # degrees, half-opening of the head
TAIL_MARKER = 0.12  # half-size of the cross at the arrow tail
PRINT_STEP = 0.25  # world units between automatic terminal prints

# --- camera: zoom & pan (geometry.Camera) --------------------------------
# `view_half_y` is how many world units fit between the viewport centre and its
# top edge, so zoom = HALF / view_half_y (1.0 at startup).
CAMERA_MIN_HALF = 0.05  # closest zoom-in: 100x, ~0.005 world units per pixel
CAMERA_MAX_HALF = 400.0  # furthest zoom-out: ~1/80x

WHEEL_ZOOM_STEP = 0.18  # exp() sensitivity per wheel notch (dy = +/-1)
PINCH_ZOOM_STEP = 6.0  # exp() sensitivity per unit of macOS `magnification`
# GLFW divides trackpad scroll deltas by 10; multiplying back restores point-sized
# deltas so a two-finger pan moves the content 1:1 with the fingers.
PAN_PIXELS_PER_SCROLL = 10.0
# A wheel notch is a unit-less +/-1, so shift+wheel panning needs its own scale.
WHEEL_PAN_PIXELS = 40.0
# Flip either sign to invert a two-finger trackpad pan (macOS has both "natural"
# scrolling and per-app expectation; middle-drag panning is unaffected).
SCROLL_PAN_SIGN_X = 1.0
SCROLL_PAN_SIGN_Y = 1.0

# --- adaptive grid -------------------------------------------------------
# Grid squares target this many pixels on screen, so the step stays 1 world unit
# at 1:1 zoom on a normal window and only changes when zooming makes squares
# unreadably small/large. Chosen from the 1/2/5 x 10^k ladder, so on-screen
# spacing lands in roughly [GRID_MAX_SPACING_PX / 5, GRID_MAX_SPACING_PX].
GRID_MAX_SPACING_PX = 110.0
GRID_MAX_LINES = 400  # total segments, a safety cap at extreme aspect ratios

# --- GPU anti-aliased renderer (renderer_aa.py) --------------------------
# Stroke widths are in *pixels* here, so lines stay the same on-screen weight
# whatever the window size or DPI is. The plain renderer has no width at all:
# it is stuck at the ONE pixel GL_LINES gives it.
AA_GRID_WIDTH_PX = 1.2
AA_ARROW_WIDTH_PX = 1.4
AA_CIRCLE_WIDTH_PX = 1.8
AA_LINE_WIDTH_PX = 1.8  # used by draw_line when thickness == 0
AA_TEXT_WIDTH_PX = 1.6  # stroke weight of the built-in font (font.py)
AA_FEATHER_PX = 1.5  # quad padding around a stroke that holds the AA fade
