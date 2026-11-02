"""Tunable constants for the 3D version of the experimental 2D system.

The 2D app (``experimental/main.py``) uses an orthographic pan/zoom camera; this
one keeps the same drawing API but swaps in step06's orbit camera. Everything
you are likely to tweak lives here: window size, grid size/spacing, colours,
arrow-head shape, camera feel, AA stroke widths and the terminal readout
interval.
"""

from __future__ import annotations

WIDTH, HEIGHT = 1100, 800
TITLE = "experimental_b — 3D grid + arrows"

HALF = 5.0  # default grid radius around the camera pivot, in world units
STEP = 1.0  # one grid square = one world unit at the initial framing
GROUND_Z = 0.0  # the floor grid lives on z = GROUND_Z (z is up)
SCENE_Z = 0.02  # scene strokes float just above it, out of the grid's depth

BACKGROUND = (0.06, 0.06, 0.07)
GRID_COLOR = (0.28, 0.28, 0.32)
# In 2D both axes shared one orange. The 3D scene is z-up, so the three axes
# get the familiar colours: x red, y green, z blue.
AXIS_X_COLOR = (1.00, 0.18, 0.18)  # red
AXIS_Y_COLOR = (0.35, 0.90, 0.35)  # green
AXIS_Z_COLOR = (0.30, 0.55, 1.00)  # blue
ARROW_COLOR = (0.20, 0.85, 0.95)
CLOCK_COLOR = (1.00, 1.00, 0.00)

ARROW_HEAD_LENGTH = 0.35
ARROW_HEAD_ANGLE = 28.0  # degrees, half-opening of the head
TAIL_MARKER = 0.12  # half-size of the cross at the arrow tail
PRINT_STEP = 0.25  # world units between automatic terminal prints

# --- camera: orbit / pan / zoom (camera.OrbitCamera) ----------------------
CAMERA_FOV = 60.0
CAMERA_NEAR = 0.1
CAMERA_FAR = 200.0
CAMERA_MIN_DISTANCE = 1.5  # closest zoom-in
CAMERA_MAX_DISTANCE = 40.0  # furthest zoom-out
CAMERA_MAX_PITCH = 1.45  # radians, just short of straight down/up
CAMERA_ORBIT_SPEED = 0.008  # radians per pixel of left-drag
CAMERA_ZOOM_SPEED = 0.5  # world units per wheel notch
CAMERA_PAN_SPEED = 0.0012  # fraction of the pivot distance per pixel of middle-drag

# --- macOS trackpad gestures (macos_gestures3d.py) -----------------------
# GLFW divides trackpad scroll deltas by 10; multiplying back restores
# point-sized deltas so two-finger panning moves the scene 1:1 with the fingers.
PAN_PIXELS_PER_SCROLL = 10.0
# A wheel notch is a unit-less +/-1, so shift+wheel panning needs its own scale.
WHEEL_PAN_PIXELS = 40.0
# Flip either sign to invert a two-finger trackpad pan.
SCROLL_PAN_SIGN_X = 1.0
SCROLL_PAN_SIGN_Y = 1.0
PINCH_ZOOM_STEP = 6.0  # exp() sensitivity per unit of macOS `magnification`

CAMERA_START_YAW = 0.6
CAMERA_START_PITCH = 0.4
CAMERA_START_DISTANCE = 6.5
CAMERA_START_TARGET = (0.0, 0.0, 0.0)

# --- adaptive grid -------------------------------------------------------
# Grid squares target this many pixels at the camera pivot, so the step stays
# 1 world unit at the initial framing and snaps to 1/2/5 x 10^k afterwards.
# The grid covers GRID_EXTENT_FACTOR * distance around the pivot, so panning
# never runs off its edge.
GRID_MAX_SPACING_PX = 110.0
GRID_MAX_LINES = 400  # total segments, a safety cap at extreme aspects/zoom
GRID_EXTENT_FACTOR = 2.5

# --- GPU anti-aliased renderer (renderer_aa.py) --------------------------
# Stroke widths are in *pixels*: they keep their on-screen weight at any zoom,
# window size or HiDPI scale. The plain renderer has no width at all for its
# GL_LINES strokes.
AA_GRID_WIDTH_PX = 1.2
AA_ARROW_WIDTH_PX = 1.4
AA_CIRCLE_WIDTH_PX = 1.8
AA_LINE_WIDTH_PX = 1.8  # used by draw_line when thickness == 0
AA_TEXT_WIDTH_PX = 1.6  # stroke weight of the built-in font (font.py)
AA_FEATHER_PX = 2.0  # quad margin (pixels) that holds the one-pixel AA fade
