"""Tunable constants for the experimental 2D mid-step.

Everything you are likely to tweak lives here: window size, grid extent and
spacing, colours, arrow-head shape, and the terminal readout interval.
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

# --- GPU anti-aliased renderer (renderer_aa.py) --------------------------
# Stroke widths are in *pixels* here, so lines stay the same on-screen weight
# whatever the window size or DPI is. The plain renderer has no width at all:
# it is stuck at the ONE pixel GL_LINES gives it.
AA_GRID_WIDTH_PX = 1.2
AA_ARROW_WIDTH_PX = 1.4
AA_CIRCLE_WIDTH_PX = 1.8
AA_LINE_WIDTH_PX = 1.8  # used by draw_line when thickness == 0
AA_FEATHER_PX = 1.5  # quad padding around a stroke that holds the AA fade
