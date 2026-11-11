"""Pan and zoom gestures, with one vocabulary on macOS, Windows and Linux.

GLFW has a single channel for all of this — the scroll callback — and every
system feeds it differently. This module listens to that channel (and, on
macOS, to the native pinch through ``macos_gestures3d``) and reduces everything
to two calls::

    on_pan(dx, dy)      move the content by this many pixels (it follows the fingers)
    on_zoom(factor)     multiply the scale by this (> 1 = closer, < 1 = further)

What produces them:

    =====================  ==========================  ===========================
    gesture                macOS                       Windows / Linux
    =====================  ==========================  ===========================
    pinch (two fingers)    zoom  (native, exact)       zoom  (see below)
    two-finger scroll      pan                         pan   (see below)
    mouse wheel            zoom                        zoom
    Ctrl + scroll / wheel  zoom                        zoom
    Shift + wheel          pan                         pan
    =====================  ==========================  ===========================

**macOS** — handled by ``macos_gestures3d``, unchanged: the pinch comes from
AppKit's own magnification event and a trackpad scroll is told apart from a
wheel by the system's "precise" flag.

**Windows** — a precision touchpad *pinch* is translated by Windows itself into
"Ctrl + wheel" before any program sees it, which is why Ctrl + scroll is a zoom
here: the pinch arrives through it, with no extra code. A two-finger scroll
arrives as small fractional wheel steps, a real wheel as whole notches; that
difference separates pan from zoom.

**Linux** — on Wayland a two-finger scroll is fractional too, and pans. On X11,
GLFW only receives whole "clicks" for a touchpad scroll, exactly like a wheel,
so there it zooms; and no Linux desktop hands the pinch to GLFW at all. Use
Ctrl + two-finger scroll to zoom and Shift + scroll (or a mouse drag) to pan.

Nothing here depends on the App: ``install`` only needs a GLFW window and two
functions. ``app3d.App`` installs it for you and exposes ``app.pan()`` and
``app.zoom()``.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
import time

import glfw
import macos_gestures3d
from settings3d import PAN_PIXELS_PER_SCROLL, WHEEL_PAN_PIXELS

WHEEL_ZOOM_STEP = 0.15  # one wheel notch multiplies the scale by e**0.15 (about 16 %)
# Windows and Linux (Wayland) report a touchpad scroll in "wheel notches" too,
# as fractions; one notch of finger travel is worth about this many pixels.
TOUCHPAD_PIXELS_PER_NOTCH = 60.0
# A touchpad occasionally reports a whole number in the middle of a scroll;
# keep treating the stream as a touchpad for this long after a fractional step.
TOUCHPAD_MEMORY_SECONDS = 0.4

MAC = sys.platform == "darwin"
WINDOWS = sys.platform == "win32"


def _control_down(window) -> bool:
    """True when Ctrl is held — or when Windows pretends it is, during a pinch."""
    if (
        glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_CONTROL) == glfw.PRESS
    ):
        return True
    if WINDOWS:
        # A touchpad pinch sets the Ctrl state for the duration of its wheel
        # messages without any key event, so GLFW's own key table misses it.
        try:
            import ctypes

            return bool(ctypes.windll.user32.GetKeyState(0x11) & 0x8000)  # VK_CONTROL
        except Exception:  # pragma: no cover - never let input handling raise
            return False
    return False


def _shift_down(window) -> bool:
    return (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )


def install(window, on_pan, on_zoom) -> str:
    """Send the pan and zoom gestures of ``window`` to the two functions.

    ``on_pan(dx, dy)`` receives pixels to add to the content's position;
    ``on_zoom(factor)`` a number to multiply the scale by. This takes over the
    window's GLFW scroll callback. Returns a short description of what is
    active on this system, for a status line or a log.
    """
    last_touchpad = [-1.0]  # time of the last scroll step that looked like a touchpad

    def on_pinch(magnification: float) -> None:
        # macOS magnification is the relative change of this step: +0.02 = 2 % closer.
        on_zoom(max(1.0 + magnification, 0.01))

    native = False
    if MAC:
        # The module announces itself on stdout; a library should stay quiet.
        with contextlib.redirect_stdout(io.StringIO()):
            native = macos_gestures3d.install(window, on_pinch)

    def looks_like_touchpad(dx: float, dy: float) -> bool:
        if native:
            return macos_gestures3d.scroll_is_precise()  # the system knows
        now = time.perf_counter()
        fractional = dx != round(dx) or dy != round(dy)
        if fractional:
            last_touchpad[0] = now
            return True
        return now - last_touchpad[0] < TOUCHPAD_MEMORY_SECONDS

    def on_scroll(_window, dx: float, dy: float) -> None:
        if _control_down(window):
            # Ctrl + scroll, and the Windows touchpad pinch, which is the same thing.
            on_zoom(math.exp(dy * WHEEL_ZOOM_STEP))
        elif _shift_down(window):
            # Shift + wheel: pan, for wheels without a horizontal axis.
            on_pan(dx * WHEEL_PAN_PIXELS, dy * WHEEL_PAN_PIXELS)
        elif looks_like_touchpad(dx, dy):
            pixels = PAN_PIXELS_PER_SCROLL if native else TOUCHPAD_PIXELS_PER_NOTCH
            on_pan(dx * pixels, dy * pixels)
        else:
            on_zoom(math.exp(dy * WHEEL_ZOOM_STEP))  # a wheel notch

    glfw.set_scroll_callback(window, on_scroll)

    if native:
        return "pinch = zoom, two-finger scroll = pan, wheel = zoom"
    if WINDOWS:
        return "pinch or Ctrl+scroll = zoom, two-finger scroll = pan, wheel = zoom"
    return "Ctrl+scroll or wheel = zoom, two-finger scroll (Wayland) or Shift+scroll = pan"
