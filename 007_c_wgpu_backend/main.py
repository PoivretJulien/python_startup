"""
A advance stand-alone python script visualization tool.
establish a 3d system to draw line circle text and particles
support anti aliasing and GPU acceleration.

GPU backend: wgpu (Rust wgpu-native, driven from Python through
``wgpu.backends.wgpu_native``) — Metal on macOS, Vulkan on Linux, DX12/Vulkan
on Windows, chosen automatically (``--backend`` / ``WGPU_BACKEND`` override).
No OpenGL anywhere: nothing here depends on Apple's deprecated GL stack.

The event loop is reactive, it draws one frame when a callback notify an event.
No input means no wake-ups, no frames, no CPU and no GPU work —
the window keeps its last presented frame until the user
touches something.

-> ``--poll`` passed as argument on launch runs the naive always-redraw loop
this constitute a solid universal graphical user interface for any linear
algebra study on a laptop battery driven.

this main.py leverage a function ( defined just before the main() ) that draw 1000000 particles,
it's an ia generated function that leverage numpy quite well, (numpy is a whole
topic by itself), but basically it's an array of particles on X an Y with a Z
modulated by a factorized Sinus function. it's pretty nice. ( the color variation is
defined in the function body itself from rgb value )
"""

from __future__ import annotations

import argparse
import functools
import math
import signal
import sys
import time
from collections.abc import Sequence

import glfw
import loop3d
import numpy as np
import reactive3d
from camera3d import OrbitCamera
from gpu3d import GPU
from macos_gestures3d import install as install_macos_gestures
from macos_gestures3d import scroll_is_precise
from renderer3d import GridArrowRenderer
from renderer_aa3d import AAGridArrowRenderer
from settings3d import (
    ARROW_COLOR,
    BACKGROUND,
    CLOCK_COLOR,
    DEFAULT_WAIT_TIMEOUT,
    HALF,
    IDLE_REFRESH_HZ,
    PAN_PIXELS_PER_SCROLL,
    PINCH_ZOOM_STEP,
    SCENE_Z,
    SCROLL_PAN_SIGN_X,
    SCROLL_PAN_SIGN_Y,
    TITLE,
    WHEEL_PAN_PIXELS,
)
from vec2 import Vec2
from vec3 import Vec3
from window3d import close_window, create_window

yellow = "\x1b[033m"
purple = "\x1b[035m"
rst = "\x1b[0m"

ONE_HOUR_OFFSET = math.tau / 12
ONE_SECOND_MINUTE_OFFSET = math.tau / 60
OCLOCK = math.pi * 0.5

CLOCK_RADIUS = 1.8
CLOCK_Z = SCENE_Z
READOUT_COLOR = (0.85, 0.92, 1.0)
SCREEN_HUD_COLOR = (1.0, 0.85, 0.40)
HAND_COLORS = ((1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, 0.0, 1.0))


def make_demo_picture(width: int = 320, height: int = 200) -> np.ndarray:
    """A procedural test card, so ``draw_image`` needs no picture file.

    It is built once at import time and drawn every frame from the same array,
    which is what the texture cache keys on: the image reaches the GPU once.
    """
    u = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    v = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    checker = (np.floor(u * 8.0) + np.floor(v * 5.0)) % 2.0

    red = 0.15 + 0.75 * u
    green = 0.20 + 0.70 * v
    blue = 0.85 - 0.45 * u
    channels = np.broadcast_arrays(red, green, blue)
    rgb = np.clip(np.stack(channels, axis=2) + 0.10 * checker[..., None], 0.0, 1.0)

    # A bright border and a dark centre disc make the orientation obvious.
    rgb[(u < 0.03) | (u > 0.97) | (v < 0.04) | (v > 0.96)] = (0.95, 0.95, 0.85)
    disc = (u - 0.5) ** 2 + (v - 0.5) ** 2 < 0.035
    rgb[np.broadcast_to(disc, rgb.shape[:2])] = (0.08, 0.08, 0.10)
    return (rgb * 255.0).astype(np.uint8)


DEMO_PICTURE = make_demo_picture()


def analog_display(
    system: GridArrowRenderer | AAGridArrowRenderer,
    seconds: float,
) -> None:
    """Draw an analog time watch lying flat on the floor (the xy plane)."""
    seconds_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * seconds)
    number_of_minutes = seconds / 60
    minutes_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * number_of_minutes)
    number_of_hours = number_of_minutes / 60
    hours_angle = OCLOCK - (ONE_HOUR_OFFSET * number_of_hours)

    def hand(angle: float, length: float, color: tuple[float, float, float]) -> None:
        # Vec2 for the dial direction, promoted to Vec3 for the 3D API.
        direction = Vec2(math.cos(angle), math.sin(angle)) * length
        system.draw_arrow(
            Vec3(0.0, 0.0, CLOCK_Z),
            Vec3(direction.x, direction.y, CLOCK_Z),
            color,
        )

    hand(seconds_angle, 1.7, HAND_COLORS[0])
    hand(minutes_angle, 1.25, HAND_COLORS[1])
    hand(hours_angle, 1.0, HAND_COLORS[2])


def clamp_ground(
    point: tuple[float, float, float],
    camera: OrbitCamera,
) -> tuple[float, float, float]:
    """Keep the cursor tip near the visible part of the floor.

    The ray/plane hit can run far away when the camera is almost level with the
    floor. The 2D app clamps to its visible rectangle; the 3D analogue is a
    generous square around the pivot, which follows panning and zooms in with
    the camera instead of pinning the tip to a fixed world corner.
    """
    limit = max(HALF, camera.distance)
    x = float(np.clip(point[0], camera.target[0] - limit, camera.target[0] + limit))
    y = float(np.clip(point[1], camera.target[1] - limit, camera.target[1] + limit))
    return x, y, CLOCK_Z


def draw_clock(
    system: GridArrowRenderer | AAGridArrowRenderer,
    aa: AAGridArrowRenderer | None,
    elapsed_second: float,
) -> None:
    """The 2D clock scene, reinterpreted on the xy floor with a z axis normal."""
    # plain line-list circle: plain_renderer.draw_circle((0, 0, 0), 1.8, ...)
    system.draw_circle(
        Vec3(0.0, 0.0, CLOCK_Z),
        CLOCK_RADIUS,
        CLOCK_COLOR,
        Vec3(0.0, 0.0, 1.0),
    )

    for i in range(1, 13):
        angle = ONE_HOUR_OFFSET * i
        dx, dy = math.cos(angle) * 1.3, math.sin(angle) * 1.3
        # plain: plain_renderer.draw_line((dx, dy, CLOCK_Z), (dx * 1.3, dy * 1.3, CLOCK_Z), CLOCK_COLOR, 0.05)
        system.draw_line(
            Vec3(dx, dy, CLOCK_Z),
            Vec3(dx * 1.3, dy * 1.3, CLOCK_Z),
            CLOCK_COLOR,
            0.05,
        )

        if aa is None:
            continue
        # Text is AA-only (the plain line-list renderer has no stroke font).
        # These labels lie flat on the floor: right = +x, up = +y reads from above.
        label = str(i)
        label_size = 0.30
        label_x = math.cos(OCLOCK - ONE_HOUR_OFFSET * i) * 2.05
        label_y = math.sin(OCLOCK - ONE_HOUR_OFFSET * i) * 2.05
        label_half = aa.measure_text(label, label_size) * 0.5
        aa.draw_text(
            Vec3(label_x - label_half, label_y - label_half, CLOCK_Z),
            label,
            label_size,
            CLOCK_COLOR,
        )

    analog_display(system, elapsed_second)


def draw_cursor_arrow(
    system: GridArrowRenderer | AAGridArrowRenderer,
    start: Vec3 | tuple[float, float, float],
    tip: Vec3 | tuple[float, float, float],
) -> None:
    # plain line-list arrow: plain_renderer.draw_arrow(start, tip, ARROW_COLOR)
    system.draw_arrow(start, tip, ARROW_COLOR)


def draw_vector_array(
    system: GridArrowRenderer | AAGridArrowRenderer,
    tip: tuple[float, float, float],
) -> None:
    """A field of small floor arrows pointing toward the cursor tip (the 2D array)."""
    step = 0.3
    array_x, array_y = 10, 5
    anchor_x, anchor_y = 1.2, -2.6
    mid_x = anchor_x + (array_x * 0.45 * step)
    mid_y = anchor_y - (array_y * 0.4 * step)

    dx, dy = tip[0] - mid_x, tip[1] - mid_y
    length = math.hypot(dx, dy)
    normalized_dir = (dx / length, dy / length) if length > 1e-9 else (0.0, 0.0)

    for i in range(array_x):
        for j in range(array_y):
            from_pt = Vec3(anchor_x + (i * step), anchor_y + (-j * step), CLOCK_Z)
            to_pt = Vec3(
                from_pt.x + normalized_dir[0],
                from_pt.y + normalized_dir[1],
                CLOCK_Z,
            )
            # plain: plain_renderer.draw_arrow(from_pt, to_pt)
            system.draw_arrow(from_pt, to_pt)


def floor_text_axes(
    camera: OrbitCamera,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Text axes on the floor that line up with the current screen.

    Text lying on the xy plane is foreshortened by perspective, but it should
    still read left-to-right from wherever the camera is. Project the camera's
    right/up vectors onto the floor and use them as the text's own axes; the
    label then rotates with the orbit instead of looking side-on.
    """
    right, up, _forward = camera.basis()
    rx, ry = float(right[0]), float(right[1])
    length = math.hypot(rx, ry)
    if length < 1e-9:
        rx, ry = 1.0, 0.0
    else:
        rx, ry = rx / length, ry / length
    ux, uy = -ry, rx  # 90 degrees counter-clockwise in the xy plane
    if ux * float(up[0]) + uy * float(up[1]) < 0.0:
        ux, uy = -ux, -uy
    return (rx, ry, 0.0), (ux, uy, 0.0)


def draw_cursor_readout(
    aa: AAGridArrowRenderer | None,
    camera: OrbitCamera,
    tip: tuple[float, float, float],
    cursor_x: float,
    cursor_y: float,
) -> None:
    """Write the cursor's floor hit and pixel position on the xy plane.

    This is the 3D analogue of the 2D HUD readout: instead of pinning text to
    the bottom-left of the window, the label lies flat on the floor near the
    arrow tip. Its axes are aligned with the current screen (still inside the
    xy plane), so the label stays readable as the camera orbits.
    """
    if aa is None:
        return
    label = f"cursor ({tip[0]:+.2f}, {tip[1]:+.2f})   px ({cursor_x:.0f}, {cursor_y:.0f})"
    size = 0.22
    right, up = floor_text_axes(camera)
    width = aa.measure_text(label, size)
    # Centre the label over the tip and lift it in screen-up, so it clears the
    # arrow head and grows away from the cursor.
    base_x = tip[0] - right[0] * width * 0.5 + up[0] * 0.40
    base_y = tip[1] - right[1] * width * 0.5 + up[1] * 0.40
    aa.draw_text(
        (base_x, base_y, CLOCK_Z + 0.01),
        label,
        size,
        READOUT_COLOR,
        right=right,
        up=up,
    )


def draw_2D_rectangle(
    system: GridArrowRenderer | AAGridArrowRenderer,
    corner_a: Sequence[float],
    corner_b: Sequence[float],
    color: tuple[float, float, float],
) -> None:
    """
    Draw a 2D rectangle from base component lines.
    (a small increment on z avoid z fight with the 2d grid)
    """
    # first define the rectangle corners.
    a = (corner_a[0], corner_a[1], 0.04)
    b = (corner_b[0], corner_a[1], 0.04)
    c = (corner_b[0], corner_b[1], 0.04)
    d = (corner_a[0], corner_b[1], 0.04)
    # draw the 4 lines.
    system.draw_line(a, b, color)
    system.draw_line(b, c, color)
    system.draw_line(c, d, color)
    system.draw_line(d, a, color)


@functools.lru_cache(maxsize=4)
def wave_lattice(nx: int, ny: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Offsets of an ``nx`` by ``ny`` sheet centred on the origin, in xy.

    Cached because the lattice is the same every frame: at 1 M particles the
    meshgrid alone is 8 MB of float32, and both wave variants would otherwise
    build it twice per frame. The arrays are read-only by convention.
    """
    columns = (np.arange(nx, dtype=np.float32) - (nx - 1) * 0.5) * spacing
    rows = (np.arange(ny, dtype=np.float32) - (ny - 1) * 0.5) * spacing
    return np.meshgrid(columns, rows)  # both are (ny, nx)


def particle_wave_points(
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The ``nx`` by ``ny`` travelling-wave sheet as ``(points, colours)``.

    Pure NumPy, no GPU: two ``(N, 3)`` float32 arrays that are already flat, so a
    caller hands them straight to a renderer. The geometry is described in
    ``draw_particle_wave``; this function is what both variants share.
    """
    dx, dy = wave_lattice(nx, ny, spacing)

    phase = math.tau * (dx + dy) / wavelength - elapsed * speed
    wave = np.sin(phase)

    points = np.stack(
        (anchor[0] + dx, anchor[1] + dy, anchor[2] + amplitude * wave),
        axis=-1,
    )

    # 0 = trough (cyan), 1 = crest (warm): the colour is the wave's height.
    height = 0.5 + 0.5 * wave
    colors = np.stack(
        (0.15 + 0.85 * height, 0.55 + 0.35 * height, 0.95 - 0.75 * height),
        axis=-1,
    )
    return points.reshape(-1, 3), colors.reshape(-1, 3)


def draw_particle_wave(
    system: GridArrowRenderer | AAGridArrowRenderer,
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    size: float = 0.03,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 2.0,
) -> None:
    """Draw an ``nx`` by ``ny`` particle field around ``anchor`` in a travelling wave.

    The sheet is centred on ``anchor`` in the xy plane, ``spacing`` apart, and
    every particle is lifted along z by ``amplitude * sin(phase)``. The phase
    advances diagonally across the grid and travels with ``elapsed`` (seconds),
    so the crests slide towards +x / +y::

        phase = 2*pi * ((dx + dy) / wavelength) - elapsed * speed
        z     = anchor_z + amplitude * sin(phase)

    Colour follows the wave — troughs cool cyan, crests warm yellow — which
    makes the height readable without a second pass.

    ``nx * ny`` particles are one ``draw_points`` call: one vertex per particle,
    one upload, one draw. Nothing is called per particle, so the field stays
    cheap (a 32 x 16 sheet is 512 particles, a fraction of a millisecond here).

    Draw order is the array order. For correct layering under the AA renderer
    (its sprites ignore the depth buffer) use ``draw_particle_wave_sorted``,
    which is the same sheet with a back-to-front permutation.
    """
    points, colors = particle_wave_points(
        anchor,
        nx,
        ny,
        elapsed,
        spacing=spacing,
        amplitude=amplitude,
        wavelength=wavelength,
        speed=speed,
    )
    system.draw_points(points, size, colors)


# One cached permutation is enough: it is valid until the camera pose or the
# wave advances past its tolerance, and a new one replaces it rather than
# accumulating 8 MB per frame. Stored as (lattice key, pose, elapsed, order).
_wave_order_state: tuple[tuple, np.ndarray, float, np.ndarray] | None = None


def _wave_order(
    camera: OrbitCamera,
    points: np.ndarray,
    lattice: tuple,
    elapsed: float,
    pose_step: float,
    time_step: float,
) -> np.ndarray:
    """Back-to-front permutation of ``points``, reused while nothing moved.

    The permutation is computed on the sheet that is actually drawn, so it is
    exact rather than approximate: the wave lifts particles by up to
    ``amplitude``, which is as large as the lattice's own depth spacing (~48% of
    neighbouring pairs would swap if the order came from the flat sheet).

    Reuse is a hysteresis on both inputs, not a bucket: the cached order is
    returned while no pose component (eye in world units, forward as a unit
    vector) moved more than ``pose_step`` **and** the wave advanced by less than
    ``time_step`` seconds. Both default to ``0``, which sorts every frame — the
    exact path, and the one to use unless ``N`` is large.
    """
    global _wave_order_state

    pose = np.concatenate((camera.eye(), camera.forward()))
    if _wave_order_state is not None:
        cached_lattice, cached_pose, cached_elapsed, cached_order = _wave_order_state
        if (
            cached_lattice == lattice
            and float(np.abs(pose - cached_pose).max()) <= pose_step
            and abs(elapsed - cached_elapsed) <= time_step
        ):
            return cached_order

    order = camera.depth_order(points)
    _wave_order_state = (lattice, pose, float(elapsed), order)
    return order


def draw_particle_wave_sorted(
    system: GridArrowRenderer | AAGridArrowRenderer,
    camera: OrbitCamera,
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    size: float = 0.03,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 2.0,
    pose_step: float = 0.0,
    time_step: float = 0.0,
) -> None:
    """``draw_particle_wave``, drawn back-to-front for ``camera``.

    Same sheet, same single draw call, but the particles are ordered far to near
    on **view depth** ``dot(p - eye, forward)``, so the ones closest to the
    camera are painted last. That matters where the depth buffer cannot help:
    the AA renderer's sprites are screen-space quads pinned to ``NDC z = 0``, and
    transparent particles need painter's order everywhere. With ``renderer3d``'s
    opaque squares the depth test already resolves them and this only costs the
    sort.

    Cost is one ``O(N log N)`` sort per frame (one ``draw_points`` upload and
    draw call around it): ~1.6 ms at 20 k particles, ~18 ms at 200 k, ~100 ms at
    1 M here. ``pose_step`` / ``time_step`` reuse the previous permutation while
    the camera and the wave stay within those tolerances, which is what keeps a
    large sheet affordable — but a *stale* order is an approximation, so leave
    both at ``0`` unless you need the speed.
    """
    points, colors = particle_wave_points(
        anchor,
        nx,
        ny,
        elapsed,
        spacing=spacing,
        amplitude=amplitude,
        wavelength=wavelength,
        speed=speed,
    )
    lattice = (nx, ny, spacing, anchor)
    order = _wave_order(camera, points, lattice, elapsed, pose_step, time_step)
    system.draw_points_sorted(points, size, colors, order=order)


# ---------------------------------------------------------------------------
# alpha test: the same sheet, with a per-particle life-time coverage channel
# ---------------------------------------------------------------------------
def particle_lifetime_alpha(
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    wavelength: float = 1.6,
    fade_rate: float = 0.35,
    power: float = 1.0,
) -> np.ndarray:
    """Per-particle coverage in ``0..1``: a life-time pulse crossing the sheet.

    Every particle owns a private clock, offset by its lattice position, so the
    sheet is not one global fade: a diagonal front of *just born* particles
    (alpha 1) sweeps towards +x / +y while the ones behind it have already given
    their alpha back::

        life  = frac(elapsed * fade_rate + (dx + dy) / wavelength)
        alpha = (1 - life) ** power

    ``power`` > 1 holds the particle visible longer and then drops it faster;
    ``power`` < 1 fades early and leaves a long tail of near-invisible dots —
    which is exactly what makes a wrong blend or a missing alpha column obvious.

    Pure NumPy over the cached lattice: two whole-array passes, no per-particle
    Python loop, no allocation per particle. Returns a flat ``(N,)`` float32
    column, ready to sit next to the ``(N, 3)`` colours from
    ``particle_wave_points``.
    """
    dx, dy = wave_lattice(nx, ny, spacing)
    life = (elapsed * fade_rate + (dx + dy) / wavelength) % 1.0
    alpha = 1.0 - life
    if power != 1.0:
        alpha = alpha**power
    return alpha.reshape(-1).astype(np.float32, copy=False)


def particle_lifetime_points(
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 0.35,
    fade_rate: float = 0.35,
    power: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """The wave sheet as ``(points, (N, 4) colours)``: same geometry, plus alpha.

    Geometry and rgb come from ``particle_wave_points`` unchanged, so the two
    sheets can be compared side by side; only the fourth channel is new. That
    column is what ``draw_points_rgba`` multiplies into the sprite coverage.
    """
    points, colors = particle_wave_points(
        anchor,
        nx,
        ny,
        elapsed,
        spacing=spacing,
        amplitude=amplitude,
        wavelength=wavelength,
        speed=speed,
    )
    alpha = particle_lifetime_alpha(
        nx,
        ny,
        elapsed,
        spacing=spacing,
        wavelength=wavelength,
        fade_rate=fade_rate,
        power=power,
    )
    return points, np.column_stack((colors, alpha))


def draw_particle_lifetime(
    system: GridArrowRenderer | AAGridArrowRenderer,
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    size: float = 0.06,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 0.35,
    fade_rate: float = 0.35,
    power: float = 1.0,
    depth_write: bool = True,
) -> None:
    """Alpha test: draw the wave sheet with a per-particle life-time fade.

    This is the exercise for the ``*_rgba`` particle path — the same call shape
    as ``draw_particle_wave``, but the colour array carries an alpha column, so
    each particle is born opaque and dies transparent as the front sweeps over
    it. What to check, in order:

    * **alpha column arrives**: the sheet shows a diagonal soft edge, not a hard
      line; particles at alpha 0 are gone, not black.
    * **both renderers agree**: the plain one shows fading squares, the AA one
      fading discs with the same coverage ramp (swap `system` in ``main``).
    * **the opaque path is untouched**: comment this call out and the wave call
      above renders exactly as before — the rgba path is a separate buffer,
      program and draw call.
    * **ordering**: every particle is blended in draw (array) order here. If the
      overlap looks flat, use ``draw_particle_lifetime_sorted``, which paints
      back-to-front like a stack of glass, or pass ``depth_write=False`` so the
      faded particles leave no depth footprints.

    Cost is one ``draw_points_rgba`` call: one upload of an 8-float-per-vertex
    batch and one blended draw, with the fade done by a multiply in the fragment
    shader. Nothing in this file loops per particle.
    """
    points, colors = particle_lifetime_points(
        anchor,
        nx,
        ny,
        elapsed,
        spacing=spacing,
        amplitude=amplitude,
        wavelength=wavelength,
        speed=speed,
        fade_rate=fade_rate,
        power=power,
    )
    system.draw_points_rgba(points, size, colors, depth_write=depth_write)


def draw_particle_lifetime_sorted(
    system: GridArrowRenderer | AAGridArrowRenderer,
    camera: OrbitCamera,
    anchor: tuple[float, float, float],
    nx: int,
    ny: int,
    elapsed: float,
    *,
    spacing: float = 0.22,
    size: float = 0.06,
    amplitude: float = 0.35,
    wavelength: float = 1.6,
    speed: float = 0.35,
    fade_rate: float = 0.35,
    power: float = 1.0,
    order: np.ndarray | None = None,
    depth_write: bool = True,
) -> None:
    """``draw_particle_lifetime`` painted back-to-front, the layering a fade needs.

    Transparent sprites cannot be resolved by the depth buffer, and the AA
    renderer pins them to ``NDC z = 0`` anyway, so **draw order is the
    layering**: the whole batch is sorted on view depth and the nearest particle
    composites last. Same upload, same single blended draw call as the unsorted
    variant, plus one ``O(N log N)`` sort — pass ``order`` to reuse a permutation
    computed when the camera moved instead of sorting every frame (this demo
    sheet is small, so it sorts each call).
    """
    points, colors = particle_lifetime_points(
        anchor,
        nx,
        ny,
        elapsed,
        spacing=spacing,
        amplitude=amplitude,
        wavelength=wavelength,
        speed=speed,
        fade_rate=fade_rate,
        power=power,
    )
    if order is None:
        order = camera.depth_order(points)
    system.draw_points_rgba_sorted(points, size, colors, order=order, depth_write=depth_write)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The loop's knobs, all of them about *when* a frame is drawn."""
    parser = argparse.ArgumentParser(
        prog="experimental_b",
        description="3D grid + arrows, drawn by a reactive (redraw-on-change) loop.",
        epilog="Reactive mode blocks in glfw.wait_events(): no input, no redraws, no CPU.",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="naive busy loop: poll + redraw every iteration (for comparison)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=IDLE_REFRESH_HZ,
        metavar="HZ",
        help="additionally redraw at this rate (0 = pure reactive)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=DEFAULT_WAIT_TIMEOUT,
        metavar="SECONDS",
        help="max block time in wait_events (0 = block forever: no wake-ups, but "
        "terminal Ctrl+C waits for the next window event)",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=("auto", "metal", "vulkan", "dx12", "gl"),
        help="wgpu backend (auto = Metal on macOS, Vulkan on Linux, DX12 on Windows; "
        "env WGPU_BACKEND works too)",
    )
    parser.add_argument(
        "--no-vsync",
        action="store_true",
        help="present without waiting for vsync (benchmarking --poll)",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="auto-exit after N seconds (for measuring idle CPU)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # The reactive switch: every callback that changes the picture calls mark(),
    # and the loop draws exactly one frame per mark.
    reactive = reactive3d.ReactiveState()

    # A bool flag designed as a stop value.
    running = True

    def on_sigint(_signum, _frame) -> None:
        nonlocal running
        running = False
        print("\n  Ctrl+C (signal) — exiting")

    # Set an interrupt linked to callback.
    signal.signal(signal.SIGINT, on_sigint)

    # Create a window object instance. A resize is a visible change but not an
    # input event, so it asks for one repaint.
    window = create_window(on_resize=lambda _win, _w, _h: reactive.mark())
    reactive.install(window)

    # The GPU: wgpu device + window surface (Metal / Vulkan / DX12, picked per
    # platform). Every renderer records into it; loop3d presents its frames.
    gpu = GPU(window, backend=args.backend, vsync=not args.no_vsync)
    gpu.clear_color = (*BACKGROUND, 1.0)
    glfw.set_window_title(window, f"{TITLE} — GPU anti-aliased, orbit + pan + zoom")

    # Two interchangeable renderers, same API
    # (one antialiased the other not(faster)):
    #   plain_renderer -> line-list, hard one-pixel edges (fastest poor quality)
    #   aa_renderer    -> screen-space capsules + per-fragment distance field (smooth slightly slower)
    # Both read the *same* OrbitCamera, so orbit/pan/zoom is one shared view

    camera = OrbitCamera()
    plain_renderer = GridArrowRenderer(gpu, camera)  # no AA faster
    aa_renderer = AAGridArrowRenderer(gpu, camera)

    # Swap this one line to render the whole scene with the aliased line-list twin.
    # (The plain calls stay commented next to every AA call, exactly like the 2D app.)
    system: GridArrowRenderer | AAGridArrowRenderer = aa_renderer
    # system: GridArrowRenderer | AAGridArrowRenderer = plain_renderer

    # Some function are aa only so an aa_render instance is
    # always needed.
    aa = aa_renderer

    # Vector standard 3D coordinates. The arrow tail: edit this point to move
    # the cursor arrow without moving the floor clock.
    origin = Vec3(0.0, 0.0, CLOCK_Z)

    # A 3d system involve depth management, which object display first from the
    # camera point of view, in this early system there is no geometry just raw
    # pixel color at a 3d point coordinate in camera space before screen projection
    # which mean that usually the camera forward matrix value act as depth
    # and use to define which pixel stroke to draw last on screen.

    # The scene is strokes, so depth writes are enough; the pipelines use a
    # less-equal depth test so coplanar strokes drawn later (the clock over the
    # grid) do not lose the depth fight. The clear colour was set on `gpu`.

    # --- input: orbit / pan / zoom --------------------------------------
    # The wheel callback accumulates; camera.handle() consumes it once per
    # frame, so several notches in one poll still produce one coherent zoom.
    # note: a callback, is a function that pass a specific runtime to be
    # executed in a specific api context at specific planned time trigger.
    def on_scroll(_window, dx: float, dy: float) -> None:
        # Any scroll changes the view (a pan or a zoom), so ask for a frame;
        # the camera consumes the accumulated delta inside draw_frame.
        reactive.mark()
        if scroll_is_precise():
            # Trackpad two-finger scroll: pan the pivot, like dragging the scene.
            camera.pan(
                dx * PAN_PIXELS_PER_SCROLL * SCROLL_PAN_SIGN_X,
                dy * PAN_PIXELS_PER_SCROLL * SCROLL_PAN_SIGN_Y,
            )
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if shift:
            # Shift + wheel: pan, for wheels without a horizontal axis.
            camera.pan(dx * WHEEL_PAN_PIXELS, dy * WHEEL_PAN_PIXELS)
            return

        # Wheel notch (or a pinch on platforms without the native bridge): zoom.
        camera.scroll_by(dy)

    # Pass the callback to the glfw api
    glfw.set_scroll_callback(window, on_scroll)

    def on_pinch(magnification: float) -> None:
        # macOS magnification is a per-event delta (fingers spread > 0), so the
        # exponential keeps the gesture smooth and always positive.
        camera.zoom_by(math.exp(magnification * PINCH_ZOOM_STEP))
        # Pinch bypasses GLFW entirely (see macos_gestures3d), so without this
        # mark a gesture would change the camera and never be drawn.
        reactive.mark()

    native_gestures = install_macos_gestures(window, on_pinch)

    # Plane the some keys runtime further execution.
    def on_key(_window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_R, glfw.KEY_HOME):
            camera.reset()
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_C and glfw.get_key(window, glfw.KEY_LEFT_CONTROL) == glfw.PRESS:
            # Ctrl+C only reaches the window while it has focus; with the
            # terminal focused it is the SIGINT handled in main.
            print("  Ctrl+C (in window) — exiting")
            glfw.set_window_should_close(window, True)
        else:
            return  # a key that changes nothing on screen: do not redraw
        reactive.mark()

    # Plane the listening for key input from previous callback.
    glfw.set_key_callback(window, on_key)

    # Readout state: the last printed tip, the previous mouse-button state, and
    # the last valid floor hit (used while the cursor ray misses the floor).
    last_printed: tuple[float, float, float] | None = None
    # parked together with the click readout in draw_frame: uncomment both, and
    # add them to the `nonlocal` line there, to bring the click state back
    # mouse_was_down = False
    last_tip: tuple[float, float, float] = (1.5, 1.5, CLOCK_Z)

    # print on terminal the control input keyboard programmed function
    sys.stdout.write("\x1b[2J")  # clear the screen.
    sys.stdout.write(
        f"{yellow}controls: left-drag = orbit, middle-drag = pan, "
        f"wheel/pinch = zoom, two-finger scroll = pan, "
        f"R/Home = reset, Esc = quit "
        f"(native gestures: {native_gestures}){rst}\n"
        f"{purple}gpu: {gpu.summary}{rst}\n"
    )
    sys.stdout.flush()

    # glfw.get_time() is the clock the (commented-out, time-based) scene below
    # reads. Pure change-detection cannot see it tick, so a scene that animates
    # wants `--refresh 60` (or IDLE_REFRESH_HZ) instead of relying on input.
    start = glfw.get_time()

    # ------------------------------------------------------------------
    # One frame: (clear) -> draw -> (present).
    # this is typically a callback to draw a frame: loop3d opens the frame
    # (the screen is cleared automatically) and presents it on the callback
    # function exit
    # the callback is linked to the call loop3d.run() in the last try and
    # catch block.
    # ------------------------------------------------------------------
    def draw_frame() -> None:
        nonlocal last_tip # mutating the upper scope variable.

        # Orbit (left), pan (middle), zoom (wheel). The camera remembers the
        # previous cursor so the deltas are exactly
        camera.handle(window)

        width, height = glfw.get_window_size(window)
        cursor_x, cursor_y = glfw.get_cursor_pos(window)

        # Cursor pixel -> ray -> floor point. Keep the last hit when the ray is
        # parallel to the floor (or points away from it), so the arrow does not
        # jump out of existence at a grazing angle.
        floor_hit = camera.floor_point(cursor_x, cursor_y, width, height)
        if floor_hit is not None:
            last_tip = (float(floor_hit[0]), float(floor_hit[1]), CLOCK_Z)
        tip = clamp_ground(last_tip, camera)

        # mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        # clicked = mouse_down and not mouse_was_down  # rising edge, not heldcyber display

        # plain line-list grid: plain_renderer.draw_grid()
        system.draw_grid()

        draw_2D_rectangle(system, (3, 2), (5, 3), (1, 1, 0))

        # A picture (texture) standing on the floor next to the clock:
        # `anchor` is the bottom-left corner, `right` / `up` are the picture's
        # own axes (only their direction matters) and `width` / `height` are its
        # size in world units along them. Both renderers implement the call.
        
        #    PNG  JPEG/JPG/JPE/JFIF  BMP/DIB  GIF/APNG  WebP  TIFF/TIF  TGA  PSD
        #    ICO/CUR/ICNS  PCX  PPM/PGM/PBM/PNM  XBM/XPM  SGI/RGB/RGBA  RAS  QOI
        #    DDS  JP2/J2K/JPX/J2C/JPC  AVIF/AVIFS  FITS  HDF5  GRIB  EPS/PS/PDF
        #    WMF/EMF  MPO  PCD  PXR  FLC/FLI  BUFR  MSP  PALM  ... 
        system.draw_image(
            anchor=(-3.7, -0.4, 0.05),
            source=DEMO_PICTURE,
            right=(1.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
            width=1.8,
            height=1.15,
        )

        pt1 = Vec2(7, 7)
        pt2 = Vec2(10, 10)
        draw_2D_rectangle(system, pt1, pt2, (1, 1, 0))

        elapsed_second = glfw.get_time() - start
        elapsed_second *= 100  # speed up the time by this scalar.
        draw_clock(system, aa, elapsed_second)

        # set it to True to sort the particle from the z depth.
        sort_particule_order = False

        spacing_v = 0.0022 # increase this value to sens the scale.
        # spacing_v = 0.22 # just like that.

        # sort the particle for the renderer.
        if not sort_particule_order:
            draw_particle_wave(
                system,
                anchor=(0.0, -3.5, 0.45),
                nx=1000,
                ny=1000,
                elapsed=glfw.get_time() - start,
                spacing=spacing_v,
                size=0.03,
                amplitude=0.35,
            )
        else:
            draw_particle_wave_sorted(
                system,
                camera,
                anchor=(0.0, -3.5, 0.45),
                nx=1000,
                ny=1000,
                elapsed=glfw.get_time() - start,
                spacing=spacing_v,
                size=0.03,
                amplitude=0.35,
            )

        # --- alpha test: a small sheet with a per-particle life-time fade ----
        # draw_points_rgba is a separate method on a separate buffer/program, so
        # this call cannot slow the wave above down. Left of the clock, above
        # the wave plane so it stays visible: each particle is born opaque and
        # dies transparent as the fade front sweeps diagonally over the sheet.
        # Comment it out to compare, or swap in draw_particle_lifetime_sorted
        # (which also takes `camera`) to layer the faded particles back-to-front.
        draw_particle_lifetime(
            system,
            anchor=(-3.0, 3.0, 0.75),
            nx=24,
            ny=24,
            elapsed=glfw.get_time() - start,
            spacing=0.16,
            size=0.07,
            amplitude=0.18,
            wavelength=0.9,
            speed=0.25,
            fade_rate=0.30,
            power=1.6,
        )

        '''

        # The cursor arrow: the 3D analogue of the 2D arrow-to-cursor.
        draw_cursor_arrow(system, origin, tip)

        # The cursor's position on the floor, written on the xy plane itself.
        # This is the 3D analogue of the 2D HUD readout: it follows the cursor,
        # lies flat on the floor, and shows both the world hit and the pixel.
        draw_cursor_readout(aa, camera, tip, cursor_x, cursor_y)

        # A field of arrows pointing at the cursor, in the floor plane.
        draw_vector_array(system, tip)

        # This commented code block represent a gizmo
        # representing the 3d systems basis axis orientation
        # those are draw from the selected system variant.
        # (re-add AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR to the settings3d
        #  import list if you uncomment this block)
        """
        gizmo = (-3.7, 3.4, CLOCK_Z)
        for direction, color in (
            ((1.0, 0.0, 0.0), AXIS_X_COLOR),
            ((0.0, 1.0, 0.0), AXIS_Y_COLOR),
            ((0.0, 0.0, 1.0), AXIS_Z_COLOR),
        ):
            end = (
                gizmo[0] + direction[0] * 0.65,
                gizmo[1] + direction[1] * 0.65,
                gizmo[2] + direction[2] * 0.65,
            )
            system.draw_arrow(gizmo, end, color)
        """
        '''

        # The draw_screen_text is aa only.
        # Draw an anti-aliased text on screen space.
        if aa is not None:
            aa.draw_screen_text(
                16.0,
                24.0,
                f"cursor px ({cursor_x:.0f}, {cursor_y:.0f})   "
                f"world ({tip[0]:+.2f}, {tip[1]:+.2f})   "
                f"yaw {math.degrees(camera.yaw):+.0f}  "
                f"pitch {math.degrees(camera.pitch):+.0f}  "
                f"dist {camera.distance:.2f}",
                14.0,
                SCREEN_HUD_COLOR,
            )
        """
        # (uncommenting this block also needs `nonlocal last_printed,
        # mouse_was_down` next to `nonlocal last_tip` above)
        # ------------------------------------------------------------------
        # -> update the arrow on the terminal console only if a significant
        # move occurs, exactly like the 2D app.
        # ------------------------------------------------------------------
        moved = last_printed is None or math.dist(tip, last_printed) >= PRINT_STEP

        if clicked or (PRINT_STEP > 0 and moved):
            sys.stdout.write(
                f"{yellow}\r\x1b[4;0H\x1b[KClick "
                f"state:{purple}{clicked}{yellow}, "
                f"origin: ({origin[0]:+.2f}, {origin[1]:+.2f}, {origin[2]:+.2f}) -> "
                f"tip: ({tip[0]:+.2f}, {tip[1]:+.2f}, {tip[2]:+.2f})   "
                f"yaw: {purple}{math.degrees(camera.yaw):+.0f}°{yellow}  "
                f"pitch: {purple}{math.degrees(camera.pitch):+.0f}°{yellow}  "
                f"distance: {purple}{camera.distance:.2f}{yellow}  "
                f"target: ({camera.target[0]:+.2f}, {camera.target[1]:+.2f}, "
                f"{camera.target[2]:+.2f})   "
                f"cursor: px ({cursor_x:.0f}, {cursor_y:.0f}){rst}"
            )
            sys.stdout.flush()
            last_printed = tip

        mouse_was_down = mouse_down
        """

    # ------------------------------------------------------------------
    # Hand the frame to the reactive loop (loop3d.run): block in wait_events
    # until a callback marks the state, draw one frame, present, sleep again.
    # --poll runs the naive always-redraw twin for comparison.
    # ------------------------------------------------------------------
    deadline = time.perf_counter() + args.seconds if args.seconds > 0.0 else None

    def should_stop() -> bool:
        if not running:
            return True
        return deadline is not None and time.perf_counter() >= deadline

    # A deadline can only expire if the loop wakes on its own, so a timed run
    # keeps a small block cap; an interactive run blocks until real input.
    wait = args.wait if args.wait > 0.0 or deadline is None else 0.05

    try:
        stats = loop3d.run(
            gpu,
            reactive,
            draw_frame,
            poll=args.poll,
            refresh=args.refresh,
            wait=wait,
            should_stop=should_stop,
        )
    finally:
        aa_renderer.close()
        plain_renderer.close()
        gpu.close()
        close_window(window)

    print(stats.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
