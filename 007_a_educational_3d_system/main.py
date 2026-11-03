"""Experimental_b — the 2D drawing system with a step06 orbit camera.

Run from ``experimental/``:

    .venv/bin/python experimental_b/main.py

The point of the exercise is the API, not the scene. ``draw_arrow``,
``draw_line`` and ``draw_circle`` keep the same signatures as the 2D app; the
only geometric addition is that a 3D circle has to say which plane it lives in,
so ``draw_circle`` grew an ``axis`` argument (the circle's normal). The grid is
still automatic and camera-aware, and both renderers still share one drawing
API so any call can be switched back to plain GL_LINES by uncommenting its twin.

Navigation is step06's:

* left drag  -> orbit the pivot
* middle drag -> pan the pivot across the screen
* wheel      -> zoom by moving closer / farther
* R / Home   -> reset the view
* Esc        -> quit

The world is z-up: the floor is the xy plane, x is red, y is green and z is
blue. The cursor arrow is the 3D analogue of the 2D arrow-to-cursor: the cursor
pixel becomes a ray, the ray meets the floor plane ``z = 0``, and the arrow
points there. A text readout follows the cursor on that same xy plane, while a
screen-space HUD in the top-left shows the same values at a fixed pixel size and
its own colour.

The first stand-alone python script visualization tool. Simply create an
instance of an application that establishes a 3D context and allows us to draw
3D vector arrows in the colours of our choice.
"""

from __future__ import annotations

import math
import sys

import glfw
import numpy as np
from camera3d import OrbitCamera
from macos_gestures3d import install as install_macos_gestures
from macos_gestures3d import scroll_is_precise
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_LEQUAL,
    glClear,
    glClearColor,
    glDepthFunc,
    glEnable,
)
from renderer3d import GridArrowRenderer
from renderer_aa3d import AAGridArrowRenderer
from settings3d import (
    ARROW_COLOR,
    BACKGROUND,
    CLOCK_COLOR,
    HALF,
    PAN_PIXELS_PER_SCROLL,
    PINCH_ZOOM_STEP,
    PRINT_STEP,
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
    # plain GL_LINES circle: plain_renderer.draw_circle((0, 0, 0), 1.8, ...)
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
        # Text is AA-only (the plain GL_LINES renderer has no stroke font).
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
    # plain GL_LINES arrow: plain_renderer.draw_arrow(start, tip, ARROW_COLOR)
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


def main() -> int:

    # Create a window object instance.
    window = create_window()
    glfw.set_window_title(window, f"{TITLE} — GPU anti-aliased, orbit + pan + zoom")

    # Two interchangeable renderers, same API
    # (one antialiased the other not(faster)):
    #   plain_renderer -> GL_LINES, hard one-pixel edges (fastest poor quality)
    #   aa_renderer    -> screen-space capsules + per-fragment distance field (smooth slightly slower)
    # Both read the *same* OrbitCamera, so orbit/pan/zoom is one shared view

    camera = OrbitCamera()
    plain_renderer = GridArrowRenderer(window, camera)  # no AA faster
    aa_renderer = AAGridArrowRenderer(window, camera)

    # Swap this one line to render the whole scene with the aliased GL_LINES twin.
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

    # The scene is strokes, so depth writes are enough; GL_LEQUAL keeps coplanar
    # strokes drawn later (the clock over the grid) from losing the depth fight.
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glClearColor(*BACKGROUND, 1.0)

    # --- input: orbit / pan / zoom --------------------------------------
    # The wheel callback accumulates; camera.handle() consumes it once per
    # frame, so several notches in one poll still produce one coherent zoom.
    # note: a callback, is a function that pass a specific runtime to be
    # executed in a specific api context at specific planned time trigger.
    def on_scroll(_window, dx: float, dy: float) -> None:
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

    native_gestures = install_macos_gestures(window, on_pinch)

    # Plane the some keys runtime further execution.
    def on_key(_window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action != glfw.PRESS:
            return
        if key in (glfw.KEY_R, glfw.KEY_HOME):
            camera.reset()
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    # Plane the listening for key input from previous callback.
    glfw.set_key_callback(window, on_key)

    # Readout state: the last printed tip, the previous mouse-button state, and
    # the last valid floor hit (used while the cursor ray misses the floor).
    last_printed: tuple[float, float, float] | None = None
    mouse_was_down = False
    last_tip: tuple[float, float, float] = (1.5, 1.5, CLOCK_Z)

    # print on terminal the control input keyboard programmed function
    sys.stdout.write("\x1b[2J")  # clear the screen.
    sys.stdout.write(
        f"{yellow}controls: left-drag = orbit, middle-drag = pan, "
        f"wheel/pinch = zoom, two-finger scroll = pan, "
        f"R/Home = reset, Esc = quit "
        f"(native gestures: {native_gestures}){rst}\n"
    )
    sys.stdout.flush()

    start = glfw.get_time()

    # ------------------------------------------------------------------
    # Events loop shape: events -> clear -> draw -> present — the same shape as
    # experimental/main.py, only the camera and the draw calls are 3D.
    # ------------------------------------------------------------------
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Orbit (left), pan (middle), zoom (wheel).
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

        mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        clicked = mouse_down and not mouse_was_down  # rising edge, not held

        glClear(int(GL_COLOR_BUFFER_BIT) | int(GL_DEPTH_BUFFER_BIT))

        # plain GL_LINES grid: plain_renderer.draw_grid()
        system.draw_grid()

        elapsed_second = glfw.get_time() - start
        elapsed_second *= 100  # speed up the time by this scalar.
        draw_clock(system, aa, elapsed_second)

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

        glfw.swap_buffers(window)
    # ------------------------------------------------------------------

    aa_renderer.close()
    plain_renderer.close()
    close_window(window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
