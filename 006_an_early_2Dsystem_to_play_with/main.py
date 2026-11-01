from __future__ import annotations

import math
import sys

import glfw
from geometry import Camera
from macos_gestures import install as install_macos_gestures
from macos_gestures import scroll_is_precise
from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor
from renderer import GridArrowRenderer
from renderer_aa import AAGridArrowRenderer
from settings import (
    ARROW_COLOR,
    BACKGROUND,
    HALF,
    PAN_PIXELS_PER_SCROLL,
    PINCH_ZOOM_STEP,
    PRINT_STEP,
    SCROLL_PAN_SIGN_X,
    SCROLL_PAN_SIGN_Y,
    TITLE,
    WHEEL_PAN_PIXELS,
    WHEEL_ZOOM_STEP,
)
from window import close_window, create_window, cursor_world

"""
    The first stand alone python script visualization tool.
    Simply create an instance of an application that establish
    a 2d context and allowing us to draw 2d vectors arrow in the colors
    of our choices.
    
    Objective:
         
         -  leverage a builtin small python app that establish a 2d system to play 
            with the previous 2d examples with the Vec2 object and it's
            transformation functions, in colors with space & time.


         ( the abstraction embedded in the api there is fine and not relevant at this
         early stage )
         note: we should already know how to create a 2d system in a 
         wished orientation from the geometry node series )
         
         *** 
         the idea is to provide a scripted and standalone approach to play
         with matrix transformations. 
         
         - blender geometry node is not optimal to 
         play with matrix multiplications at least the python option ( the
         scripted one ) provide the last degree of freedom needed. 
         
         info: ( the inner GPU acceleration already provide simple shaders 
         quickly compiled from this simple 2d python application ).

    Navigation (geometry.Camera):
         - wheel / trackpad pinch  -> zoom, anchored to the pointer
         - middle-button drag      -> pan (content follows the pointer)
         - trackpad two-finger pan -> pan (macOS native bridge, see macos_gestures.py)
         - R or Home               -> reset the view
"""

yellow = "\x1b[033m"
purple = "\x1b[035m"
rst = "\x1b[0m"

ONE_HOUR_OFFSET = math.tau / 12
ONE_SECOND_MINUTE_OFFSET = math.tau / 60
OCLOCK = math.pi * 0.5


def analog_display(
    system: GridArrowRenderer | AAGridArrowRenderer,
    seconds: float,
) -> None:
    """
    Simply draw an analog time watch representation.
    """
    seconds_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * seconds)
    number_of_minutes = seconds / 60
    minutes_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * number_of_minutes)
    number_of_hours = number_of_minutes / 60
    hours_angle = OCLOCK - (ONE_HOUR_OFFSET * number_of_hours)

    sx, sy = (math.cos(seconds_angle), math.sin(seconds_angle))
    mx, my = (math.cos(minutes_angle), math.sin(minutes_angle))
    hx, hy = (math.cos(hours_angle), math.sin(hours_angle))

    system.draw_arrow((0, 0), (sx * 1.7, sy * 1.7), (1, 0, 0))
    system.draw_arrow((0, 0), (mx * 1.25, my * 1.25), (1, 1, 0))
    system.draw_arrow((0, 0), (hx, hy), (1, 0, 1))


def digits_for_zoom(zoom: float) -> int:
    """Decimals to show for world coordinates at the current zoom (2 at 1:1)."""
    return max(2, min(6, int(math.floor(math.log10(max(zoom, 1.0)))) + 2))


def main() -> int:

    # Create a window object instance.
    window = create_window()

    # Create 2d System app instance.
    # ( not the subject there so the Api abstraction is fine for now)
    # ( we should know already how to create a 2d system in a wished orientation )
    #
    # Two interchangeable renderers, same API:
    #   plain_renderer -> GL_LINES, hard edges (the original look)
    #   aa_renderer    -> stroke quads + per-fragment distance field (GPU AA)
    # The loop below draws with aa_renderer; every plain call is kept right next
    # to it as a comment, so switching one shape back is uncommenting one line.
    #
    # Both renderers read the *same* Camera, so pan/zoom is one shared view
    # transform (a vec4 uniform in each vertex shader) rather than two states.
    camera = Camera()
    plain_renderer = GridArrowRenderer(window, camera)
    aa_renderer = AAGridArrowRenderer(window, camera)
    glfw.set_window_title(window, f"{TITLE} — GPU anti-aliased, pan + zoom")

    # Vector standard 2d coordinates.
    origin = (0.0, 0.0)  # arrow tail: edit this coordinate to move the arrow
    arrow_color = ARROW_COLOR  # edit to recolour the arrow, e.g. (1.0, 0.3, 0.1)

    # Readout state: the last tip printed, and the previous mouse-button state.
    last_printed: tuple[float, float] | None = None
    mouse_was_down = False

    # Middle-button panning state: previous button state and previous position.
    middle_was_down = False
    last_cursor = glfw.get_cursor_pos(window)

    # --- input: zoom & pan ----------------------------------------------
    # The scroll callback is the only trackpad/mouse-wheel channel GLFW offers.
    # macos_gestures tells us whether this event came from a trackpad (precise,
    # used for panning) or a wheel (used for zoom), and adds the pinch gesture
    # GLFW never forwards.
    def on_scroll(_window, dx: float, dy: float) -> None:
        width, height = glfw.get_window_size(window)
        if scroll_is_precise():
            # Trackpad two-finger scroll: pan the content under the fingers.
            camera.pan_pixels(
                dx * PAN_PIXELS_PER_SCROLL * SCROLL_PAN_SIGN_X,
                dy * PAN_PIXELS_PER_SCROLL * SCROLL_PAN_SIGN_Y,
                width,
                height,
            )
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if shift:
            # Shift + wheel: pan, for wheels without a horizontal axis.
            camera.pan_pixels(
                dx * WHEEL_PAN_PIXELS,
                dy * WHEEL_PAN_PIXELS,
                width,
                height,
            )
            return

        # Wheel notch (or a pinch on platforms without the native bridge): zoom
        # around the pointer so the point under the cursor does not move.
        px, py = glfw.get_cursor_pos(window)
        camera.zoom_at(px, py, math.exp(dy * WHEEL_ZOOM_STEP), width, height)

    glfw.set_scroll_callback(window, on_scroll)

    def on_pinch(magnification: float) -> None:
        # macOS magnification is a per-event delta (fingers spread > 0), so the
        # exponential keeps the gesture smooth and always positive.
        width, height = glfw.get_window_size(window)
        px, py = glfw.get_cursor_pos(window)
        camera.zoom_at(px, py, math.exp(magnification * PINCH_ZOOM_STEP), width, height)

    native_gestures = install_macos_gestures(window, on_pinch)

    def on_key(_window, key: int, _scancode: int, action: int, _mods: int) -> None:
        if action == glfw.PRESS and key in (glfw.KEY_R, glfw.KEY_HOME):
            camera.reset()

    glfw.set_key_callback(window, on_key)

    # *BACKGROUND mean unpacking the tuple value
    glClearColor(*BACKGROUND, 1.0)
    start = glfw.get_time()

    sys.stdout.write("\x1b[2J")  # clear the screen.
    sys.stdout.write(
        f"{yellow}controls: wheel/pinch = zoom at pointer, middle-drag = pan, "
        f"two-finger scroll = pan, R = reset view "
        f"(native gestures: {native_gestures}){rst}\n"
    )
    sys.stdout.flush()
    # ------------------------------------------------------------------
    # Events loop shape: events -> clear -> draw -> present.
    # Everything between clear and swap can be customized and rendered at
    # each frames.
    # ------------------------------------------------------------------
    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Current viewport (window points: the unit GLFW cursor coordinates use).
        # The camera only needs the aspect ratio here, which is DPI independent.
        width, height = glfw.get_window_size(window)
        cursor_x, cursor_y = glfw.get_cursor_pos(window)

        # Middle-button drag: pan by the pointer delta, in pixels, so the content
        # stays glued to the cursor at any zoom level.
        middle_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        if middle_down and middle_was_down:
            camera.pan_pixels(cursor_x - last_cursor[0], cursor_y - last_cursor[1], width, height)
        last_cursor = (cursor_x, cursor_y)
        middle_was_down = middle_down

        raw = cursor_world(window, camera)  # unclamped cursor coordinate
        # Clamp to the *visible* world rectangle, not to the old +/-HALF grid:
        # the domain moves with the camera, so a fixed clamp would make the tip
        # stick to a corner while the rest of the scene pans away.
        tip = camera.clamp_to_view(raw, width, height)
        mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        clicked = mouse_down and not mouse_was_down  # rising edge, not held

        glClear(GL_COLOR_BUFFER_BIT)  # the technical location of the buffer bit in memory
        # plain GL_LINES grid: plain_renderer.draw_grid()
        aa_renderer.draw_grid()

        # draw arrow from origin to cursor position
        # plain GL_LINES arrow: plain_renderer.draw_arrow(origin, tip, arrow_color)
        # aa_renderer.draw_arrow(origin, tip, arrow_color)

        # draw a circle.
        # plain GL_LINES circle: plain_renderer.draw_circle((0, 0), 1.8, (1, 1, 0))
        aa_renderer.draw_circle((0, 0), 1.8, (1, 1, 0))
        # draw 12 hours markers from a simple loop.
        for i in range(1, 13):
            dx, dy = (math.cos(ONE_HOUR_OFFSET * i) * 1.3, math.sin(ONE_HOUR_OFFSET * i) * 1.3)
            # plain: plain_renderer.draw_line((dx, dy), (dx * 1.3, dy * 1.3), (1, 1, 0), 0.05)
            aa_renderer.draw_line((dx, dy), (dx * 1.3, dy * 1.3), (1, 1, 0), 0.05)

            # text is AA-only (font.py has no GL_LINES twin): `size` is the cap
            # height in world units, and draw_text returns the advance width so a
            # label can be centred on its hour. These labels live in world space,
            # so they scale with the zoom like the rest of the scene.
            label = str(i)
            label_size = 0.30
            label_x = math.cos(OCLOCK - ONE_HOUR_OFFSET * i) * 2.05
            label_y = math.sin(OCLOCK - ONE_HOUR_OFFSET * i) * 2.05
            label_half = aa_renderer.measure_text(label, label_size) * 0.5
            aa_renderer.draw_text(
                (label_x - label_half, label_y - label_size * 0.36), label, label_size, (1, 1, 0)
            )

        # get the elapsed seconds from the program start.
        elapsed_second = glfw.get_time() - start
        elapsed_second *= 100  # speed up the time by this scalar.
        # the above function, parameters: application instance, origin, seconds...
        # plain GL_LINES clock: analog_display(plain_renderer, elapsed_second)
        analog_display(aa_renderer, elapsed_second)

        # create/display an array of vectors pointing in the direction of the cursor
        step = 0.3
        array_x, array_y = (10, 5)
        anchor_x, anchor_y = (1.2, -2.6)
        mid_x, mid_y = (anchor_x + (array_x * 0.45 * step), anchor_y - (array_y * 0.4 * step))
        cx, cy = tip
        dx, dy = (cx - mid_x, cy - mid_y)
        length = math.sqrt(dx * dx + dy * dy)
        # Guard the normalization: with pan/zoom the cursor can sit exactly on the
        # array centre, which used to raise ZeroDivisionError.
        normalized_dir = (dx / length, dy / length) if length > 1e-9 else (0.0, 0.0)
        for i in range(0, array_x):
            for j in range(0, array_y):
                from_pt = (anchor_x + (i * step), anchor_y + (-j * step))
                to_pt = (from_pt[0] + normalized_dir[0], from_pt[1] + normalized_dir[1])
                # plain: plain_renderer.draw_arrow(from_pt, to_pt)
                aa_renderer.draw_arrow(from_pt, to_pt)

        # Anti-aliased HUD readout, anchored to the *visible* bottom-left corner.
        # `hud_scale` cancels the zoom so the text keeps a constant screen size:
        # world-space text would otherwise become unreadable when zoomed out.
        zoom = camera.zoom_factor()
        decimals = digits_for_zoom(zoom)
        readout = (
            f"tip ≈ ({tip[0]:+.{decimals}f}, {tip[1]:+.{decimals}f})"
            f"   zoom ×{zoom:.3g}"
            f"   centre ({camera.center[0]:+.{decimals}f}, {camera.center[1]:+.{decimals}f})"
        )
        fb_width, fb_height = glfw.get_framebuffer_size(window)
        hud_scale = camera.view_half_y / HALF
        min_x, min_y, _max_x, _max_y = camera.visible_rect(fb_width, fb_height)
        aa_renderer.draw_text(
            (min_x + 0.30 * hud_scale, min_y + 0.46 * hud_scale),
            readout,
            0.26 * hud_scale,
            (0.85, 0.92, 1.0),
        )

        """
        -> update the arrow on terminal console only if a significant move occur
        """
        # boolean predicate if hypotenuse is bigger or equalt than a threshold
        moved = (
            last_printed is None
            or math.hypot(tip[0] - last_printed[0], tip[1] - last_printed[1]) >= PRINT_STEP
        )

        # update position on terminal only if a click or a significant move occur.
        if clicked or (PRINT_STEP > 0 and moved):
            sys.stdout.write(
                f"{yellow}\r\x1b[4;0H\x1b[KClick "
                f"state:{purple}{clicked}{yellow}, Origin: ({origin[0]:+.2f}, {origin[1]:+.2f}) -> "
                f"tip: ({tip[0]:+.2f}, {tip[1]:+.2f})   "
                f"cursor: px ({cursor_x:.0f}, {cursor_y:.0f}) = world "
                f"({raw[0]:+.2f},{raw[1]:+.2f})   "
                f"zoom: {purple}×{zoom:.2f}{yellow}  centre: "
                f"({camera.center[0]:+.2f}, {camera.center[1]:+.2f}){rst}"
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
