from __future__ import annotations

import math
import sys

import glfw
from geometry import clamp_to_grid
from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor
from renderer import GridArrowRenderer
from settings import ARROW_COLOR, PRINT_STEP
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
"""

yellow = "\x1b[033m"
purple = "\x1b[035m"
rst = "\x1b[0m"

ONE_HOUR_OFFSET = math.tau / 12
ONE_SECOND_MINUTE_OFFSET = math.tau / 60
OCLOCK = math.pi * 0.5


def analog_display(
    system: GridArrowRenderer,
    seconds: float,
) -> None:
    """
        Simply draw and analog time watch representation.
    """
    seconds_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * seconds)
    number_of_minutes = seconds / 60
    minutes_angle = OCLOCK - (ONE_SECOND_MINUTE_OFFSET * number_of_minutes)
    number_of_hours = number_of_minutes / 60
    hours_angle = OCLOCK - (ONE_HOUR_OFFSET * number_of_hours)

    sx, sy = (math.cos(seconds_angle), math.sin(seconds_angle))
    mx, my = (math.cos(minutes_angle), math.sin(minutes_angle))
    hx, hy = (math.cos(hours_angle), math.sin(hours_angle))
    
    system.draw_arrow((0,0), (sx * 1.7, sy * 1.7), (1, 0, 0))
    system.draw_arrow((0,0), (mx * 1.3, my * 1.3), (1, 1, 0))
    system.draw_arrow((0,0), (hx, hy), (1, 0, 1))


def main() -> int:

    # Create a window object instance.
    window = create_window()

    # Create 2d System app instance.
    # ( not the subject there so the Api abstraction is fine for now)
    # ( we should know already how to create a 2d system in a wished orientation )
    renderer = GridArrowRenderer(window)

    # Vector standard 2d coordinates.
    origin = (0.0, 0.0)  # arrow tail: edit this coordinate to move the arrow
    arrow_color = ARROW_COLOR  # edit to recolour the arrow, e.g. (1.0, 0.3, 0.1)

    # Readout state: the last tip printed, and the previous mouse-button state.
    last_printed: tuple[float, float] | None = None
    mouse_was_down = False

    # Default background color
    BACKGROUND = (0.06, 0.06, 0.07)
    # *BACKGOUND mean unpacking the tuple value
    glClearColor(*BACKGROUND, 1.0)
    start = glfw.get_time()

    sys.stdout.write("\x1b[2J")  # clear the screen.
    # ------------------------------------------------------------------
    # Events loop shape: events -> clear -> draw -> present.
    # Everything between clear and swap can be customized and rendered at
    # each frames.
    # ------------------------------------------------------------------
    while not glfw.window_should_close(window):
        glfw.poll_events()

        raw = cursor_world(window)  # unclamped cursor coordinate
        tip = clamp_to_grid(raw)  # arrow endpoint, kept inside the grid
        mouse_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        clicked = mouse_down and not mouse_was_down  # rising edge, not held

        glClear(GL_COLOR_BUFFER_BIT)  # the technical location of the buffer bit in memory
        renderer.draw_grid()
       
        # draw arrow from origin to cursor position
        # renderer.draw_arrow(origin, tip, arrow_color)

        # draw a circle.
        renderer.draw_circle((0, 0), 1.8, (1, 1, 0))
        # draw 12 hours markers from a simple loop.
        for i in range(1, 13):
            dx, dy = (math.cos(ONE_HOUR_OFFSET * i) * 1.3, math.sin(ONE_HOUR_OFFSET * i) * 1.3)
            renderer.draw_line((dx, dy), (dx * 1.3, dy * 1.3), (1, 1, 0),0.05)

        # get the elapsed seconds from the program start.
        elapsed_second = glfw.get_time() - start
        elapsed_second *= 100  # speed up the time by this scalar.
        # the above function, parameters: application instance, origin, seconds...
        analog_display(renderer, elapsed_second)

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
            px, py = glfw.get_cursor_pos(window)
            sys.stdout.write(
                f"{yellow}\r\x1b[4;0H\x1b[KClick "
                f"state:{purple}{clicked}{yellow}, Origin: ({origin[0]:+.2f}, {origin[1]:+.2f}) -> "
                f"tip: ({tip[0]:+.2f}, {tip[1]:+.2f})   "
                f"cursor: px ({px:.0f}, {py:.0f}) = world ({raw[0]:+.2f},{raw[1]:+.2f}){rst}"
            )
            sys.stdout.flush()
            last_printed = tip

        mouse_was_down = mouse_down
        glfw.swap_buffers(window)
    # ------------------------------------------------------------------

    renderer.close()
    close_window(window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
