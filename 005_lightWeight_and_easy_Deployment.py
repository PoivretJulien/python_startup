"""
IMPORTANT TO KNOW WHY:

-- 3d space and multi scalars graphical evaluation --
from robotic to statistical simulation, 3d graphic are the
core of the software engineering and any interactions
feed back: forces, interactions, spacial orientation,
augmented reality...

those usually need Graphical Processor Unit (GPU),
for this OpenGl, can be used from smart python management
of course at some point python will can't keep up anymore,
because of it's non compiled nature.
but this can be specifically managed by a compiled software infrastructure
when the time will come, with real context, anticipated situation,
real way to finance the development cost, and the real knowledge of the
shaders fundamentals programming which are the key of gpu computing.

no point to start your workout directly with the 'pro' dumbbell
because this is what's really it is... for instance the low api Vulkan involve
more than 1000 lines to shade a single triangle.
do you really need this for just one triangle or ( 50 objects 3 light one
camera ? do you have the arms of a vulkan daily body builder to move
irons 3d cubes, will you enjoy it ? (without a map of the context...) )

python is great for ia development Pytorch, TensorFlow, embedded edge IA,
micro python computing vision ect, even if you are a low level expert in
the language of you choice, python still a pretty good tool to have
in you tool box, very complete, very math friendly, fast to deploy,
for shaping algorithmic prototype, ia agent routines, or simply focusing on
the math of gpu shader it's a win of time and far more convenient for the
mental model which is the real target of effectiveness.

-- the Key force: Easy deployment.
height level flexibility open source by nature (not a compiled language)
and can delegate easily to compiled binary when needed.

Objective:
   instance an OpenGl object from glfw and create some easy graphical basic entity.
   step A - create a 60 fps vsync windows.
"""

import sys
import glfw
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_RENDERER,
    GL_SHADING_LANGUAGE_VERSION,
    GL_VENDOR,
    GL_VERSION,
    glClear,
    glClearColor,
    glGetString,
    glViewport,
)

WIDTH, HEIGHT = 1000, 640

# Define some ascii terminal colors
yellow = "\x1b[33m"
red = "\x1b[31m"
blue = "\x1b[34m"
rst = "\x1b[0m"  # reset
green = "\x1b[32m"
purple = "\x1b[35m"

def main() -> int:

    print(f"{purple}Hello open Gl program{yellow}")

    # 1. GLFW is a C library: initialise it once, terminate at the end.
    if not glfw.init():
        raise RuntimeError("glfw.init() failed: is DISPLAY / WAYLAND_DISPLAY set?")

    # 2. Ask for a specific OpenGL version *before* the window exists.
    #    Without these hints you get a compatibility context (GL 2.1 style API)
    #    on some drivers — which is why the 2016 tutorials "just worked".
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # required on macOS

    # Make a windows object instance.
    window = glfw.create_window(
        WIDTH, HEIGHT, "Example 01 — a simple window", None, None
    )

    # Evaluate the returned evaluated object.
    if not window:
        glfw.terminate()
        raise RuntimeError("could not create an OpenGL 3.3 core context")

    # 3. The context is current for this thread until another one is made current.
    glfw.make_context_current(window)
    glfw.swap_interval(1)  # vsync: swap buffers at most once per screen refresh

    # 4. Resizes come as callbacks. Polling the size in the loop also works, but
    #    the framebuffer size (HiDPI) is what the viewport must use,
    #    not the window size.
    # - a call back is simply a function runtime to be scheduled in the glfw api.
    def on_resize(win, width: int, height: int) -> None:
        glViewport(0, 0, max(width, 1), max(height, 1))

    # pass the window object and the resize call back function in glfw.
    glfw.set_framebuffer_size_callback(window, on_resize)
    glViewport(0, 0, *glfw.get_framebuffer_size(window))

    def gl_string(enum: int) -> str:
        """glGetString returns None if the enum is invalid in this context."""
        raw = glGetString(enum)
        return raw.decode() if raw else "?"

    # Get the OpenGL device informations from the gl_string() method
    # matching the custom enum denomination.
    for label, enum in (
        ("vendor", GL_VENDOR),
        ("renderer", GL_RENDERER),
        ("GL", GL_VERSION),
        ("GLSL", GL_SHADING_LANGUAGE_VERSION),
    ):
        print(f"  {label:<9} {gl_string(enum)}")

    # Set some even loop runtime parameters.
    glClearColor(0.6, 0.3, 0.9, 1.0) # a purple custom clear color.
    frames = 0
    fps_window, fps_frames = 0.5, 0
    last = glfw.get_time()

    # 5. The loop: events -> draw -> present. Never busy-wait, never sleep.
    # later a better system will manage power more efficiently but for now 
    # the objective is to identify each events loop steps.
    while not glfw.window_should_close(window):
        glfw.poll_events() # execute every events
        glClear(GL_COLOR_BUFFER_BIT) # clear the screen with the clear color.
        glfw.swap_buffers(window) # update the double buffer swapchain frame buffer 
        # frame counter increment.
        frames += 1
        fps_frames += 1 
        now = glfw.get_time()
        # window and terminal fps display feedback.
        if now - last > fps_window:  # cheap FPS readout, no allocation per frame
            glfw.set_window_title(window, f"step 01 — {fps_frames / (now - last):.0f} fps")
            # Compute the frames per sec from builtin glfw timer
            fps = fps_frames / (now - last) if (now - last) > 0 else 0
            # 2. Use sys.stdout.write directly to bypass Python's print() wrapper over-checks
            # 3. Add \r (carriage return) right before your cursor command to force alignment
            sys.stdout.write(f"\r\x1b[11;1h\x1b[KTotal frames computed:{frames}  vsync:{fps:.0f} fps")
            sys.stdout.flush()
            last, fps_frames = now, 0

    # 6. Order matters: destroy the window (and it's context) before terminating GLFW.
    glfw.destroy_window(window)
    glfw.terminate()
    return 1

if __name__ == "__main__":
    sys.exit(main())
