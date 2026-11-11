# 008 — graphical user interface

A small, self-contained Python project: a window you draw into with one
function, screen-space shapes that know when the mouse is on them, and plain
strings sent between programs over UDP.

**Start with [guide.html](guide.html)** — open it in a browser. It explains the
whole API step by step.

```
toolchain/   installers for macOS, Linux and Windows + an environment checker
graphics/    the drawing library (wgpu): 3D lines, arrows, particles, pictures, text,
             and screen-space rectangles / circles (filled, outlined or gradient) / inverted frames / triangles / lines / dashed lines / pictures
             with their hit tests
network/     udp_link (strings over UDP) and video_link (camera over UDP)
examples/    01_hover_shapes   two shapes that change colour under the cursor
             02_udp_drag       drag a circle in window A, window B follows over UDP
             03_udp_video      a camera (or test pattern) in window A, shown in window B over UDP
             04_linked_circles two draggable circles joined by a dashed line
             05_keyboard       the keys held, the last presses and the typed text, on screen
             06_slider         sliders that set a colour and a corner radius
             07_gradient_gauge a gradient gauge whose marker moves with the Up / Down keys
             08_map_viewer     pinch / wheel zoom and pan of a large map inside a rounded frame
guide.html   the step-by-step guide
```

## Install

From this folder, once. Needs Python 3.11 or newer.

```
./toolchain/setup-macos.sh                                              # macOS
./toolchain/setup-linux.sh                                              # Linux
powershell -ExecutionPolicy Bypass -File .\toolchain\setup-windows.ps1  # Windows
```

This creates `.venv/` here and installs glfw, wgpu, rendercanvas, numpy, Pillow
and (optional, `--no-opencv` / `-NoOpenCV` to skip) OpenCV, then checks the GPU.

## Run

```
.venv/bin/python toolchain/check_env.py --window
.venv/bin/python examples/01_hover_shapes/main.py
.venv/bin/python examples/02_udp_drag/main.py
.venv/bin/python examples/03_udp_video/main.py            # add --source 0 for a real camera
.venv/bin/python examples/04_linked_circles/main.py
.venv/bin/python examples/05_keyboard/main.py
.venv/bin/python examples/06_slider/main.py
.venv/bin/python examples/07_gradient_gauge/main.py
.venv/bin/python examples/08_map_viewer/main.py           # add --image your_map.jpg for your own picture
.venv/bin/python graphics/main.py
```

On Windows the interpreter is `.venv\Scripts\python`. Every program is reactive
by default (no event, no frame, no CPU) and accepts `--poll` to redraw
continuously.

## The API in ten lines

```python
import app3d                                   # graphics/
app = app3d.App("my tool", size=(640, 400))
ui = app.aa

def draw_frame():
    over = ui.in_screen_rect(app.cursor(), (60, 90), (260, 150), 28)
    ui.draw_screen_rect((60, 90), (260, 150), (1, .85, .1) if over else (.58, .25, .9), 28)
    ui.draw_screen_circle((500, 165), 75, (.58, .25, .9))
    ui.draw_screen_gradient_rect((360, 90), (60, 150), [0, .5, 1],                        # stops ...
                                 [(0, 0, 1), (0, 1, 0), (1, 0, 0)], (0, -1), 20)          # ... colours, direction, radius
    ui.draw_screen_triangle((320, 250), 60, 80, (0, -1), (1, .55, .1))                    # base, height, direction
    ui.draw_screen_line((60, 280), (575, 280), (1, 1, 1), 2)                              # thickness
    ui.draw_screen_dashed_line((60, 310), (575, 310), (1, .55, .1), 4, dash=12, spacing=6)

app.run(draw_frame)
```

```python
from widgets2d import Slider                   # graphics/
volume = Slider((60, 200), (360, 200), 0, 100, value=50)    # start, end, minimum, maximum
# in draw_frame:  level = volume.draw(app)     -> a float; drag the handle or click the line
```

```python
from udp_link.bridge import LinkThread         # network/
link = LinkThread("server", "network/link.toml", on_text=print).start()
link.send("x=120;y=80")

from video_link.bridge import VideoThread      # network/ (needs OpenCV)
video = VideoThread("client", "network/link.toml", on_frame=lambda picture: app.request_redraw()).start()
# in draw_frame:  texture.update(video.latest);  ui.draw_screen_image(anchor, size, texture)
```

The connection settings (same machine by default, or two machines on a local
network) are in `network/link.toml`; the guide and the headers of
`examples/02_udp_drag/main.py` and `examples/03_udp_video/main.py` explain them.
