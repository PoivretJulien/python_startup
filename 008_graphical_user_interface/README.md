<img width="3000" height="2015" alt="python" src="https://github.com/user-attachments/assets/b28eecf9-b65d-4ced-bdda-d4ec6a2c832d" />

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
             09_solids         lit solid shapes, a model from an .obj file, shaders, ray marching
             10_curves         polylines and NURBS curves, 3D control points moved with a gizmo, an acrylic panel
guide.html   the step-by-step guide, ending with the full API reference (chapter 20)
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
.venv/bin/python examples/09_solids/main.py               # add --raymarch for exact shapes and soft shadows
.venv/bin/python examples/10_curves/main.py               # add --points 200000 for a longer polyline
.venv/bin/python graphics/main.py
```

Add `--ruler` to any of them for a measuring overlay (cursor position, and the
distance, `anchor` and `size` of a click-drag), to help place interface
elements; F1 switches it on and off.

Add `--raymarch` to ray-march the solid shapes (F2 switches it on and off).
See-through solids (a colour with an alpha, optional `refraction=1.5`) need one
more drawing pass; `--no-transparency` or F3 draws them opaque instead.
Solids cast shadows on each other (a shadow map for triangles, exact soft
shadows for ray-marched shapes); `--no-shadows` or F4 removes the shadow map.
See-through solids cast a light shadow tinted by their colour;
`--no-glass-shadows` or F5 removes it.

On Windows the interpreter is `.venv\Scripts\python`. Every program is reactive
by default (no event, no frame, no CPU) and accepts `--poll` to redraw
continuously.

## The API in ten lines

```python
import app3d                                   # graphics/
from vec2 import Vec2                          # optional: points with arithmetic ...
from vec3 import Vec3                          # ... usable wherever a tuple is
app = app3d.App("my tool", size=(640, 400))
ui = app.aa

def draw_frame():
    center = Vec2(*app.size()) / 2                                                        # the middle of the window
    ui.draw_screen_line(center, center + Vec2(80, 40), (1, 1, 1), 2)                      # screen: Vec2
    ui.draw_arrow(Vec3(0, 0, 0), Vec3(1, 0, 0).cross(Vec3(0, 1, 0)) * 2, (.2, .85, .95))  # world: Vec3
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
from curve3d import Nurbs                      # graphics/
# in draw_frame:
#   ui.draw_rect((0, 0, 0), (2, 0, 0), (0, 0, 1), (1, .2, .4, .5))   # 3D rectangle: corner, side u, side v, rgba
#   ui.draw_polyline(points, (.2, .85, .95))                     # N rows of (x, y, z), one call for any N
#   ui.draw_nurbs([(0, 0, 0), (1, 2, 1), (3, 2, 2), (4, 0, 0)], (1, .35, .7), thickness=0.05)   # control points
#   ui.draw_nurbs(curve, (1, .35, .7), pick=True)                # curve = Nurbs([...]): click a point, drag its 3D gizmo
#   ui.draw_point(curve.point(0.5), 0.1, (1, .85, .1))           # point(t): the position on the curve, t from 0 to 1
#   ui.draw_screen_acrylic_rect((20, 20), (320, 120), (.1, .1, .14, .45), radius=16)            # frosted glass panel
```

```python
from solid3d import Mesh                       # graphics/
ship = Mesh.load("ship.obj").fitted(2.0)       # once: centred, longest side = 2
# in draw_frame:
#   ui.set_light((0.4, -0.5, 0.8), softness=0.15)                # toward the light, z is up; softer shadows = larger
#   ui.set_area_light((2, -3, 3), radius=0.5)                    # or a lamp in the scene: its size softens shadows
#   ui.draw_sphere((0, 0, 1), 1.0, (1, .55, .1))                 # also box, cylinder, cone, capsule, torus
#   ui.draw_mesh(ship, (.8, .8, .9), position=(3, 0, 1), shader="toon")   # or your own Shader(wgsl)
#   ui.draw_sphere((0, -3, 1), 1.0, (1, 1, 1, 0.06), refraction=1.5)      # last: a glass ball (alpha = opacity)
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
