"""Example 9 — solid shapes, a model from a file, a light and shaders.

Six solid shapes stand on a floor around a knot loaded from ``knot.obj``, with
a glass ball and a pane of blue glass in front. One light shines on all of
them; a yellow dot shows where it is.

From the project root (after running the installer in toolchain/):

    .venv/bin/python examples/09_solids/main.py                 # triangles (default)
    .venv/bin/python examples/09_solids/main.py --raymarch      # ray-marched: soft shadows
    .venv/bin/python examples/09_solids/main.py --model my.obj  # your own model

    .venv/bin/python examples/09_solids/main.py --no-transparency   # glass drawn opaque

The camera:

    left-drag                 turn around the scene
    right-drag, or two fingers on a trackpad      slide the view (pan)
    wheel, or pinch           zoom

The light:

    arrow keys                Left / Right turn it around, Up / Down raise and lower it
    hold L and move the mouse the same, by hand
    A                         switch between the sun (a far light) and an area
                              light: a glowing ball in the scene
    Z / X                     sharper / softer shadows (the sun's softness, or
                              the size of the area light)

Other keys:

    F2            switch ray marching on and off while the program runs
    F3            switch the see-through solids on and off (off = opaque)
    F4            switch the shadows of triangles (the knot, and every shape
                  when not ray-marched) on and off
    F5            switch the shadows of the see-through solids on and off:
                  light, and tinted with the colour of the glass
    Space         next shader on the knot (lit, toon, rim, normals, flat,
                  wireframe, glow, stripes)

Three ideas, each one line of code:

* a solid is one call — ``ui.draw_sphere(center, radius, color)``;
* a model is loaded once — ``Mesh.load("knot.obj")`` — and drawn like a shape;
* the look is a *shader*: a name, or a few lines of your own (see STRIPES).

With ``--raymarch`` the same calls are drawn another way: every pixel searches
for the exact surface of the sphere, the torus ..., so curves are perfectly
round and the shapes cast soft-edged shadows. The knot stays made of triangles.

Shadows come from two places. Triangles — the knot, and every shape when not
ray-marched — cast theirs through a *shadow map*: the scene is drawn once more
as the light sees it. Ray-marched shapes compute their own, with softer edges.
See-through solids cast none.

A colour with a fourth number is see-through: ``(r, g, b, 0.15)`` is 15 %
opaque. ``refraction`` bends what is seen through the solid, as glass (1.5) or
water (1.33) do. This costs one more drawing pass, so it can be switched off.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# The drawing library is the graphics/ folder of the project.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "graphics"))

import app3d  # noqa: E402
from solid3d import Mesh  # noqa: E402
from solid_shaders3d import Shader  # noqa: E402

FLOOR = (0.36, 0.37, 0.43)
ORANGE = (1.00, 0.55, 0.10)
BLUE = (0.20, 0.50, 1.00)
GREEN = (0.30, 0.80, 0.40)
RED = (0.90, 0.20, 0.30)
YELLOW = (0.85, 0.80, 0.20)
PURPLE = (0.70, 0.30, 0.90)
SILVER = (0.80, 0.82, 0.88)
CLEAR_GLASS = (1.00, 1.00, 1.00, 0.06)  # r, g, b, opacity: almost invisible
BLUE_GLASS = (0.35, 0.65, 1.00, 0.25)
SUN = (1.00, 0.90, 0.30)
LABEL = (0.85, 0.92, 1.00)
MUTED = (0.60, 0.60, 0.68)

# A shader of our own: the default look (`lit`), darkened in horizontal bands.
# `s` describes the point being coloured; s.position.z is its height and
# s.time the seconds since the start (run with --poll to see the bands move).
STRIPES = Shader(
    """
fn shade(s: Surface) -> vec3<f32> {
    let band = step(0.5, fract(s.position.z * 5.0 - s.time * 0.5));
    return lit(s) * mix(0.35, 1.0, band);
}
""",
    name="stripes",
)

KNOT_SHADERS = ["lit", "toon", "rim", "normals", "flat", "wireframe", "glow", STRIPES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=str(HERE / "knot.obj"), help="a Wavefront .obj file")
    app = app3d.App("Example 9 — solids", size=(900, 620), parser=parser)
    ui = app.aa

    # Loaded once. `fitted` centres the model and makes its longest side 2.6 units,
    # so any file fits the scene whatever its own size.
    knot = Mesh.load(app.args.model).fitted(2.6)
    app.camera.distance = 11.0  # step back: the scene is 10 units wide
    print(f"{Path(app.args.model).name}: {knot.triangle_count} triangles")

    light_angle = 300.0  # degrees around the scene
    light_height = 45.0  # degrees above the floor
    softness = 0.15  # of the sun's shadows: the width of their edge per unit of distance
    lamp_on = False  # False: a far light (the sun). True: an area light in the scene
    lamp_radius = 0.5  # the size of the area light: larger = softer shadows
    shader_index = 0
    last_cursor = app.cursor()

    def draw_frame() -> None:
        nonlocal light_angle, light_height, shader_index, last_cursor, softness, lamp_on, lamp_radius
        cursor = app.cursor()
        moved = (cursor[0] - last_cursor[0], cursor[1] - last_cursor[1])
        last_cursor = cursor

        # --- the light: arrow keys, or hold L and move the mouse ----------
        for key in app.keys_pressed(repeat=True):
            if key == "LEFT":
                light_angle -= 10.0
            elif key == "RIGHT":
                light_angle += 10.0
            elif key == "UP":
                light_height += 5.0
            elif key == "DOWN":
                light_height -= 5.0
            elif key == "A":
                lamp_on = not lamp_on
            elif key in ("X", "Z"):  # softer / sharper shadows
                step = 1.25 if key == "X" else 1 / 1.25
                if lamp_on:
                    lamp_radius = min(max(lamp_radius * step, 0.05), 1.0)
                else:
                    softness = min(max(softness * step, 0.01), 1.0)
        if app.key_down("L"):
            light_angle += 0.5 * moved[0]
            light_height -= 0.3 * moved[1]
        else:
            app.handle_camera()  # the mouse orbits, pans and zooms
        light_height = min(max(light_height, 5.0), 90.0)
        if app.key_pressed("SPACE"):
            shader_index = (shader_index + 1) % len(KNOT_SHADERS)

        # A direction that points toward the light, from two angles.
        around, above = math.radians(light_angle), math.radians(light_height)
        toward_light = (
            math.cos(above) * math.cos(around),
            math.cos(above) * math.sin(around),
            math.sin(above),
        )
        if lamp_on:
            # An area light: a glowing ball 4.5 units from the centre. It is in
            # the scene, so each shape is lit from its own side.
            lamp = tuple(4.5 * v for v in toward_light)
            ui.set_area_light(lamp, lamp_radius, ambient=0.3)
        else:
            ui.set_light(toward_light, ambient=0.3, softness=softness)

        # --- the scene ----------------------------------------------------
        ui.draw_box((0, 0, -0.05), (10, 10, 0.1), FLOOR, specular=0.0)  # the floor

        ui.draw_sphere((-3.0, 0.0, 0.8), 0.8, ORANGE)
        ui.draw_box((-3.0, 2.6, 0.6), (1.2, 1.2, 1.2), BLUE, rotation=(0, 0, 30))
        ui.draw_cylinder((3.0, 2.6, 0.0), (3.0, 2.6, 1.6), 0.5, GREEN)
        ui.draw_cone((3.0, 0.0, 0.0), (3.0, 0.0, 1.7), 0.7, RED)
        ui.draw_capsule((-3.3, -2.6, 0.4), (-2.3, -2.6, 1.3), 0.35, YELLOW)
        ui.draw_torus((3.0, -2.6, 0.9), 0.65, 0.22, PURPLE, axis=(0.3, 1.0, 0.0))

        shader = KNOT_SHADERS[shader_index]
        ui.draw_mesh(knot, SILVER, position=(0, 0, 1.2), shader=shader, shininess=60)

        # Where the light is: a dot 6 units away in its direction, and a ray.
        if not lamp_on:  # the area light draws itself
            sun = tuple(6.0 * v for v in toward_light)
            ui.draw_line(sun, (0, 0, 0), SUN)
            ui.draw_point(sun, 0.18, SUN)

        # See-through solids, after everything they must show: a glass ball
        # (it bends the view like a lens) and a flat pane of blue glass.
        ui.draw_sphere((0.0, -3.2, 0.9), 0.9, CLEAR_GLASS, refraction=1.5, specular=0.9, shininess=90)
        ui.draw_box((-0.2, 3.3, 1.0), (2.6, 0.12, 2.0), BLUE_GLASS, refraction=1.05)

        # --- the labels, on the screen ------------------------------------
        mode = "ray-marched (soft shadows)" if app.raymarch else "triangles"
        name = shader if isinstance(shader, str) else shader.name
        ui.draw_screen_text(16, 26, f"solids: {mode}", 14, LABEL)
        ui.draw_screen_text(16, 46, f"knot shader: {name}", 14, LABEL)
        glass = "on" if app.transparency else "off (opaque)"
        shadows = "on" if app.shadows else "off"
        tinted = "on" if app.glass_shadows else "off"
        ui.draw_screen_text(
            16, 66, f"see-through solids: {glass}   shadow map: {shadows}   glass shadows: {tinted}", 14, LABEL
        )
        light = f"area light, radius {lamp_radius:.2f}" if lamp_on else f"sun, softness {softness:.2f}"
        ui.draw_screen_text(16, 86, f"light: {light}", 14, LABEL)
        ui.draw_screen_text(
            16, 110, "F2 ray marching   F3 see-through   F4 shadows   F5 glass shadows   Space: shader", 12, MUTED
        )
        ui.draw_screen_text(
            16, 128, "light: arrows or L+mouse to move it   A: sun / area light   Z / X: sharper / softer shadows",
            12, MUTED,
        )
        ui.draw_screen_text(16, 146, "left-drag: orbit   right-drag / two fingers: pan   wheel / pinch: zoom", 12, MUTED)

    app.run(draw_frame)


if __name__ == "__main__":
    main()
