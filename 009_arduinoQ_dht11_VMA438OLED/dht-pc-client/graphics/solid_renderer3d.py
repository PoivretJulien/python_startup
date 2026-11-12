"""Solid, lit 3D shapes: the drawing methods (mixed into both renderers).

    ui.set_light((0.4, -0.5, 0.8))                      # once, or every frame
    ui.draw_sphere((0, 0, 1), 1.0, ORANGE)
    ui.draw_box((2, 0, 0.5), (1, 1, 1), BLUE, rotation=(0, 0, 30))
    ui.draw_mesh(ship, GREY, position=(0, 2, 0))        # ship = Mesh.load("ship.obj")

Two ways of drawing the same calls, chosen by ``ui.raymarch`` (the
``--raymarch`` option of every program, or the F2 key):

* off (default) — every shape is triangles, lit by one light. Fast everywhere.
* on — spheres, boxes, cylinders, cones, capsules and tori are *ray-marched*:
  each pixel finds the exact mathematical surface, so curves are perfectly
  round at any zoom, and the shapes cast soft shadows on each other and on
  meshes, with darker creases. Meshes stay triangles.

Every method takes the same optional ``shader``, ``specular`` and ``shininess``
(see ``solid_shaders3d``).

See-through solids: give a colour with a fourth number, its opacity —
``(0.6, 0.8, 1.0, 0.2)`` is a light blue glass — and optionally
``refraction``, how much the material bends what is seen through it (1.0 not
at all, 1.33 water, 1.5 glass). They need one more drawing pass, used only in
frames that contain one; ``ui.transparency = False`` (the ``--no-transparency``
option, the F3 key) draws them opaque instead. Draw them after the opaque
things of the scene.
"""

from __future__ import annotations

import time

import gpu3d
import math3d
import numpy as np
import solid3d
import solid_shaders3d
import wgpu
from solid_shaders3d import MAX_SOLIDS, SCENE_FLOATS, SCENE_HEADER, SHADOW_MAP_SIZE

_UP = np.eye(3)
SEGMENTS = 48  # slices around a round shape
WIRE_SEGMENTS = 20  # ... when its shader draws the triangles' edges


def _sphere(segments: int) -> np.ndarray:
    return solid3d.unit_sphere(segments, segments // 2)


def _box(_segments: int) -> np.ndarray:
    return solid3d.unit_box()


class SolidMethods:
    """``draw_sphere`` and friends. Needs ``self.gpu`` and ``self._camera_uniforms``."""

    #: True = ray-march the analytic shapes (soft shadows, exact curves).
    raymarch = False
    #: False = draw see-through solids opaque: no second pass, no extra cost.
    transparency = True
    #: False = no shadow map: meshes (and every solid drawn as triangles) cast
    #: no shadow, and its drawing pass is skipped.
    shadows = True

    # ------------------------------------------------------------- state
    def _solids(self) -> dict:
        """Everything the solids need, made the first time one is drawn."""
        state = self.__dict__.get("_solid_state")
        if state is None:
            gpu = self.gpu
            buffer = gpu.device.create_buffer(
                size=SCENE_FLOATS * 4, usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
            )
            state = self._solid_state = {
                "buffer": buffer,
                "group": None,  # made by _shadow_target, with the shadow map
                "shadow_map": None,
                "shadow_view": None,
                "casters": [],  # (vertex buffer, vertex count, model matrix)
                "caster_buffer": None,
                "caster_group": None,
                "caster_layout": None,
                "shadow_pipeline": None,
                "light_matrix": None,
                "scene": np.zeros(SCENE_FLOATS, dtype=np.float32),
                "count": 0,
                "glass": 0,  # see-through solids, stored from the end: they cast no shadow
                "sizes": [],
                "low": np.full(3, np.inf),
                "high": np.full(3, -np.inf),
                "serial": -1,
                "shapes": {},  # name -> (GPU buffer, vertex count)
                "start": time.perf_counter(),
                "light": (np.array((0.35, -0.45, 0.82)) / np.linalg.norm((0.35, -0.45, 0.82)), (1.0, 1.0, 1.0), 0.3),
            }
            self._shadow_target(state, 1)  # a 1-pixel stand-in until a shadow is needed
            gpu.before_end_frame.append(self._flush_solids)
            gpu.before_main_pass.append(self._shadow_pass)
        if state["serial"] != self.gpu.frame_serial:  # first solid of a new frame
            state["serial"] = self.gpu.frame_serial
            # The shadow map gets its real size here, at the start of a frame,
            # and never later: draw calls already recorded in a frame keep a
            # reference to the texture they were given.
            if self.shadows:
                self._shadow_target(state, SHADOW_MAP_SIZE)
            state["count"] = state["glass"] = 0
            state["sizes"].clear()
            state["casters"].clear()
            state["cast_low"] = np.full(3, np.inf)
            state["cast_high"] = np.full(3, -np.inf)
            state["low"][:] = np.inf
            state["high"][:] = -np.inf
        return state

    def _flush_solids(self) -> None:
        """Send the frame's list of solids to the GPU (called before encoding)."""
        state = self.__dict__.get("_solid_state")
        if state is None or state["serial"] != self.gpu.frame_serial:
            return
        scene = state["scene"]
        count = state["count"]
        detail = float(np.median(state["sizes"])) if state["sizes"] else 1.0
        reach = float(np.linalg.norm(state["high"] - state["low"])) if count else 1.0
        scene[0:4] = (count, detail, max(reach, detail), 0.0)
        self._aim_shadow_map(state)
        self.gpu.queue.write_buffer(state["buffer"], 0, scene)

    # ---------------------------------------------------------- shadow map
    # Ray-marched shapes know their own shadows. Triangles do not, so they are
    # drawn once more, seen from the light, keeping only the depths: a point
    # that is farther from the light than the depth stored there is in shadow.
    def _shadow_target(self, state: dict, size: int) -> None:
        """Make the shadow map ``size`` pixels wide, and the group that binds it."""
        if state["shadow_map"] is not None:
            if state["shadow_map"].size[0] == size:
                return
            state["shadow_map"].destroy()
        gpu = self.gpu
        state["shadow_map"] = gpu.device.create_texture(
            size=(size, size, 1),
            format="depth32float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
        )
        state["shadow_view"] = state["shadow_map"].create_view()
        buffer = state["buffer"]
        state["group"] = gpu.device.create_bind_group(
            layout=gpu.scene_layout,
            entries=[
                {"binding": 0, "resource": {"buffer": buffer, "offset": 0, "size": buffer.size}},
                {"binding": 1, "resource": state["shadow_view"]},
                {"binding": 2, "resource": gpu.shadow_sampler},
            ],
        )

    def _aim_shadow_map(self, state: dict) -> None:
        """Fit the light's view around this frame's casters; fill the scene header."""
        scene = state["scene"]
        scene[4:24] = 0.0
        state["light_matrix"] = None
        if not state["casters"]:
            return
        center = 0.5 * (state["cast_low"] + state["cast_high"])
        radius = max(0.5 * float(np.linalg.norm(state["cast_high"] - state["cast_low"])), 1e-6)
        toward_light = state["light"][0]
        # Look from the light at the casters; the view is a box (the light is
        # infinitely far, its rays are parallel) just large enough for them.
        up = (0.0, 0.0, 1.0) if abs(toward_light[2]) < 0.99 else (1.0, 0.0, 0.0)
        view = math3d.look_at(center + toward_light * radius, center, up=up).astype(np.float64)
        box = np.diag((1.0 / radius, 1.0 / radius, -0.5 / radius, 1.0))
        matrix = box @ view
        state["light_matrix"] = matrix
        texel = 2.0 * radius / SHADOW_MAP_SIZE  # world size of one pixel of the map
        scene[4:8] = (1.0, 2.5 * texel, 1.5 / SHADOW_MAP_SIZE, 1.0 / SHADOW_MAP_SIZE)
        scene[8:24] = matrix.astype(np.float32).T.ravel()

    def _shadow_pass(self, encoder) -> None:
        """Draw this frame's casters into the shadow map (before the main pass)."""
        state = self.__dict__.get("_solid_state")
        if state is None or state["serial"] != self.gpu.frame_serial or state["light_matrix"] is None:
            return
        gpu = self.gpu
        casters = state["casters"]
        stride = gpu.uniform_stride
        if state["shadow_pipeline"] is None:
            state["caster_layout"] = gpu.device.create_bind_group_layout(
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": "uniform", "has_dynamic_offset": True, "min_binding_size": 64},
                    }
                ]
            )
            state["shadow_pipeline"] = gpu.device.create_render_pipeline(
                layout=gpu.device.create_pipeline_layout(bind_group_layouts=[state["caster_layout"]]),
                vertex={
                    "module": gpu.shader(solid_shaders3d.SHADOW_PASS),
                    "entry_point": "vs_main",
                    "buffers": [gpu3d.vertex_layout(*solid_shaders3d.SOLID_LAYOUT)],
                },
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={"format": "depth32float", "depth_write_enabled": True, "depth_compare": "less"},
                multisample={"count": 1},
            )
        if state["caster_buffer"] is None or state["caster_buffer"].size < len(casters) * stride:
            if state["caster_buffer"] is not None:
                state["caster_buffer"].destroy()
            state["caster_buffer"] = gpu.device.create_buffer(
                size=max(len(casters) * 2, 16) * stride,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            state["caster_group"] = gpu.device.create_bind_group(
                layout=state["caster_layout"],
                entries=[{"binding": 0, "resource": {"buffer": state["caster_buffer"], "offset": 0, "size": 64}}],
            )
        block = np.zeros((len(casters), stride // 4), dtype=np.float32)
        for index, (_buffer, _count, model) in enumerate(casters):
            block[index, :16] = (state["light_matrix"] @ model).astype(np.float32).T.ravel()
        gpu.queue.write_buffer(state["caster_buffer"], 0, block)

        shadow_pass = encoder.begin_render_pass(
            color_attachments=[],
            depth_stencil_attachment={
                "view": state["shadow_view"],
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store",
            },
        )
        shadow_pass.set_pipeline(state["shadow_pipeline"])
        for index, (buffer, count, _model) in enumerate(casters):
            shadow_pass.set_bind_group(0, state["caster_group"], [index * stride])
            shadow_pass.set_vertex_buffer(0, buffer)
            shadow_pass.draw(count, 1)
        shadow_pass.end()

    def _shape_buffer(self, name: str, build) -> tuple:
        shapes = self._solids()["shapes"]
        entry = shapes.get(name)
        if entry is None:
            vertices = build()
            entry = shapes[name] = (self.gpu.create_vertex_buffer(vertices), int(vertices.shape[0]))
        return entry

    def _solid_pipeline(self, shader: solid_shaders3d.Shader, marched: bool, glass: bool):
        # Ray-marched: far faces only, so each pixel is computed once, even
        # with the camera inside the box.
        return self.gpu.pipeline(
            shader.sources[marched, glass],
            gpu3d.vertex_layout(*solid_shaders3d.SOLID_LAYOUT),
            scene=True,
            backdrop=glass,
            cull_mode="front" if marched else "none",
        )

    def _opacity(self, color) -> float:
        """The alpha of a colour, or 1 when see-through solids are switched off."""
        if not self.transparency or len(color) < 4:
            return 1.0
        return min(max(float(color[3]), 0.0), 1.0)

    def _submit_solid(self, marched, entry, color, model, look, center, thickness, index=0, bounds=(0, 0, 0), reach=0.0) -> None:
        shader, specular, shininess, refraction = look
        # Triangles cast their shadow through the shadow map (ray-marched
        # shapes have their own; see-through solids cast none).
        if self.shadows and not marched and not (len(color) > 3 and float(color[3]) < 1.0):
            state = self._solids()
            state["casters"].append((entry[0], entry[1], np.asarray(model, dtype=np.float64)))
            middle = np.asarray(center, dtype=np.float64)[:3]
            state["cast_low"] = np.minimum(state["cast_low"], middle - reach)
            state["cast_high"] = np.maximum(state["cast_high"], middle + reach)
        opacity = self._opacity(color)
        glass = opacity < 1.0
        uniforms = self._solid_uniforms(color, model, specular, shininess, index, bounds)
        uniforms[27] = opacity
        uniforms[37] = refraction
        uniforms[38] = thickness
        buffer, count = entry
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self._solid_pipeline(solid_shaders3d.resolve(shader), marched, glass),
                uniforms=uniforms,
                vertex_count=count,
                buffer=buffer,
                texture_group=self._solids()["group"],
                backdrop=glass,
                order=float(np.linalg.norm(np.asarray(center, dtype=np.float64)[:3] - self.camera.eye())),
            )
        )

    def _solid_uniforms(self, color, model, specular, shininess, index=0, bounds=(0, 0, 0)) -> np.ndarray:
        state = self._solids()
        direction, light_color, ambient = state["light"]
        out = np.zeros(gpu3d.UNIFORM_SLOT_FLOATS, dtype=np.float32)
        out[:40] = self._camera_uniforms(eye=tuple(float(v) for v in self.camera.eye()))
        out[20:23] = direction
        out[23] = ambient
        out[24:27] = color[:3]
        out[27] = 1.0
        out[32:36] = (specular, shininess, time.perf_counter() - state["start"], 1.0 if self.raymarch else 0.0)
        out[36] = index
        out[40:56] = np.asarray(model, dtype=np.float32).T.ravel()
        out[56:59] = light_color
        out[60:63] = bounds
        return out

    def _draw_solid(self, kind, params, center, frame, color, look, parts) -> None:
        """One analytic shape: ray-marched when on (and there is room), else triangles."""
        state = self._solids()
        center = np.asarray(center, dtype=np.float64)[:3]
        bounds = solid3d.half_bounds(kind, params)
        thickness = 2.0 * min(bounds)
        needs_triangles = solid_shaders3d.resolve(look[0]).triangles
        if self.raymarch and not needs_triangles and state["count"] + state["glass"] < MAX_SOLIDS:
            # A solid declared see-through never casts a shadow, even when the
            # switch draws it opaque: switching transparency off must not add
            # work to the shadow computation of every other pixel.
            if len(color) > 3 and float(color[3]) < 1.0:
                state["glass"] += 1
                index = MAX_SOLIDS - state["glass"]
            else:
                index = state["count"]
                state["count"] += 1
                state["sizes"].append(max(bounds))
                reach = float(np.linalg.norm(bounds))
                state["low"] = np.minimum(state["low"], center - reach)
                state["high"] = np.maximum(state["high"], center + reach)
            first = SCENE_HEADER + 16 * index
            state["scene"][first : first + 16] = solid3d.solid_row(kind, params, center, frame)
            padded = tuple(b * 1.02 + 1e-4 for b in bounds)
            model = solid3d.model_matrix(center, frame, padded)
            box = self._shape_buffer("box", solid3d.unit_box)
            self._submit_solid(True, box, color, model, look, center, thickness, index, padded)
            return
        # Round shapes are cut into `segments` slices: many for a smooth look,
        # few when the shader shows the triangles (a readable wireframe).
        segments = WIRE_SEGMENTS if needs_triangles else SEGMENTS
        for name, build, offset, scale in parts:
            model = solid3d.model_matrix(center + frame @ np.asarray(offset, dtype=np.float64), frame, scale)
            entry = self._shape_buffer(f"{name} {segments}", lambda: build(segments))
            self._submit_solid(
                False, entry, color, model, look, center, thickness,
                reach=float(np.linalg.norm(bounds)),
            )

    # --------------------------------------------------------------- light
    def set_light(self, direction=(0.35, -0.45, 0.82), color=(1.0, 1.0, 1.0), ambient: float = 0.3) -> None:
        """Choose the one light that shines on every solid.

        ``direction`` points *toward* the light, like the sun's position in the
        sky: ``(0, 0, 1)`` is straight overhead (z is up). ``color`` is the
        light's colour, ``ambient`` (0..1) how bright the side facing away
        stays. Call it once, or every frame to move the light.
        """
        d = np.asarray(direction, dtype=np.float64)[:3]
        length = float(np.linalg.norm(d))
        d = d / length if length > 1e-12 else np.array((0.0, 0.0, 1.0))
        self._solids()["light"] = (d, tuple(float(c) for c in color[:3]), float(ambient))

    # -------------------------------------------------------------- shapes
    def draw_sphere(self, center, radius: float, color, *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A ball of ``radius`` around ``center``."""
        r = float(radius)
        parts = (("sphere", _sphere, (0, 0, 0), (r, r, r)),)
        self._draw_solid(solid3d.SPHERE, (r,), center, _UP, color, (shader, specular, shininess, refraction), parts)

    def draw_box(self, center, size, color, rotation=(0.0, 0.0, 0.0), *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A box of ``size = (x, y, z)`` centred on ``center``.

        ``rotation`` turns it: degrees around x, y and z. A wide, thin box
        makes a floor.
        """
        half = tuple(0.5 * float(v) for v in size[:3])
        frame = solid3d.rotation(rotation) if any(rotation) else _UP
        parts = (("box", _box, (0, 0, 0), half),)
        self._draw_solid(solid3d.BOX, half, center, frame, color, (shader, specular, shininess, refraction), parts)

    def _between(self, start, end):
        a = np.asarray(start, dtype=np.float64)[:3]
        b = np.asarray(end, dtype=np.float64)[:3]
        return 0.5 * (a + b), solid3d.frame_along(b - a), 0.5 * float(np.linalg.norm(b - a))

    def draw_cylinder(self, start, end, radius: float, color, *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A cylinder of ``radius`` from the point ``start`` to the point ``end``."""
        center, frame, half = self._between(start, end)
        r = float(radius)
        parts = (("cylinder", solid3d.unit_cylinder, (0, 0, 0), (r, r, half)),)
        self._draw_solid(solid3d.CYLINDER, (r, half), center, frame, color, (shader, specular, shininess, refraction), parts)

    def draw_cone(self, base, tip, radius: float, color, *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A cone: a disc of ``radius`` at ``base``, narrowing to the point ``tip``."""
        center, frame, half = self._between(base, tip)
        r = float(radius)
        parts = (("cone", solid3d.unit_cone, (0, 0, 0), (r, r, half)),)
        self._draw_solid(solid3d.CONE, (r, half), center, frame, color, (shader, specular, shininess, refraction), parts)

    def draw_capsule(self, start, end, radius: float, color, *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A cylinder with round ends: every point within ``radius`` of the segment."""
        center, frame, half = self._between(start, end)
        r = float(radius)
        parts = (
            ("cylinder_open", lambda n: solid3d.unit_cylinder(n)[: 6 * n], (0, 0, 0), (r, r, half)),
            ("sphere", _sphere, (0, 0, -half), (r, r, r)),
            ("sphere", _sphere, (0, 0, half), (r, r, r)),
        )
        self._draw_solid(solid3d.CAPSULE, (r, half), center, frame, color, (shader, specular, shininess, refraction), parts)

    def draw_torus(self, center, radius: float, tube: float, color, axis=(0.0, 0.0, 1.0), *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """A ring of ``radius`` made of a tube of radius ``tube``.

        ``axis`` is the direction of the hole: ``(0, 0, 1)`` lays the ring flat.
        """
        big, small = float(radius), float(tube)
        ratio = round(small / big, 3) if big > 0.0 else 0.0
        parts = ((f"torus {ratio}", lambda n: solid3d.unit_torus(ratio, n, n // 2), (0, 0, 0), (big, big, big)),)
        self._draw_solid(
            solid3d.TORUS, (big, small), center, solid3d.frame_along(axis), color, (shader, specular, shininess, refraction), parts
        )

    def draw_mesh(self, mesh: solid3d.Mesh, color, position=(0.0, 0.0, 0.0), scale=1.0, rotation=(0.0, 0.0, 0.0), *, shader=None, specular: float = 0.35, shininess: float = 32.0, refraction: float = 1.0) -> None:
        """Draw a :class:`solid3d.Mesh` (for instance ``Mesh.load("file.obj")``).

        The mesh's own origin goes to ``position``; ``scale`` is one number or
        ``(x, y, z)``; ``rotation`` is degrees around x, y and z.
        """
        if mesh._buffer is None:
            mesh._buffer = self.gpu.create_vertex_buffer(mesh.vertices)
        scale3 = (scale, scale, scale) if np.isscalar(scale) else tuple(scale[:3])
        frame = solid3d.rotation(rotation) if any(rotation) else _UP
        model = solid3d.model_matrix(position, frame, scale3)
        thickness = 0.5 * float(np.min(np.abs(mesh.size * np.asarray(scale3, dtype=np.float64))))
        center = np.asarray(position, dtype=np.float64)[:3] + frame @ (mesh.center * np.asarray(scale3))
        look = (shader, specular, shininess, refraction)
        reach = 0.5 * float(np.linalg.norm(mesh.size * np.asarray(scale3, dtype=np.float64)))
        self._submit_solid(
            False, (mesh._buffer, mesh.triangle_count * 3), color, model, look, center, thickness, reach=reach
        )

    # ------------------------------------------------------------- cleanup
    def _close_solids(self) -> None:
        state = self.__dict__.pop("_solid_state", None)
        if state is None:
            return
        if self._flush_solids in self.gpu.before_end_frame:
            self.gpu.before_end_frame.remove(self._flush_solids)
        if self._shadow_pass in self.gpu.before_main_pass:
            self.gpu.before_main_pass.remove(self._shadow_pass)
        state["buffer"].destroy()
        state["shadow_map"].destroy()
        if state["caster_buffer"] is not None:
            state["caster_buffer"].destroy()
        for buffer, _count in state["shapes"].values():
            buffer.destroy()
