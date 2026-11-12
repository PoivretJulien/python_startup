"""Plain 3D renderer: one MVP shader, line-list batches, hard edges (wgpu).

This is the 3D twin of ``experimental/renderer.py``. It shares the drawing API
with ``renderer_aa3d.py`` so the scene can switch a shape back to aliased lines
by uncommenting one call. The only difference in the geometry is the camera:
every batch is multiplied by ``projection * view`` instead of the 2D camera's
``vec4`` scale/offset.

Port notes (OpenGL -> wgpu): every former ``glDrawArrays`` is now a
``gpu3d.DrawCall`` recorded in call order and replayed in one render pass at the
end of the frame. ``GL_LINES`` is the ``line-list`` topology, ``GL_POINTS``
sprites are instanced quads (WebGPU points are 1 px), and the GL state toggles
(blend, depth mask) are baked into cached pipelines.
"""

from __future__ import annotations

import geometry3d
import gpu3d
import numpy as np
import screen2d
import curve3d
import curve_edit3d
import shaders3d
import solid_renderer3d
import texture3d
from settings3d import ARROW_COLOR, POINT_MAX_PX, POINT_MIN_PX

# Sprite radius ceiling in pixels. GL clamped the sprite *diameter* to the
# driver's GL_POINT_SIZE_RANGE (POINT_MAX_PX here); instanced quads have no
# driver cap, so the same half-diameter is kept for identical output.
MAX_POINT_RADIUS_PX = 0.5 * POINT_MAX_PX


class SceneRenderer(solid_renderer3d.SolidMethods):
    """What both renderers share: the GPU, the camera and the uniform packing."""

    def __init__(self, gpu: gpu3d.GPU, camera) -> None:
        self.gpu = gpu
        self.window = gpu.window
        self.camera = camera
        # Pictures: one textured-quad pipeline plus the texture cache that keeps
        # a source from being uploaded again every frame.
        self.images = texture3d.ImageBatch(gpu)
        self.textures = texture3d.TextureStore()
        self.grid_buffer = None
        self.grid_count = 0
        self.grid_key: tuple | None = None
        # Screen-space shapes (rounded rectangles, circles): one instanced quad
        # each, never depth-tested, so they sit on top of the scene in draw
        # order. Three cached pipelines: the SDF one (rounded, blended) and the
        # sharp-rectangle twins (opaque = no blending at all, or blended).
        shapes = gpu3d.vertex_layout(*shaders3d.SCREEN_SHAPE_LAYOUT, instanced=True)
        overlay = {"depth_write": False, "depth_compare": "always"}
        self.shape_pipeline = gpu.pipeline(shaders3d.SCREEN_SHAPES, shapes, blend=True, **overlay)
        # Screen-space lines, solid or dashed: one instanced quad per line.
        self.screen_line_pipeline = gpu.pipeline(
            shaders3d.SCREEN_LINES,
            gpu3d.vertex_layout(*shaders3d.SCREEN_LINE_LAYOUT, instanced=True),
            blend=True,
            **overlay,
        )
        # Outlines and the inverted box: the same distance field, used on its edge.
        self.edge_pipeline = gpu.pipeline(
            shaders3d.SCREEN_SHAPE_EDGES,
            gpu3d.vertex_layout(*shaders3d.SCREEN_EDGE_LAYOUT, instanced=True),
            blend=True,
            **overlay,
        )
        # Gradient shapes: the SDF of the plain shapes, coloured by a 1D gradient
        # texture. Textures are cached per distinct gradient (see _gradient).
        self.gradient_pipeline = gpu.pipeline(
            shaders3d.SCREEN_GRADIENT_SHAPES, shapes, blend=True, textured=True, **overlay
        )
        self._gradients: dict[tuple, texture3d.Texture] = {}
        # Screen-space triangles: the SDF one (anti-aliased, a blended quad) and
        # the plain twins (three vertices; opaque = no blending, or blended).
        triangles = gpu3d.vertex_layout(*shaders3d.SCREEN_TRIANGLE_LAYOUT, instanced=True)
        self.screen_triangle_pipeline = gpu.pipeline(
            shaders3d.SCREEN_TRIANGLES, triangles, blend=True, **overlay
        )
        self.sharp_triangle_pipelines = {
            blend: gpu.pipeline(shaders3d.SCREEN_TRIANGLE_SHARP, triangles, blend=blend, **overlay)
            for blend in (False, True)
        }
        self.sharp_rect_pipelines = {
            blend: gpu.pipeline(shaders3d.SCREEN_RECT_SHARP, shapes, blend=blend, **overlay)
            for blend in (False, True)
        }

    # -- internals ---------------------------------------------------------
    def _framebuffer(self) -> tuple[int, int]:
        return self.gpu.framebuffer_size()

    def _mvp(self) -> np.ndarray:
        width, height = self._framebuffer()
        return self.camera.mvp(width / max(height, 1))

    def _camera_uniforms(self, **extra) -> np.ndarray:
        width, height = self._framebuffer()
        return gpu3d.pack_uniforms(
            mvp=self.camera.mvp(width / max(height, 1)),
            viewport=(float(max(width, 1)), float(max(height, 1))),
            screen=tuple(float(max(v, 1)) for v in self.gpu.window_size()),
            **extra,
        )

    def _point_uniforms(self, *, flat_depth: bool, feather: float) -> np.ndarray:
        height = self._framebuffer()[1]
        return self._camera_uniforms(
            eye=tuple(float(v) for v in self.camera.eye()),
            forward=tuple(float(v) for v in self.camera.forward()),
            pixel_scale=float(self.camera.pixel_scale(height)),
            min_radius_px=POINT_MIN_PX,
            max_radius_px=MAX_POINT_RADIUS_PX,
            flat_depth=flat_depth,
            feather=feather,
        )

    def _submit(self, pipeline, uniforms, vertices: np.ndarray) -> None:
        if vertices.shape[0]:
            self.gpu.submit(
                gpu3d.DrawCall(
                    pipeline=pipeline,
                    uniforms=uniforms,
                    vertex_count=int(vertices.shape[0]),
                    vertices=vertices,
                )
            )

    def _submit_points(self, pipeline, uniforms, batch: np.ndarray) -> None:
        """One instance per particle, six generated corners each: one draw call."""
        if batch.shape[0]:
            self.gpu.submit(
                gpu3d.DrawCall(
                    pipeline=pipeline,
                    uniforms=uniforms,
                    vertex_count=6,
                    instance_count=int(batch.shape[0]),
                    vertices=batch,
                )
            )

    def _grid_needs_upload(self) -> tuple | None:
        """The grid cache key when the viewport, zoom, pivot or step changed."""
        size = self._framebuffer()
        step = self.camera.grid_step(size[1])
        extent = self.camera.grid_extent()
        key = (size, step, extent, float(self.camera.target[0]), float(self.camera.target[1]))
        return None if key == self.grid_key else key

    def _grid_lines(self) -> np.ndarray:
        size = self._framebuffer()
        return geometry3d.grid_vertices(
            float(self.camera.target[0]),
            float(self.camera.target[1]),
            self.camera.grid_extent(),
            self.camera.grid_step(size[1]),
        )

    def _store_grid(self, key: tuple, vertices: np.ndarray) -> None:
        if self.grid_buffer is not None:
            self.grid_buffer.destroy()
        self.grid_buffer = self.gpu.create_vertex_buffer(vertices)
        self.grid_count = int(vertices.shape[0])
        self.grid_key = key

    def _draw_grid_buffer(self, pipeline, uniforms) -> None:
        if self.grid_buffer is not None and self.grid_count:
            self.gpu.submit(
                gpu3d.DrawCall(
                    pipeline=pipeline,
                    uniforms=uniforms,
                    vertex_count=self.grid_count,
                    buffer=self.grid_buffer,
                )
            )

    # -- pictures (identical in both renderers) ---------------------------
    def draw_image(
        self,
        anchor: geometry3d.Point3,
        source: texture3d.ImageSource,
        right: geometry3d.Vector3 = (1.0, 0.0, 0.0),
        up: geometry3d.Vector3 = (0.0, 1.0, 0.0),
        width: float = 1.0,
        height: float | None = None,
        *,
        tint: texture3d.Tint = (1.0, 1.0, 1.0, 1.0),
        depth_write: bool | None = None,
    ) -> None:
        """Draw a picture (texture) in an arbitrary 3D plane.

        ``anchor`` is the picture's **bottom-left** corner, ``right`` and ``up``
        are its own axes (only their direction matters), and ``width`` /
        ``height`` are its size in world units along those axes. ``height=None``
        keeps the source's pixel aspect ratio from ``width``.

        ``source`` is a file path, a Pillow image, an ``(H, W, 3)`` /
        ``(H, W, 4)`` NumPy array, or an already-uploaded
        ``texture3d.Texture``. Paths and in-memory images are cached by
        ``texture3d.TextureStore``, so the same source drawn every frame uploads
        once (a different array per frame is the exception — build one
        ``Texture`` and reuse it).

        ``tint`` multiplies the texel: an ``(r, g, b)`` colour tints the picture
        and an RGBA fades it. ``depth_write`` defaults to ``True`` for an opaque
        picture and ``False`` when it has transparency, so blended fragments can
        see what is behind them.
        """
        texture = self.textures.get(source)
        if height is None:
            height = width * texture.height / max(texture.width, 1)
        vertices = geometry3d.image_quad_vertices(anchor, right, up, width, height)
        self.images.draw(vertices, texture, self._mvp(), tint, depth_write=depth_write)

    # -- screen-space shapes (identical in both renderers) ----------------
    def _screen_uniforms(self, feather: float) -> np.ndarray:
        """Only the two sizes a screen-space shader needs: no camera, no matrix."""
        width, height = self._framebuffer()
        return gpu3d.pack_uniforms(
            viewport=(float(max(width, 1)), float(max(height, 1))),
            screen=tuple(float(max(v, 1)) for v in self.gpu.window_size()),
            feather=feather,
        )

    def draw_screen_rect(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        radius: float = 0.0,
    ) -> None:
        """Draw a filled rectangle in **screen space**, with optional round corners.

        ``anchor`` is the **top-left** corner ``(x, y)`` in window pixels (origin
        top-left, y down: the convention of ``glfw.get_cursor_pos``), ``size`` is
        ``(width, height)`` in pixels, ``color`` is ``(r, g, b)`` or
        ``(r, g, b, a)`` and ``radius`` rounds the four corners (clamped to half
        of the smaller side, so a huge radius makes a pill).

        ``radius > 0``: the fragment shader evaluates one signed distance
        function and fades the edge over one pixel — smooth corners at any size
        or HiDPI scale, no tessellation.

        ``radius == 0``: a different, minimal program — a solid quad with no
        distance field, no ``discard`` and, when the colour is opaque, no
        blending. Its edges are the rasteriser's own, which are exact for an
        axis-aligned rectangle.

        Either way it is one instance row (9 floats) and one draw call, drawn on
        top of the 3D scene in call order. ``in_screen_rect`` is the matching hit
        test and takes the same ``anchor`` / ``size`` / ``radius``.
        """
        row = screen2d.rect_instance(anchor, size, color, radius)
        if row[0, 8] > 0.0:
            self._submit_points(self.shape_pipeline, self._screen_uniforms(1.0), row)
        else:
            pipeline = self.sharp_rect_pipelines[bool(row[0, 7] < 1.0)]
            self._submit_points(pipeline, self._screen_uniforms(0.0), row)

    def draw_screen_acrylic_rect(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        tint: screen2d.ColorRGBA = (0.10, 0.10, 0.13, 0.55),
        radius: float = 0.0,
        blur: float = 18.0,
    ) -> None:
        """Draw a panel of frosted glass ("acrylic") in screen space.

        Same ``anchor``, ``size`` and ``radius`` as ``draw_screen_rect`` (and
        the same hit test, ``in_screen_rect``). Everything drawn **before** it
        — the 3D scene and the interface under the panel — shows through,
        blurred over ``blur`` pixels. ``tint`` is ``(r, g, b, a)``: the colour
        of the glass and how much of it covers the blur (0 = clear frosted
        glass, 1 = a plain rectangle). What you draw after the panel (its
        text, its buttons) is sharp, on top.

        The blur needs the finished picture behind the panel, so a frame that
        contains a panel is drawn in one more step. ``ui.transparency = False``
        (``--no-transparency``, F3) removes that step: the panel becomes a
        plain, mostly opaque rectangle of the tint colour.
        """
        r, g, b, a = screen2d.rgba(tint)
        if not self.transparency or blur <= 0.0:
            self.draw_screen_rect(anchor, size, (r, g, b, max(a, 0.9)), radius)
            return
        row = screen2d.rect_instance(anchor, size, (r, g, b, a), radius)
        if self.__dict__.get("_acrylic_pipeline") is None:
            self._acrylic_pipeline = self.gpu.pipeline(
                shaders3d.SCREEN_ACRYLIC,
                gpu3d.vertex_layout(*shaders3d.SCREEN_SHAPE_LAYOUT, instanced=True),
                blend=True,
                textured=True,
                depth_write=False,
                depth_compare="always",
            )
        width, height = self._framebuffer()
        window = tuple(float(max(v, 1)) for v in self.gpu.window_size())
        uniforms = gpu3d.pack_uniforms(
            viewport=(float(max(width, 1)), float(max(height, 1))),
            screen=window,
            feather=1.0,
            pixel_scale=float(blur) * min(width / window[0], height / window[1]),
        )
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self._acrylic_pipeline,
                uniforms=uniforms,
                vertex_count=6,
                instance_count=1,
                vertices=row,
                backdrop=True,
                overlay=True,
            )
        )

    def draw_screen_circle(
        self,
        center: screen2d.Point2,
        radius: float,
        color: screen2d.ColorRGBA = ARROW_COLOR,
    ) -> None:
        """Draw a filled, anti-aliased disc in **screen space**.

        ``center`` is ``(x, y)`` and ``radius`` a length, both in window pixels
        (top-left origin, y down); ``color`` is ``(r, g, b)`` or
        ``(r, g, b, a)``. It is the rounded-rectangle distance field with the
        corner radius equal to the half size, so it shares that program: one
        instance row, one draw call, a one-pixel fade on the rim.
        ``in_screen_circle`` is the matching hit test.
        """
        if radius > 0.0:
            row = screen2d.circle_instance(center, radius, color)
            self._submit_points(self.shape_pipeline, self._screen_uniforms(1.0), row)

    # -- borders and the inverted box ----------------------------------------
    def draw_screen_rect_outline(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        radius: float = 0.0,
        thickness: float = 2.0,
    ) -> None:
        """Draw only the border of a rounded rectangle, in **screen space**.

        ``anchor``, ``size``, ``color`` and ``radius`` are those of
        ``draw_screen_rect``; ``thickness`` is the width of the border in
        pixels. The border lies **inside** the rectangle: its outer edge is
        exactly the edge ``draw_screen_rect`` would fill, so an outline drawn
        with the same arguments frames the filled shape without growing it.

        Anti-aliased on both sides by the distance field; the inside stays
        untouched (whatever was drawn before shows through). One instance, one
        draw call. ``in_screen_rect`` is still the hit test of the whole shape.
        """
        shape = screen2d.rect_instance(anchor, size, color, radius)
        if thickness > 0.0 and shape[0, 2] > 0.0 and shape[0, 3] > 0.0:
            row = screen2d.edge_instance(shape, thickness, inverted=False)
            self._submit_points(self.edge_pipeline, self._screen_uniforms(1.0), row)

    def draw_screen_circle_outline(
        self,
        center: screen2d.Point2,
        radius: float,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        thickness: float = 2.0,
    ) -> None:
        """Draw only the border of a circle (a ring), in **screen space**.

        ``center``, ``radius`` and ``color`` are those of ``draw_screen_circle``;
        ``thickness`` is the width of the ring in pixels, measured inwards from
        the circle of that radius.
        """
        if radius > 0.0 and thickness > 0.0:
            shape = screen2d.circle_instance(center, radius, color)
            row = screen2d.edge_instance(shape, thickness, inverted=False)
            self._submit_points(self.edge_pipeline, self._screen_uniforms(1.0), row)

    def draw_screen_inverted_rect(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        radius: float = 0.0,
    ) -> None:
        """Fill everything **outside** a rounded rectangle; leave its inside clear.

        The opposite of ``draw_screen_rect`` with the same arguments: the
        rounded box is a transparent hole and ``color`` (``(r, g, b)`` or
        ``(r, g, b, a)``) covers the rest of the window, up to its edges.

        This is how to make a *frame* (or a mask): draw the content first, as
        large as it likes, then this on top — only what is inside the box stays
        visible, with smooth rounded corners. An opaque colour hides the
        outside completely; an alpha below 1 dims it instead, like a spotlight.

        ``in_screen_rect(cursor, anchor, size, radius)`` tells whether the
        cursor is in the hole, which is the test for "is the user pointing at
        the content?". One instance, one draw call covering the window.
        """
        shape = screen2d.rect_instance(anchor, size, color, radius)
        row = screen2d.edge_instance(shape, 0.0, inverted=True)
        self._submit_points(self.edge_pipeline, self._screen_uniforms(1.0), row)

    # -- gradient shapes ----------------------------------------------------
    MAX_GRADIENTS = 64  # distinct gradients kept on the GPU before the oldest goes

    def _gradient(self, stops, colors) -> texture3d.Texture:
        """The 1D texture of a gradient, built on first use and then reused."""
        key = screen2d.gradient_stops(stops, colors)
        texture = self._gradients.get(key)
        if texture is None:
            if len(self._gradients) >= self.MAX_GRADIENTS:
                # An animated gradient makes a new key every frame: drop the
                # oldest one instead of growing without bound.
                self._gradients.pop(next(iter(self._gradients))).destroy()
            texture = texture3d.Texture(
                screen2d.gradient_pixels(*key), self.gpu, mipmaps=False
            )
            self._gradients[key] = texture
        return texture

    def _submit_gradient(self, row: np.ndarray, texture: texture3d.Texture) -> None:
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self.gradient_pipeline,
                uniforms=self._screen_uniforms(1.0),
                vertex_count=6,
                instance_count=1,
                vertices=row,
                texture_group=texture.bind_group,
            )
        )

    def draw_screen_gradient_rect(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        stops: screen2d.Sequence[float],
        colors: screen2d.Sequence[screen2d.ColorRGBA],
        direction: screen2d.Point2 = screen2d.RIGHT,
        radius: float = 0.0,
    ) -> None:
        """Draw a rounded rectangle filled with a multi-colour gradient.

        ``anchor``, ``size`` and ``radius`` are those of ``draw_screen_rect``
        (top-left corner, ``(width, height)``, corner radius; window pixels).
        The single colour is replaced by three arguments:

        * ``stops`` — positions along the gradient, floats from ``0.0`` (where
          it starts) to ``1.0`` (where it ends), e.g. ``[0.0, 0.5, 1.0]``;
        * ``colors`` — one ``(r, g, b)`` or ``(r, g, b, a)`` per stop: the
          colour at that position. Same length as ``stops``;
        * ``direction`` — an ``(x, y)`` tuple or ``Vec2`` saying which way the
          gradient runs: ``(1, 0)`` left to right (default), ``(0, -1)`` bottom
          to top, ``(1, 1)`` towards the bottom-right corner. Its length does
          not matter.

        Between two stops the colour is interpolated; position 0 is the edge
        (or corner) the direction comes from, position 1 the one it points to.

        Hardware accelerated: the stops are baked once into a 256-texel 1D
        texture, cached for as long as the same gradient is drawn, and the
        fragment shader does one dot product and one texture fetch per pixel —
        the same cost for 2 stops or 20. The edge is anti-aliased by the same
        distance field as ``draw_screen_rect``. One instance, one draw call.

        ``in_screen_rect`` is its hit test, and ``screen2d.gradient_color``
        returns the colour of a gradient at one position.
        """
        row = screen2d.gradient_rect_instance(anchor, size, direction, radius)
        if row[0, 2] > 0.0 and row[0, 3] > 0.0:
            self._submit_gradient(row, self._gradient(stops, colors))

    def draw_screen_gradient_circle(
        self,
        center: screen2d.Point2,
        radius: float,
        stops: screen2d.Sequence[float],
        colors: screen2d.Sequence[screen2d.ColorRGBA],
        direction: screen2d.Point2 = screen2d.RIGHT,
    ) -> None:
        """Draw an anti-aliased disc filled with a multi-colour gradient.

        ``center`` and ``radius`` are those of ``draw_screen_circle``; ``stops``,
        ``colors`` and ``direction`` those of ``draw_screen_gradient_rect``. The
        gradient runs across the diameter, along ``direction``. Same program,
        same cached texture, one draw call. ``in_screen_circle`` is its hit test.
        """
        if radius > 0.0:
            row = screen2d.gradient_circle_instance(center, radius, direction)
            self._submit_gradient(row, self._gradient(stops, colors))

    def draw_screen_triangle(
        self,
        anchor: screen2d.Point2,
        base: float,
        height: float,
        direction: screen2d.Point2 = screen2d.UP,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        *,
        antialias: bool = True,
    ) -> None:
        """Draw a filled isosceles triangle in **screen space**.

        ``anchor`` is the **middle of the base**, ``base`` the width of the
        base and ``height`` the distance from the base to the tip, all in
        window pixels (top-left origin, y down). ``direction`` is where the
        height points, from the base towards the tip: an ``(x, y)`` tuple or a
        ``Vec2`` of any length. Screen y grows down, so ``(0, -1)`` (the
        default) points up and ``(1, 0)`` right. ``color`` is ``(r, g, b)`` or
        ``(r, g, b, a)``.

        ``antialias=True`` (default): the fragment shader measures each pixel's
        signed distance to the triangle and fades the three edges over one
        pixel — smooth at any orientation and HiDPI scale.

        ``antialias=False``: the triangle is drawn as three plain vertices with
        a constant colour — no distance field, no ``discard`` and, when the
        colour is opaque, no blending. The cheapest path; slanted edges show
        the pixel staircase.

        Either way it is one instance row (10 floats) and one draw call, on top
        of the 3D scene in call order. ``in_screen_triangle`` is the matching
        hit test; ``screen2d.triangle_points`` returns the three corners.
        """
        if base <= 0.0 or height == 0.0:
            return
        row = screen2d.triangle_instance(anchor, base, height, direction, color)
        if antialias:
            self._submit_points(self.screen_triangle_pipeline, self._screen_uniforms(1.0), row)
            return
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self.sharp_triangle_pipelines[bool(row[0, 9] < 1.0)],
                uniforms=self._screen_uniforms(0.0),
                vertex_count=3,  # the three corners themselves, not a quad
                instance_count=1,
                vertices=row,
            )
        )

    def draw_screen_line(
        self,
        start: screen2d.Point2,
        end: screen2d.Point2,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        thickness: float = 2.0,
    ) -> None:
        """Draw an anti-aliased straight line in **screen space**.

        ``start`` and ``end`` are ``(x, y)`` in window pixels (top-left origin,
        y down), ``color`` is ``(r, g, b)`` or ``(r, g, b, a)`` and
        ``thickness`` is the full width of the line in pixels. Both ends are
        round, so lines that share a point join cleanly.

        The fragment shader measures each pixel's distance to the segment and
        fades the edge over one pixel: smooth at any angle, any thickness
        (including below one pixel) and any HiDPI scale. One instance row, one
        draw call, drawn on top of the 3D scene in call order.
        ``in_screen_line`` is the matching hit test.
        """
        row = screen2d.line_instance(start, end, color, thickness)
        if row[0, 8] > 0.0:
            self._submit_points(self.screen_line_pipeline, self._screen_uniforms(1.0), row)

    def draw_screen_dashed_line(
        self,
        start: screen2d.Point2,
        end: screen2d.Point2,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        thickness: float = 2.0,
        dash: float = 10.0,
        spacing: float = 6.0,
    ) -> None:
        """Draw an anti-aliased **dashed** line in screen space.

        ``draw_screen_line`` plus two lengths, in pixels: ``dash`` is how long
        each dash is and ``spacing`` the empty space between two dashes. The
        pattern starts with a full dash at ``start`` and is cut cleanly at
        ``end``. Dashes have square ends, so both lengths are exact whatever
        the thickness; ``dash == thickness`` gives a row of squares.

        The dashes are not separate pieces: the whole line is still one
        instance row and one draw call, and the shader decides per pixel
        whether it falls on a dash. A line with a thousand dashes costs the
        same as a solid one. ``spacing <= 0`` (or ``dash <= 0``) draws the
        solid line.
        """
        row = screen2d.line_instance(start, end, color, thickness, dash, spacing)
        if row[0, 8] > 0.0:
            self._submit_points(self.screen_line_pipeline, self._screen_uniforms(1.0), row)

    # -- curves -------------------------------------------------------------
    def draw_screen_polyline(
        self,
        points,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        thickness: float = 2.0,
        closed: bool = False,
    ) -> None:
        """Draw connected anti-aliased segments in screen space.

        ``points`` is a list or array of ``(x, y)`` in window pixels: the line
        goes through them in order, and back to the first when ``closed``.
        ``thickness`` is in pixels, joints are round. Whatever the number of
        points, the whole polyline is one array handed to the graphics card and
        one draw call — use it instead of a loop of ``draw_screen_line``.
        """
        starts, ends = curve3d.segments(np.asarray(points, dtype=np.float32)[:, :2], closed)
        if not len(starts) or thickness <= 0.0:
            return
        rows = np.zeros((len(starts), screen2d.LINE_FLOATS), dtype=np.float32)
        rows[:, 0:2] = starts
        rows[:, 2:4] = ends
        rows[:, 4:8] = screen2d.rgba(color)
        rows[:, 8] = 0.5 * float(thickness)
        self._submit_points(self.screen_line_pipeline, self._screen_uniforms(1.0), rows)

    @staticmethod
    def _curve_points(curve, degree, weights, knots, samples, closed) -> tuple[np.ndarray, bool]:
        """Points along a NURBS given as a ``Nurbs`` object or as control points."""
        if isinstance(curve, curve3d.Nurbs):
            return curve.points(samples), curve.closed
        return curve3d.nurbs_points(curve, degree, weights, knots, samples, closed), closed

    def draw_nurbs(
        self,
        curve,
        color=ARROW_COLOR,
        thickness: float = 0.0,
        *,
        degree: int = 3,
        weights=None,
        knots=None,
        samples: int = 64,
        closed: bool = False,
        pick: bool = False,
    ) -> None:
        """Draw a NURBS curve in the 3D scene.

        ``curve`` is a ``curve3d.Nurbs``, or simply its control points — a
        list of ``(x, y, z)`` — with ``degree``, ``weights``, ``knots`` and
        ``closed`` given here. ``samples`` is how many points are computed
        along the curve: raise it for long or very bent curves. ``thickness``
        is as in ``draw_polyline``.

        ``pick=True`` (with a ``Nurbs`` object) lets the user edit the curve
        with the mouse. Its control points are drawn as dots; a click on one
        makes it active and pops up a gizmo of three arrows. Dragging an arrow
        slides the point along x, y or z; dragging the dot moves it freely,
        facing the camera; a click on the background puts the gizmo away.
        Read ``curve.busy`` to know when the mouse belongs to the curve::

            if not curve.busy:
                app.handle_camera()
            ui.draw_nurbs(curve, PINK, pick=True)
        """
        if pick:
            if not isinstance(curve, curve3d.Nurbs):
                raise TypeError("pick=True needs a curve3d.Nurbs object: it remembers which point is active")
            curve_edit3d.edit(self, curve)  # first: the curve is drawn where the mouse just put it
        points, loop = self._curve_points(curve, degree, weights, knots, samples, closed)
        self.draw_polyline(points, color, thickness, loop)
        if pick:
            curve_edit3d.draw_handles(self, curve)  # last: the dots and the gizmo are on top

    def world_to_screen(self, point) -> tuple[float, float] | None:
        """Where a 3D point appears in the window, in pixels, or ``None`` if it is behind the camera.

        The bridge from world space to screen space: use it to put a label, a
        circle or a hit test on something that lives in the scene.
        """
        spots, visible = curve_edit3d.to_screen(self, point)
        return (float(spots[0, 0]), float(spots[0, 1])) if visible[0] else None

    def draw_screen_nurbs(
        self,
        curve,
        color: screen2d.ColorRGBA = ARROW_COLOR,
        thickness: float = 2.0,
        *,
        degree: int = 3,
        weights=None,
        knots=None,
        samples: int = 64,
        closed: bool = False,
    ) -> None:
        """Draw a NURBS curve in screen space: control points are ``(x, y)`` pixels."""
        points, loop = self._curve_points(curve, degree, weights, knots, samples, closed)
        self.draw_screen_polyline(points, color, thickness, loop)

    def draw_screen_image(
        self,
        anchor: screen2d.Point2,
        size: screen2d.Size2,
        source: texture3d.ImageSource,
        *,
        tint: texture3d.Tint = (1.0, 1.0, 1.0, 1.0),
    ) -> None:
        """Draw a picture in **screen space**: flat, in pixels, on top of the scene.

        ``anchor`` is its **top-left** corner and ``size`` its ``(width,
        height)``, in window pixels like every screen-space method — so
        ``in_screen_rect(cursor, anchor, size)`` is its hit test. The picture is
        stretched to ``size``; ``screen2d.fit`` computes an ``anchor`` / ``size``
        that keeps its proportions inside a box.

        ``source`` is whatever ``draw_image`` accepts: a file path, a Pillow
        image, an ``(H, W, 3 or 4)`` NumPy array (row 0 = top), or a
        ``texture3d.Texture``. For a picture that changes every frame (video),
        keep one ``Texture(first, mipmaps=False)`` and call its ``update(new)``:
        the GPU texture is then reused instead of re-created.

        ``tint`` multiplies the texel; an alpha below 1 fades the picture.
        """
        texture = self.textures.get(source)
        x, y = float(anchor[0]), float(anchor[1])
        width, height = float(size[0]), float(size[1])
        # The 3D picture quad, laid in the pixel plane: its bottom-left corner
        # is (x, y + height) and its "up" is screen-up, which is -y.
        vertices = geometry3d.image_quad_vertices(
            (x, y + height, 0.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0), width, height
        )
        screen_w, screen_h = (float(max(v, 1)) for v in self.gpu.window_size())
        # Pixels -> clip space, in place of the camera matrix. The z row pins
        # the quad to the nearest depth, so it passes the depth test anywhere;
        # it does not write depth, so it hides nothing drawn after it.
        pixels_to_clip = np.array(
            (
                (2.0 / screen_w, 0.0, 0.0, -1.0),
                (0.0, -2.0 / screen_h, 0.0, 1.0),
                (0.0, 0.0, 0.0, -1.0),
                (0.0, 0.0, 0.0, 1.0),
            ),
            dtype=np.float32,
        )
        self.images.draw(vertices, texture, pixels_to_clip, tint, depth_write=False)

    # The hit tests need no GPU and no renderer state; they are here so the
    # question sits next to the draw call it belongs to (see screen2d).
    in_screen_rect = staticmethod(screen2d.in_rect)
    in_screen_circle = staticmethod(screen2d.in_circle)
    in_screen_line = staticmethod(screen2d.in_line)
    in_screen_triangle = staticmethod(screen2d.in_triangle)

    def close(self) -> None:
        self._close_solids()
        for texture in self._gradients.values():
            texture.destroy()
        self._gradients.clear()
        self.textures.close()
        self.images.close()
        if self.grid_buffer is not None:
            self.grid_buffer.destroy()
            self.grid_buffer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GridArrowRenderer(SceneRenderer):
    """Aliased renderer: one-pixel line-list strokes and hard square particles."""

    def __init__(self, gpu: gpu3d.GPU, camera) -> None:
        super().__init__(gpu, camera)
        lines = gpu3d.vertex_layout(*shaders3d.LINE_LAYOUT)
        self.line_pipeline = gpu.pipeline(shaders3d.LINES, lines, topology="line-list")
        self.triangle_pipeline = gpu.pipeline(shaders3d.LINES, lines)
        self.point_pipeline = gpu.pipeline(
            shaders3d.POINTS_PLAIN,
            gpu3d.vertex_layout(*shaders3d.POINT_LAYOUT, instanced=True),
        )
        # The alpha-carrying particle twin: its own 8-float layout and shader,
        # blended, with a depth-write and a no-depth-write variant.
        rgba = gpu3d.vertex_layout(*shaders3d.POINT_RGBA_LAYOUT, instanced=True)
        self.point_rgba_pipelines = {
            depth_write: gpu.pipeline(
                shaders3d.POINTS_PLAIN_RGBA, rgba, blend=True, depth_write=depth_write
            )
            for depth_write in (True, False)
        }

    # -- internals ---------------------------------------------------------
    def _draw(self, vertices) -> None:
        self._submit(self.line_pipeline, self._camera_uniforms(), vertices)

    def _bind_points(self) -> np.ndarray:
        return self._point_uniforms(flat_depth=False, feather=0.0)  # depth-correct

    def _draw_points_rgba(self, batch: np.ndarray, depth_write: bool = True) -> None:
        """One 8-float batch, blended over the scene (blend lives in the pipeline)."""
        self._submit_points(
            self.point_rgba_pipelines[bool(depth_write)], self._bind_points(), batch
        )

    # -- the renderer API shared with renderer_aa3d.AAGridArrowRenderer ----
    def draw_grid(self) -> None:
        key = self._grid_needs_upload()
        if key is not None:
            self._store_grid(key, self._grid_lines())
        self._draw_grid_buffer(self.line_pipeline, self._camera_uniforms())

    def draw_arrow(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
    ) -> None:
        """Draw an arrow from `start` to `tip`; `color` is an (r, g, b) tuple."""
        self._draw(
            geometry3d.arrow_vertices(
                start,
                tip,
                view_dir=self.camera.forward(),
                color=color,
            )
        )

    def draw_line(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw a plain 3D segment.

        ``thickness`` is in world units; ``0`` keeps the thin one-pixel
        line-list path. A positive thickness becomes a camera-facing ribbon,
        because line-list itself cannot change width.
        """
        if thickness > 0.0:
            self._submit(
                self.triangle_pipeline,
                self._camera_uniforms(),
                geometry3d.thick_line_vertices(
                    start,
                    tip,
                    thickness,
                    view_dir=self.camera.forward(),
                    color=color,
                ),
            )
            return
        self._draw(geometry3d.line_vertices(start, tip, color=color))

    def draw_polyline(self, points, color=ARROW_COLOR, thickness: float = 0.0, closed: bool = False) -> None:
        """Draw connected segments through ``points`` (rows of ``(x, y, z)``).

        ``color`` is one ``(r, g, b)`` or one per point. This plain renderer
        draws thin one-pixel lines and ignores ``thickness``; ``app.aa`` draws
        the anti-aliased, thick version of the same call.
        """
        starts, ends = curve3d.segments(np.asarray(points, dtype=np.float32)[:, :3], closed)
        if not len(starts):
            return
        colors = np.asarray(color, dtype=np.float32)
        batch = np.empty((len(starts), 2, 6), dtype=np.float32)
        batch[:, 0, :3] = starts
        batch[:, 1, :3] = ends
        batch[:, :, 3:] = (colors[: len(starts), None, :3] if colors.ndim == 2 else colors[:3])
        self._draw(batch.reshape(-1, 6))

    def draw_circle(
        self,
        center: geometry3d.Point3,
        radius: float,
        color: geometry3d.Color = ARROW_COLOR,
        axis: geometry3d.Vector3 = (0.0, 0.0, 1.0),
        segments: int = 64,
    ) -> None:
        """Draw a circle outline in the plane whose normal is ``axis``."""
        self._draw(
            geometry3d.circle_vertices(
                center,
                radius,
                color=color,
                axis=axis,
                segments=segments,
            )
        )

    def draw_points(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
    ) -> None:
        """Draw a field of particles in one draw call, one vertex per particle.

        ``points`` is an ``(N, 3)`` array-like (or one ``(x, y, z)`` triple),
        ``size`` is the particle **radius in world units** — a scalar or one
        value per point — and ``color`` is one ``(r, g, b)`` or ``(N, 3)``.
        the GPU expands each sprite itself (an instanced quad), so no geometry is tessellated here; the
        plain renderer draws them as hard squares and depth-tests them like any
        other aliased draw. The vertex shader converts the world-space radius to
        pixels per point, clamps it to the driver's sprite limit, and the
        pipeline clips anything behind the eye.

        For per-particle transparency (a life-time fade) use ``draw_points_rgba``,
        which is a separate method on a separate 8-float buffer so this opaque
        path stays exactly as fast as it was.
        """
        batch = geometry3d.point_vertices(points, size, color=color)
        if not batch.shape[0]:
            return
        self._submit_points(self.point_pipeline, self._bind_points(), batch)

    def draw_point(
        self,
        center: geometry3d.Point3,
        size: float,
        color: geometry3d.Color = ARROW_COLOR,
    ) -> None:
        """Draw one particle; ``size`` is the radius in world units."""
        self.draw_points(center, size, color)

    def draw_points_sorted(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
        *,
        order: np.ndarray | None = None,
    ) -> None:
        """Draw particles back-to-front, so the nearest one is painted last.

        Same batch, same upload, same single draw call as ``draw_points``, plus
        one view-depth sort (``OrbitCamera.depth_order``) and one gather of the
        packed vertices. Pass ``order`` to reuse a permutation computed earlier
        — e.g. recompute only when the camera moves.

        The plain renderer already depth-tests its points, so for opaque
        particles this changes nothing visually: sorting here exists so the call
        site works with either renderer and stays correct if you turn blending
        on (transparent particles need painter's order, since the depth buffer
        cannot resolve them). Transparent batches use
        ``draw_points_rgba_sorted``, the same permutation on the alpha path.
        """
        batch = geometry3d.point_vertices(points, size, color=color)
        if not batch.shape[0]:
            return
        if order is None:
            order = self.camera.depth_order(batch[:, :3])
        self._submit_points(self.point_pipeline, self._bind_points(), batch[order])

    # -- particles with alpha: separate buffer, program and methods ---------
    def draw_points_rgba(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.PointColor | np.ndarray = ARROW_COLOR,
        *,
        depth_write: bool = True,
    ) -> None:
        """``draw_points`` with a coverage channel: a per-particle life-time fade.

        Same one-vertex-per-particle batch, except ``color`` may be one
        ``(r, g, b, a)`` or an ``(N, 4)`` array (``(r, g, b)`` is still accepted
        and packs ``a = 1.0``). ``a`` is the particle's coverage and the fragment
        shader writes it straight through, so ``0`` makes the sprite invisible and
        ``1`` is the old opaque square. Driving it towards ``0`` over a particle's
        life is the whole life-time effect.

        This is a separate method on a separate 8-float VAO and program, so the
        opaque ``draw_points`` path keeps its exact old cost. Here the batch is
        one extra float column (32-byte aligned stride), the draw is still one
        upload and one draw call, and blending is switched on around that call
        only (then restored). Fading therefore costs no extra pass and no extra
        draw call.

        ``depth_write`` defaults to ``True``: a nearly invisible particle then
        still writes depth like an opaque one. Pass ``depth_write=False`` to
        composite the batch like glass (the mask is saved and restored), or use
        ``draw_points_rgba_sorted`` when the field has to layer back-to-front.
        """
        batch = geometry3d.point_vertices_rgba(points, size, color=color)
        if not batch.shape[0]:
            return
        self._draw_points_rgba(batch, depth_write)

    def draw_point_rgba(
        self,
        center: geometry3d.Point3,
        size: float,
        color: geometry3d.PointColor = ARROW_COLOR,
        *,
        depth_write: bool = True,
    ) -> None:
        """Draw one particle with coverage; ``size`` is the radius in world units.

        ``color`` may be ``(r, g, b)`` or ``(r, g, b, a)`` (see
        ``draw_points_rgba``).
        """
        self.draw_points_rgba(center, size, color, depth_write=depth_write)

    def draw_points_rgba_sorted(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.PointColor | np.ndarray = ARROW_COLOR,
        *,
        order: np.ndarray | None = None,
        depth_write: bool = True,
    ) -> None:
        """``draw_points_rgba`` painted back-to-front for glass-like layering.

        Same batch, upload and single blended draw call, plus one view-depth sort
        (``OrbitCamera.depth_order``) and one gather of the packed vertices. Pass
        ``order`` to reuse a permutation computed earlier, e.g. only when the
        camera moves — the same contract as ``draw_points_sorted``.
        """
        batch = geometry3d.point_vertices_rgba(points, size, color=color)
        if not batch.shape[0]:
            return
        if order is None:
            order = self.camera.depth_order(batch[:, :3])
        self._draw_points_rgba(batch[order], depth_write)
