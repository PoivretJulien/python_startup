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
import shaders3d
import texture3d
from settings3d import ARROW_COLOR, POINT_MAX_PX, POINT_MIN_PX

# Sprite radius ceiling in pixels. GL clamped the sprite *diameter* to the
# driver's GL_POINT_SIZE_RANGE (POINT_MAX_PX here); instanced quads have no
# driver cap, so the same half-diameter is kept for identical output.
MAX_POINT_RADIUS_PX = 0.5 * POINT_MAX_PX


class SceneRenderer:
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

    def close(self) -> None:
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
