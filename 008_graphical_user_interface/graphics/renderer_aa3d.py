"""GPU anti-aliased 3D renderer: screen-space capsules, one fragment distance field.

Same public API as ``renderer3d.GridArrowRenderer`` — ``draw_grid``,
``draw_arrow``, ``draw_line``, ``draw_circle``, ``draw_text``, ``measure_text``,
``close`` — so the scene can switch one call between the two without changing
anything else.

How the 3D version works:

1. every segment becomes a quad whose vertices carry both endpoints, a side
   sign and its half width in **pixels**;
2. the vertex shader projects both endpoints to screen pixels, offsets the quad
   perpendicular to the projected line, and writes the corner back as clip
   coordinates (multiplying by ``w`` so perspective division lands on the
   intended screen position);
3. the fragment shader measures the exact distance from the fragment to the
   segment in pixels and fades over one pixel: ``coverage = clamp(half_width +
   0.5 - distance, 0, 1)``;
4. segments crossing the camera's near plane are clipped in homogeneous clip
   space, so zooming into the scene does not produce projected streaks.

No MSAA, no supersampled framebuffer, no post-process pass. The shaders are the
WGSL ports in ``shaders3d`` (``STROKES``, ``SCREEN_STROKES``, ``POINTS_AA*``),
recorded through ``gpu3d`` and replayed in one wgpu render pass per frame. Because widths are
stated in pixels, strokes keep their on-screen weight at any orbit, zoom, window
size or HiDPI scale.
"""

from __future__ import annotations

import font3d
import geometry3d
import geometry_aa3d
import gpu3d
import numpy as np
import shaders3d
from camera3d import OrbitCamera
from renderer3d import SceneRenderer
from settings3d import (
    AA_ARROW_WIDTH_PX,
    AA_CIRCLE_WIDTH_PX,
    AA_FEATHER_PX,
    AA_GRID_WIDTH_PX,
    AA_LINE_WIDTH_PX,
    AA_TEXT_WIDTH_PX,
    ARROW_COLOR,
)


class AAGridArrowRenderer(SceneRenderer):
    """GPU anti-aliased 3D renderer; the GPU supplies the surface, the camera the view."""

    def __init__(self, gpu: gpu3d.GPU, camera: OrbitCamera) -> None:
        super().__init__(gpu, camera)
        # Coverage arrives as alpha, so every AA pipeline blends (the GL version
        # enabled GL_BLEND once, globally). Depth test stays LEQUAL so coplanar
        # strokes drawn later (the clock over the grid) win the depth fight.
        strokes = gpu3d.vertex_layout(*shaders3d.STROKE_LAYOUT)
        self.stroke_pipeline = gpu.pipeline(shaders3d.STROKES, strokes, blend=True)
        # Screen-space text: same distance-field fragment shader, pixel input,
        # never depth-tested — the HUD always sits on top of the scene.
        self.screen_pipeline = gpu.pipeline(
            shaders3d.SCREEN_STROKES,
            strokes,
            blend=True,
            depth_write=False,
            depth_compare="always",
        )
        # Particles: feathered discs from a distance field, one instance each.
        self.point_pipeline = gpu.pipeline(
            shaders3d.POINTS_AA,
            gpu3d.vertex_layout(*shaders3d.POINT_LAYOUT, instanced=True),
            blend=True,
        )
        rgba = gpu3d.vertex_layout(*shaders3d.POINT_RGBA_LAYOUT, instanced=True)
        self.point_rgba_pipelines = {
            depth_write: gpu.pipeline(
                shaders3d.POINTS_AA_RGBA, rgba, blend=True, depth_write=depth_write
            )
            for depth_write in (True, False)
        }

    # -- internals ---------------------------------------------------------
    def _stroke_uniforms(self) -> np.ndarray:
        return self._camera_uniforms(feather=AA_FEATHER_PX)

    def _draw_strokes(self, vertices, half_width_px) -> None:
        self._submit(
            self.stroke_pipeline,
            self._stroke_uniforms(),
            geometry_aa3d.stroke_vertices(vertices, half_width_px),
        )

    def _bind_points(self) -> np.ndarray:
        # AA strokes are screen-space quads pinned to the middle depth layer and
        # the grid writes that depth everywhere it covers. Sprites use the same
        # layer, so particles layer with the strokes in draw order.
        return self._point_uniforms(flat_depth=True, feather=AA_FEATHER_PX)

    def _draw_points_rgba(self, batch: np.ndarray, depth_write: bool = True) -> None:
        """One 8-float batch; blending and the depth mask live in the pipeline."""
        self._submit_points(
            self.point_rgba_pipelines[bool(depth_write)], self._bind_points(), batch
        )

    # -- the renderer API shared with renderer3d.GridArrowRenderer ---------
    def draw_grid(self) -> None:
        key = self._grid_needs_upload()
        if key is not None:
            self._store_grid(
                key, geometry_aa3d.stroke_vertices(self._grid_lines(), AA_GRID_WIDTH_PX * 0.5)
            )
        self._draw_grid_buffer(self.stroke_pipeline, self._stroke_uniforms())

    def draw_arrow(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
    ) -> None:
        """Draw an arrow from `start` to `tip`; `color` is an (r, g, b) tuple."""
        self._draw_strokes(
            geometry3d.arrow_vertices(
                start,
                tip,
                view_dir=self.camera.forward(),
                color=color,
            ),
            AA_ARROW_WIDTH_PX * 0.5,
        )

    def draw_line(
        self,
        start: geometry3d.Point3,
        tip: geometry3d.Point3,
        color: geometry3d.Color = ARROW_COLOR,
        thickness: float = 0.0,
    ) -> None:
        """Draw an anti-aliased segment with rounded caps.

        ``thickness`` is in world units, exactly like the 2D API; ``0`` falls
        back to ``AA_LINE_WIDTH_PX``. The world thickness is converted to a
        pixel half width at the segment midpoint, so a marker keeps the same
        apparent weight as the rest of a 3D scene.
        """
        vertices = geometry3d.line_vertices(start, tip, color=color)
        if thickness <= 0.0:
            half_width = AA_LINE_WIDTH_PX * 0.5
        else:
            half_width = geometry_aa3d.world_thickness_to_pixels(
                vertices,
                thickness,
                self.camera,
                self._framebuffer()[1],
            )
        self._draw_strokes(vertices, half_width)

    def draw_circle(
        self,
        center: geometry3d.Point3,
        radius: float,
        color: geometry3d.Color = ARROW_COLOR,
        axis: geometry3d.Vector3 = (0.0, 0.0, 1.0),
        segments: int = 64,
    ) -> None:
        """Draw a circle in the plane whose normal is ``axis``.

        The outline is the usual ring of segments, but each one is a capsule, so
        the joints are round and the shader fades the rim. ``axis`` is the one
        piece of information a 3D circle needs that a 2D one did not.
        """
        self._draw_strokes(
            geometry3d.circle_vertices(
                center,
                radius,
                color=color,
                axis=axis,
                segments=segments,
            ),
            AA_CIRCLE_WIDTH_PX * 0.5,
        )

    def draw_points(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
    ) -> None:
        """Draw a field of anti-aliased particles in one draw call.

        ``points`` is an ``(N, 3)`` array-like (or one ``(x, y, z)`` triple),
        ``size`` is the particle **radius in world units** — a scalar or one
        value per point — and ``color`` is one ``(r, g, b)`` or ``(N, 3)``.
        Every particle is a single vertex: the sprite is expanded by the GPU and
        its rim is a distance field over the instanced quad, so it stays round and
        feathered at any radius. The vertex shader converts the world-space
        radius to pixels and clamps it to the driver's sprite limit; points
        behind the eye are clipped by the pipeline.

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
        """Draw one anti-aliased particle; ``size`` is the radius in world units."""
        self.draw_points(center, size, color)

    def draw_points_sorted(
        self,
        points: geometry3d.Points3,
        size: float | np.ndarray,
        color: geometry3d.Color | np.ndarray = ARROW_COLOR,
        *,
        order: np.ndarray | None = None,
    ) -> None:
        """Draw particles back-to-front, so the nearest one covers the rest.

        The AA sprites are screen-space quads pinned to ``NDC z = 0``, so the
        depth buffer cannot resolve them: **draw order is the layering**. Sorting
        by view depth (``OrbitCamera.depth_order``) makes the nearest particle
        win every overlap, which is what makes a dense cloud read as 3D.

        Same batch, upload and single draw call as ``draw_points``, plus one
        ``O(N log N)`` argsort and one gather of the packed vertices. If the
        field is static that is wasted work every frame: compute the order when
        the camera moves and pass it back in as ``order=``.
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
        and packs ``a = 1.0``). ``a`` is the particle's coverage: the fragment
        shader multiplies it into the disc's anti-aliased coverage, so ``0`` makes
        the particle invisible and ``1`` is the old opaque disc. Driving it
        towards ``0`` over a particle's life is the whole life-time effect.

        This is a separate method on a separate 8-float VAO and program, so the
        opaque ``draw_points`` path keeps its exact old cost. Here the batch is
        one extra float column (32-byte aligned stride) and the fade costs one
        more multiply in a shader that already runs per fragment: still one
        upload, one draw call and no additional state change (blending is already
        enabled for the AA renderer).

        ``depth_write`` defaults to ``True``. The AA sprites are pinned to
        ``NDC z = 0`` anyway, so the depth buffer already cannot order them: draw
        order is the layering, which is what ``draw_points_rgba_sorted`` exists
        for. Pass ``depth_write=False`` if a faded batch must not leave
        footprints for the depth-tested geometry behind it.
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
        """Draw one anti-aliased particle with coverage; ``size`` is the radius.

        ``size`` is in world units; ``color`` may be ``(r, g, b)`` or
        ``(r, g, b, a)`` (see ``draw_points_rgba``).
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
        """``draw_points_rgba`` painted back-to-front, the layering a fade needs.

        Faded particles are transparent, and the AA sprites are pinned to ``NDC
        z = 0``, so draw order *is* the layering: sorting on view depth
        (``OrbitCamera.depth_order``) makes the nearest particle composite last.
        Same batch, upload and one blended draw call as ``draw_points_rgba``, plus
        one ``O(N log N)`` argsort and one gather; pass ``order`` to reuse a
        permutation computed earlier.
        """
        batch = geometry3d.point_vertices_rgba(points, size, color=color)
        if not batch.shape[0]:
            return
        if order is None:
            order = self.camera.depth_order(batch[:, :3])
        self._draw_points_rgba(batch[order], depth_write)

    def draw_text(
        self,
        anchor: geometry3d.Point3,
        text: str,
        size: float = 0.30,
        color: geometry3d.Color = ARROW_COLOR,
        *,
        right: geometry3d.Vector3 = (1.0, 0.0, 0.0),
        up: geometry3d.Vector3 = (0.0, 1.0, 0.0),
        tracking: float = font3d.TRACKING,
        width_px: float = AA_TEXT_WIDTH_PX,
    ) -> float:
        """Draw anti-aliased `text` starting at `anchor` (left end of the baseline).

        ``right`` and ``up`` orient the text plane; the defaults are the xy plane
        (the 2D behaviour). ``size`` is the cap height in world units, so text
        scales with the scene. Returns the advance width in world units, handy
        for centring a label or anchoring something after the string.
        """
        vertices = geometry_aa3d.text_vertices(
            anchor,
            text,
            size,
            color=tuple(color[:3]),
            right=right,
            up=up,
            tracking=tracking,
        )
        if vertices.shape[0]:
            self._draw_strokes(vertices, width_px * 0.5)
        return geometry_aa3d.text_width(text, size, tracking)

    def measure_text(
        self,
        text: str,
        size: float = 0.30,
        tracking: float = font3d.TRACKING,
    ) -> float:
        """Advance width of world-space `text` in world units, for layout."""
        return geometry_aa3d.text_width(text, size, tracking)

    def draw_screen_text(
        self,
        x: float,
        y: float,
        text: str,
        size_px: float = 16.0,
        color: geometry3d.Color = ARROW_COLOR,
        *,
        tracking: float = font3d.TRACKING,
        width_px: float = AA_TEXT_WIDTH_PX,
    ) -> float:
        """Draw anti-aliased text in **screen space**, on top of the 3D scene.

        ``(x, y)`` is the left end of the baseline in window pixels, with the
        origin at the **top-left** and y growing down (the same convention as
        ``glfw.get_cursor_pos``). ``size_px`` is the cap height in pixels, so the
        text keeps a fixed on-screen size regardless of the camera. ``color`` is
        the usual ``(r, g, b)`` tuple; unlike world-space text it is not
        foreshortened, rotated or depth-tested.

        Returns the advance width in pixels, so a caller can right-align or
        anchor something after the string. Text is AA-only, like ``draw_text``.
        """
        vertices = geometry_aa3d.text_vertices(
            (x, y, 0.0),
            text,
            size_px,
            color=tuple(color[:3]),  # strokes carry rgb only: a 4th value is dropped
            right=(1.0, 0.0, 0.0),
            up=(0.0, -1.0, 0.0),  # screen y grows down
            tracking=tracking,
        )
        if vertices.shape[0]:
            self._submit(
                self.screen_pipeline,
                self._camera_uniforms(feather=AA_FEATHER_PX),
                geometry_aa3d.stroke_vertices(vertices, width_px * 0.5),
            )

        return self.measure_screen_text(text, size_px, tracking)

    def measure_screen_text(
        self,
        text: str,
        size_px: float = 16.0,
        tracking: float = font3d.TRACKING,
    ) -> float:
        """Advance width of screen-space `text` in window pixels."""
        return geometry_aa3d.text_width(text, size_px, tracking)
