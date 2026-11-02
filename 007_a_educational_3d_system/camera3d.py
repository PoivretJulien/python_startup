"""Orbit / pan / zoom camera for the 3D experiment.

This is step06's ``glcore.OrbitCamera`` reduced to its essentials and wired to
raw GLFW, so the experiment keeps the same plain event loop as
``experimental/main.py`` instead of importing the tutorial's window wrapper.

Three controls move three different things:

* ``yaw`` / ``pitch``  where the camera sits on a sphere around ``target``
* ``target``           the pivot; middle-drag panning slides it across the screen
* ``distance``         how far out on that sphere; the wheel zooms it

Left drag orbits, middle drag pans, the wheel zooms and Home/R resets the view —
exactly the navigation step06 teaches. The camera also exposes the inverse
mapping the demo needs: a screen pixel becomes a ray, and the ray meets the
floor plane ``z = 0``, which is how the cursor arrow tip is found in 3D.

The world is **z-up** (the floor is the xy plane), unlike step06's y-up scene;
basis and eye are rotated accordingly.
"""

from __future__ import annotations

import math

import glfw
import numpy as np
from geometry3d import grid_line_count, next_nice_step, nice_step_at_most
from math3d import look_at, mul, perspective
from settings3d import (
    CAMERA_FAR,
    CAMERA_FOV,
    CAMERA_MAX_DISTANCE,
    CAMERA_MAX_PITCH,
    CAMERA_MIN_DISTANCE,
    CAMERA_NEAR,
    CAMERA_ORBIT_SPEED,
    CAMERA_PAN_SPEED,
    CAMERA_START_DISTANCE,
    CAMERA_START_PITCH,
    CAMERA_START_TARGET,
    CAMERA_START_YAW,
    CAMERA_ZOOM_SPEED,
    GRID_EXTENT_FACTOR,
    GRID_MAX_LINES,
    GRID_MAX_SPACING_PX,
    HALF,
)


class OrbitCamera:
    """Orbit the pivot, pan it, zoom out from it — the step06 camera."""

    def __init__(
        self,
        yaw: float = CAMERA_START_YAW,
        pitch: float = CAMERA_START_PITCH,
        distance: float = CAMERA_START_DISTANCE,
        target: tuple[float, float, float] = CAMERA_START_TARGET,
        *,
        min_distance: float = CAMERA_MIN_DISTANCE,
        max_distance: float = CAMERA_MAX_DISTANCE,
        max_pitch: float = CAMERA_MAX_PITCH,
        orbit_speed: float = CAMERA_ORBIT_SPEED,
        zoom_speed: float = CAMERA_ZOOM_SPEED,
        pan_speed: float = CAMERA_PAN_SPEED,
        fov: float = CAMERA_FOV,
        near: float = CAMERA_NEAR,
        far: float = CAMERA_FAR,
    ) -> None:
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.distance = float(distance)
        self.target = np.asarray(target, dtype=np.float64)
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.max_pitch = max_pitch
        self.orbit_speed = orbit_speed
        self.zoom_speed = zoom_speed
        self.pan_speed = pan_speed
        self.fov = fov
        self.near = near
        self.far = far
        self.changed = False

        self._home = (self.yaw, self.pitch, self.distance, self.target.copy())
        self._last_cursor: tuple[float, float] | None = None
        self._scroll = 0.0

    # -- input ------------------------------------------------------------
    def scroll_by(self, dy: float) -> None:
        """Accumulate wheel scroll from the GLFW callback; consumed per frame."""
        self._scroll += dy

    def handle(self, window) -> bool:
        """Apply one frame of left-drag orbit, middle-drag pan and wheel zoom."""
        self.changed = False

        cursor = glfw.get_cursor_pos(window)
        if self._last_cursor is None:
            self._last_cursor = cursor
        dx = cursor[0] - self._last_cursor[0]
        dy = cursor[1] - self._last_cursor[1]
        self._last_cursor = cursor

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            self.orbit(dx, dy)

        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS:
            self.pan(dx, dy)

        if self._scroll:
            self.distance = float(
                np.clip(
                    self.distance - self._scroll * self.zoom_speed,
                    self.min_distance,
                    self.max_distance,
                )
            )
            self._scroll = 0.0
            self.changed = True

        return self.changed

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * self.orbit_speed
        self.pitch = float(
            np.clip(self.pitch - dy * self.orbit_speed, -self.max_pitch, self.max_pitch)
        )
        self.changed = True

    def pan(self, dx: float, dy: float) -> None:
        """Slide the pivot across the camera plane, scaled by the distance."""
        right, up, _forward = self.basis()
        self.target = self.target + (-right * dx + up * dy) * self.distance * self.pan_speed
        self.changed = True

    def reset(self) -> None:
        self.yaw, self.pitch, self.distance, home_target = self._home
        self.target = home_target.copy()
        self.changed = True

    # -- derived ----------------------------------------------------------
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(right, up, forward)`` for the current yaw/pitch (z-up world)."""
        cos_pitch = math.cos(self.pitch)
        forward = np.array(
            (
                -cos_pitch * math.cos(self.yaw),
                -cos_pitch * math.sin(self.yaw),
                -math.sin(self.pitch),
            ),
            dtype=np.float64,
        )
        right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
        right /= max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, forward)
        return right, up, forward

    def forward(self) -> np.ndarray:
        """Unit eye -> target direction, used to make arrow heads face the camera."""
        return self.basis()[2]

    def eye(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        return self.target + self.distance * np.array(
            (
                cos_pitch * math.cos(self.yaw),
                cos_pitch * math.sin(self.yaw),
                math.sin(self.pitch),
            ),
            dtype=np.float64,
        )

    def view(self) -> np.ndarray:
        return look_at(self.eye(), self.target, up=(0.0, 0.0, 1.0))

    def projection(self, aspect: float) -> np.ndarray:
        return perspective(self.fov, max(aspect, 1e-6), self.near, self.far)

    def mvp(self, aspect: float) -> np.ndarray:
        """Projection * view; the scene is already in world coordinates."""
        return mul(self.projection(aspect), self.view())

    # -- screen <-> world -------------------------------------------------
    def screen_ray(
        self,
        px: float,
        py: float,
        width: float,
        height: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cursor in window pixels -> ``(point_on_near_plane, direction)``.

        The inverse of projection * view. Using window coordinates is safe on
        HiDPI because the aspect ratio is identical in points and pixels.
        """
        aspect = max(width, 1.0) / max(height, 1.0)
        inverse = np.linalg.inv(self.projection(aspect) @ self.view())
        ndc_x = 2.0 * px / max(width, 1.0) - 1.0
        ndc_y = 1.0 - 2.0 * py / max(height, 1.0)

        near_h = inverse @ np.array((ndc_x, ndc_y, -1.0, 1.0))
        far_h = inverse @ np.array((ndc_x, ndc_y, 1.0, 1.0))
        near = near_h[:3] / max(near_h[3], 1e-12)
        far = far_h[:3] / max(far_h[3], 1e-12)
        direction = far - near
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        return near, direction

    def floor_point(
        self,
        px: float,
        py: float,
        width: float,
        height: float,
        plane_z: float = 0.0,
    ) -> np.ndarray | None:
        """Where the cursor ray meets the floor plane ``z = plane_z``.

        Returns ``None`` when the ray is parallel to the floor or points away
        from it, so the caller can keep the previous tip instead of jumping.
        """
        origin, direction = self.screen_ray(px, py, width, height)
        if abs(float(direction[2])) < 1e-9:
            return None
        t = (plane_z - origin[2]) / direction[2]
        if t <= 0.0:
            return None
        return origin + direction * t

    def pixels_per_world_unit(self, point, height_px: float) -> float:
        """Screen pixels covered by one world unit at ``point`` (perpendicular)."""
        depth = float((np.asarray(point, dtype=np.float64) - self.eye()) @ self.forward())
        if depth <= 1e-6:
            return 0.0
        return height_px / (2.0 * math.tan(math.radians(self.fov) * 0.5) * depth)

    # -- grid -------------------------------------------------------------
    def grid_extent(self) -> float:
        """Floor half-size around the pivot; grows as the camera zooms out."""
        return max(HALF, GRID_EXTENT_FACTOR * self.distance)

    def grid_step(self, height_px: float) -> float:
        """Human 1/2/5 x 10^k spacing whose squares stay inside the pixel budget.

        The perspective scale is sampled at the pivot: ``distance`` is the
        depth of the target, which is where the user is looking. The result is
        coarsened until the target-centred grid fits ``GRID_MAX_LINES``.
        """
        depth = max(self.distance, self.near)
        pixels_per_unit = height_px / (2.0 * math.tan(math.radians(self.fov) * 0.5) * depth)
        step = nice_step_at_most(GRID_MAX_SPACING_PX / max(pixels_per_unit, 1e-9))
        extent = self.grid_extent()
        while (
            grid_line_count(float(self.target[0]), float(self.target[1]), extent, step)
            > GRID_MAX_LINES
        ):
            step = next_nice_step(step)
        return step

    # -- readout ----------------------------------------------------------
    def as_tuple(self) -> tuple[float, float, float]:
        return self.yaw, self.pitch, self.distance

    def __enter__(self) -> OrbitCamera:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None
