"""Solid 3D shapes: the geometry, the ``.obj`` loader and the placement maths.

Pure NumPy, no GPU: ``solid_renderer3d`` turns what is here into draw calls.

A solid is a list of triangles. Each corner is six floats — a position and a
*normal*, the direction the surface faces there, which is what the light needs::

    x, y, z,  nx, ny, nz

Three things live here:

* ``Mesh`` — triangles from your own arrays or from a Wavefront ``.obj`` file
  (``Mesh.load("ship.obj")``);
* the unit shapes behind ``draw_sphere``, ``draw_box`` ... (built once, then
  stretched and placed by a matrix);
* the small matrices that place a shape in the world (z is up, as everywhere
  in this project).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

Point3 = Sequence[float]

# The kinds of solid the ray-marched mode knows (see solid_shaders3d.sd_shape).
SPHERE, BOX, CYLINDER, CONE, CAPSULE, TORUS = range(6)


# ---------------------------------------------------------------- placement
def rotation(degrees: Point3) -> np.ndarray:
    """A 3x3 rotation from three angles in degrees: around x, then y, then z."""
    rx, ry, rz = (math.radians(float(a)) for a in degrees)
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)), dtype=np.float64)
    y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float64)
    z = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)), dtype=np.float64)
    return z @ y @ x


def frame_along(axis: Point3) -> np.ndarray:
    """A 3x3 rotation whose z column is ``axis``: how a cylinder is aimed."""
    z = np.asarray(axis, dtype=np.float64)
    length = float(np.linalg.norm(z))
    z = z / length if length > 1e-12 else np.array((0.0, 0.0, 1.0))
    helper = np.array((1.0, 0.0, 0.0)) if abs(z[0]) < 0.9 else np.array((0.0, 1.0, 0.0))
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack((x, y, z), axis=1)


def model_matrix(center: Point3, frame: np.ndarray, scale: Point3) -> np.ndarray:
    """4x4 ``translate * rotate * scale`` (row-major, column vectors)."""
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = frame * np.asarray(scale, dtype=np.float64)[None, :]
    m[:3, 3] = center[:3]
    return m


def solid_row(kind: int, params: Point3, center: Point3, frame: np.ndarray) -> np.ndarray:
    """One solid as the ray marcher reads it: 16 floats.

    ``kind, p0, p1, p2`` then the three axes of the solid (the columns of
    ``frame``), each followed by one coordinate of its centre.
    """
    row = np.zeros(16, dtype=np.float32)
    row[0] = kind
    row[1 : 1 + len(params)] = params
    for i in range(3):
        row[4 + 4 * i : 7 + 4 * i] = frame[:, i]
        row[7 + 4 * i] = center[i]
    return row


def half_bounds(kind: int, params: Point3) -> tuple[float, float, float]:
    """Half the size of the box that contains a solid, in the solid's own axes."""
    a, b = float(params[0]), float(params[1]) if len(params) > 1 else 0.0
    if kind == SPHERE:
        return (a, a, a)
    if kind == BOX:
        return (a, b, float(params[2]))
    if kind in (CYLINDER, CONE):
        return (a, a, b)
    if kind == CAPSULE:
        return (a, a, b + a)
    return (a + b, a + b, b)  # torus: big radius a, tube radius b


# ---------------------------------------------------------------- unit shapes
def _interleave(positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.concatenate((positions.reshape(-1, 3), normals.reshape(-1, 3)), axis=1), dtype=np.float32
    )


def _grid(positions: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Two triangles per cell of a ``(rows, columns, 3)`` sheet of points."""
    def cells(a: np.ndarray) -> np.ndarray:
        p00, p01, p10, p11 = a[:-1, :-1], a[:-1, 1:], a[1:, :-1], a[1:, 1:]
        return np.stack((p00, p10, p11, p00, p11, p01), axis=2).reshape(-1, 3)

    return _interleave(cells(positions), cells(normals))


def _disc(z: float, facing: float, segments: int) -> np.ndarray:
    """A flat cap of radius 1 at height ``z``, as a fan of triangles."""
    angle = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    rim = np.stack((np.cos(angle), np.sin(angle), np.full_like(angle, z)), axis=1)
    centre = np.broadcast_to(np.array((0.0, 0.0, z)), (segments, 3))
    positions = np.stack((centre, rim[:-1], rim[1:]), axis=1)
    normals = np.broadcast_to(np.array((0.0, 0.0, facing)), positions.shape)
    return _interleave(positions, normals)


@functools.cache
def unit_sphere(segments: int = 48, rings: int = 24) -> np.ndarray:
    """Radius 1, centred on the origin."""
    polar = np.linspace(0.0, math.pi, rings + 1)[:, None]
    around = np.linspace(0.0, 2.0 * math.pi, segments + 1)[None, :]
    points = np.stack(
        (np.sin(polar) * np.cos(around), np.sin(polar) * np.sin(around), np.cos(polar) * np.ones_like(around)),
        axis=2,
    )
    return _grid(points, points)


@functools.cache
def unit_box() -> np.ndarray:
    """From -1 to +1 on every axis; each face has its own flat normal."""
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            u, v = (axis + 1) % 3, (axis + 2) % 3
            corners = np.zeros((4, 3))
            corners[:, axis] = sign
            # Counter-clockwise seen from outside, so "front" means "outside".
            corners[:, u] = (-1, 1, 1, -1) if sign > 0 else (-1, -1, 1, 1)
            corners[:, v] = (-1, -1, 1, 1) if sign > 0 else (-1, 1, 1, -1)
            normal = np.zeros(3)
            normal[axis] = sign
            quad = corners[[0, 1, 2, 0, 2, 3]]
            faces.append(_interleave(quad, np.broadcast_to(normal, quad.shape)))
    return np.concatenate(faces)


@functools.cache
def unit_cylinder(segments: int = 48) -> np.ndarray:
    """Radius 1 around the z axis, from z = -1 to z = +1, with both caps."""
    around = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    ring = np.stack((np.cos(around), np.sin(around), np.zeros_like(around)), axis=1)
    points = np.stack((ring + (0, 0, -1), ring + (0, 0, 1)))
    side = _grid(points, np.stack((ring, ring)))
    return np.concatenate((side, _disc(-1.0, -1.0, segments), _disc(1.0, 1.0, segments)))


@functools.cache
def unit_cone(segments: int = 48) -> np.ndarray:
    """Base of radius 1 at z = -1, tip at z = +1."""
    around = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    ring = np.stack((np.cos(around), np.sin(around), np.zeros_like(around)), axis=1)
    tip = np.zeros_like(ring) + (0, 0, 1)
    # The slope rises 2 for a run of 1, so the normal leans up by 1 for 2 out.
    slope = (ring * 2.0 + (0, 0, 1)) / math.sqrt(5.0)
    side = _grid(np.stack((ring + (0, 0, -1), tip)), np.stack((slope, slope)))
    return np.concatenate((side, _disc(-1.0, -1.0, segments)))


@functools.lru_cache(maxsize=32)
def unit_torus(tube: float, segments: int = 48, sides: int = 24) -> np.ndarray:
    """A ring of radius 1 around the z axis, made of a tube of radius ``tube``."""
    around = np.linspace(0.0, 2.0 * math.pi, segments + 1)[:, None]
    section = np.linspace(0.0, 2.0 * math.pi, sides + 1)[None, :]
    out = np.stack((np.cos(around) * np.ones_like(section), np.sin(around) * np.ones_like(section),
                    np.zeros_like(around) * section), axis=2)
    up = np.array((0.0, 0.0, 1.0))
    normals = out * np.cos(section)[..., None] + up * np.sin(section)[..., None]
    return _grid(out + tube * normals, normals)


# ----------------------------------------------------------------------- mesh
class Mesh:
    """A solid made of triangles, ready for ``draw_mesh``.

    Make one from a file — ``Mesh.load("model.obj")`` — or from your own
    points: ``Mesh(positions)`` where ``positions`` holds three ``(x, y, z)``
    rows per triangle. ``normals`` (one per point) are computed when not given:
    ``smooth=True`` blends them across neighbouring triangles (round look),
    ``smooth=False`` keeps every triangle flat (faceted look).

    Read-only facts about the shape: ``triangle_count``, ``minimum`` and
    ``maximum`` (two corners of the box around it), ``center``, ``size``.

    The triangles are sent to the graphics card the first time the mesh is
    drawn and stay there; ``destroy()`` releases them.
    """

    def __init__(self, positions, normals=None, *, smooth: bool = True) -> None:
        positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if positions.shape[0] % 3:
            raise ValueError("a mesh needs three points per triangle")
        if normals is None:
            normals = compute_normals(positions, smooth=smooth)
        normals = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if normals.shape != positions.shape:
            raise ValueError("a mesh needs one normal per point")
        self.vertices = _interleave(positions, normals)
        self.triangle_count = positions.shape[0] // 3
        if positions.shape[0]:
            self.minimum = positions.min(axis=0)
            self.maximum = positions.max(axis=0)
        else:
            self.minimum = self.maximum = np.zeros(3, dtype=np.float32)
        self.center = 0.5 * (self.minimum + self.maximum)
        self.size = self.maximum - self.minimum
        self._buffer = None  # the GPU copy, made by the renderer on first draw

    @classmethod
    def load(cls, path, *, smooth: bool = True) -> Mesh:
        """Read a Wavefront ``.obj`` file. See :func:`load_obj`."""
        return load_obj(path, smooth=smooth)

    def fitted(self, size: float = 1.0) -> Mesh:
        """A copy centred on the origin whose longest side measures ``size``.

        Files come in any unit and any position; this gives a shape you can
        place and scale without knowing how it was modelled.
        """
        longest = float(self.size.max()) or 1.0
        positions = (self.vertices[:, :3] - self.center) * (float(size) / longest)
        return Mesh(positions, self.vertices[:, 3:])

    def destroy(self) -> None:
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None


def compute_normals(positions: np.ndarray, *, smooth: bool = True) -> np.ndarray:
    """One normal per point of a triangle list (three points per triangle)."""
    triangles = positions.reshape(-1, 3, 3).astype(np.float64)
    # The cross product of two edges faces away from the triangle; its length
    # is twice the area, so big triangles weigh more in the smooth blend.
    face = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals = np.repeat(face, 3, axis=0)
    if smooth and len(normals):
        _unique, index = np.unique(np.round(positions, 5), axis=0, return_inverse=True)
        index = index.reshape(-1)
        total = np.zeros((index.max() + 1, 3))
        np.add.at(total, index, normals)
        normals = total[index]
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return (normals / np.maximum(length, 1e-20)).astype(np.float32)


def load_obj(path, *, smooth: bool = True) -> Mesh:
    """Read the triangles of a Wavefront ``.obj`` file into a :class:`Mesh`.

    Understood: ``v`` (points), ``vn`` (normals) and ``f`` (faces, with any
    number of corners, in the ``7``, ``7/1``, ``7//3`` and ``7/1/3`` forms,
    negative numbers included). Everything else — materials, texture
    coordinates, groups, curves — is skipped. When the file has no normals,
    they are computed (``smooth`` as in :class:`Mesh`).
    """
    points: list[tuple[float, float, float]] = []
    file_normals: list[tuple[float, float, float]] = []
    corners_p: list[int] = []
    corners_n: list[int] = []
    has_normals = True

    with Path(path).open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            if line.startswith("v "):
                x, y, z = line.split()[1:4]
                points.append((float(x), float(y), float(z)))
            elif line.startswith("vn "):
                x, y, z = line.split()[1:4]
                file_normals.append((float(x), float(y), float(z)))
            elif line.startswith("f "):
                face_p, face_n = [], []
                for corner in line.split()[1:]:
                    parts = corner.split("/")
                    p = int(parts[0])
                    face_p.append(p - 1 if p > 0 else len(points) + p)
                    if len(parts) > 2 and parts[2]:
                        n = int(parts[2])
                        face_n.append(n - 1 if n > 0 else len(file_normals) + n)
                    else:
                        has_normals = False
                # A face with more than three corners is cut into a fan.
                for i in range(1, len(face_p) - 1):
                    corners_p += (face_p[0], face_p[i], face_p[i + 1])
                    if len(face_n) == len(face_p):
                        corners_n += (face_n[0], face_n[i], face_n[i + 1])

    if not corners_p:
        raise ValueError(f"{path}: no faces found (is it a Wavefront .obj file?)")
    positions = np.asarray(points, dtype=np.float32)[corners_p]
    normals = None
    if has_normals and file_normals and len(corners_n) == len(corners_p):
        normals = np.asarray(file_normals, dtype=np.float32)[corners_n]
    return Mesh(positions, normals, smooth=smooth)
