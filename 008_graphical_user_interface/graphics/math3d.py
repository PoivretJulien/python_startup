"""Tiny 4x4 matrix math on NumPy, trimmed to what the 3D experiment needs.

This is the same convention used by the tutorial's ``gmath.py``:

* vectors are *columns*: ``clip = projection @ view @ model @ position``;
* translation lives in the last column;
* NumPy matrices are row-major, OpenGL wants column-major, so exactly one
  transpose happens — in :func:`to_gl` — at the upload boundary.
"""

from __future__ import annotations

import functools
import math

import numpy as np


def identity() -> np.ndarray:
    return np.eye(4, dtype=np.float32)


def mul(*matrices: np.ndarray) -> np.ndarray:
    """``mul(P, V, M)`` == ``P @ V @ M`` (left to right, like GLSL)."""
    return functools.reduce(np.matmul, matrices).astype(np.float32)


def perspective(fovy_degrees: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed projection, depth mapped to [-1, 1] (OpenGL clip space)."""
    f = 1.0 / math.tan(math.radians(fovy_degrees) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """World -> view matrix: ``eye`` is where the camera is, ``target`` where it looks.

    The default up is +z: this experiment uses a z-up world (the floor is the xy
    plane), unlike the tutorial's y-up steps 06+.
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)

    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-9)
    side = np.cross(forward, up)
    side /= max(np.linalg.norm(side), 1e-9)
    true_up = np.cross(side, forward)

    m = identity()
    m[0, 0:3], m[1, 0:3], m[2, 0:3] = side, true_up, -forward
    m[0, 3], m[1, 3], m[2, 3] = -side @ eye, -true_up @ eye, forward @ eye
    return m


def to_gl(m: np.ndarray) -> np.ndarray:
    """Row-major (math) -> column-major, contiguous float32, as OpenGL wants it."""
    return np.ascontiguousarray(np.asarray(m, dtype=np.float32).T)
