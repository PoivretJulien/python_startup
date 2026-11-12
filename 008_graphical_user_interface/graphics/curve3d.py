"""Curves: polylines and NURBS, as arrays of points.

Pure NumPy, no GPU. The renderers draw what these functions return.

A **polyline** is a list of points joined by straight segments. A **NURBS**
curve is a smooth curve pulled toward a few *control points*; it is drawn by
computing many points along it and joining them — so a NURBS ends up as a
polyline, and everything here is about producing that array fast::

    from curve3d import Nurbs

    curve = Nurbs([(0, 0, 0), (1, 2, 0), (3, 2, 0), (4, 0, 0)])   # control points
    points = curve.points(64)                                      # 64 points along it
    ui.draw_polyline(points, YELLOW)                               # or ui.draw_nurbs(curve, YELLOW)

Points may have two coordinates (screen) or three (world); the result has the
same number.

The words of a NURBS:

* ``degree`` — 1 gives straight segments between the control points, 2 arcs,
  3 (the default) the smooth curve of drawing programs;
* ``weights`` — one number per control point, 1 by default; a larger weight
  pulls the curve closer to that point. Weights are what let a NURBS be an
  exact circle;
* ``knots`` — where each control point starts and stops having influence along
  the curve. Leave it out and the curve starts on the first control point and
  ends on the last (a "clamped" curve);
* ``closed`` — the curve loops back smoothly onto itself.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence

import numpy as np


# ------------------------------------------------------------------ polyline
def segments(points, closed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The start and the end of every segment of a polyline: two ``(N, dim)`` views."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 2:
        empty = np.zeros((0, points.shape[1] if points.ndim == 2 else 3), dtype=np.float32)
        return empty, empty
    if closed:
        return points, np.roll(points, -1, axis=0)
    return points[:-1], points[1:]


def length(points, closed: bool = False) -> float:
    """The total length of a polyline."""
    starts, ends = segments(points, closed)
    return float(np.linalg.norm(ends - starts, axis=1).sum())


# --------------------------------------------------------------------- NURBS
def clamped_knots(count: int, degree: int) -> tuple[float, ...]:
    """The usual knots: the curve touches its first and last control points."""
    inner = count - degree - 1
    middle = [i / (inner + 1) for i in range(1, inner + 1)]
    return (0.0,) * (degree + 1) + tuple(middle) + (1.0,) * (degree + 1)


def basis_at(count: int, degree: int, knots: Sequence[float], u: np.ndarray) -> np.ndarray:
    """How much each control point counts at each parameter of ``u``: ``(len(u), count)``."""
    k = np.asarray(knots, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    samples = len(u)

    # Degree 0: 1 in the knot span that contains u, 0 elsewhere.
    spans = len(k) - 1
    table = np.zeros((samples, spans))
    index = np.clip(np.searchsorted(k, u, side="right") - 1, degree, count - 1)
    table[np.arange(samples), index] = 1.0

    # Cox–de Boor: each degree blends two neighbours of the degree below.
    for p in range(1, degree + 1):
        width = spans - p
        left_span = k[p : p + width] - k[:width]
        right_span = k[p + 1 : p + 1 + width] - k[1 : 1 + width]
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(left_span > 0, (u[:, None] - k[:width]) / left_span, 0.0)
            right = np.where(right_span > 0, (k[p + 1 : p + 1 + width] - u[:, None]) / right_span, 0.0)
        table = left * table[:, :width] + right * table[:, 1 : width + 1]
    return table[:, :count]


@functools.lru_cache(maxsize=64)
def basis(count: int, degree: int, knots: tuple[float, ...], samples: int, closed: bool) -> np.ndarray:
    """``basis_at`` for ``samples`` evenly spaced parameters, remembered.

    This only depends on the *shape* of the curve's definition, not on where
    the control points are, so it is computed once: dragging a control point
    afterwards costs one small matrix product per frame.
    """
    # A closed curve's last point is its first: do not compute it twice.
    u = np.linspace(knots[degree], knots[count], samples, endpoint=not closed)
    return basis_at(count, degree, knots, u)


def _definition(control_points, degree, weights, knots, closed):
    """Check a curve's description and complete it: control, weights, degree, knots."""
    control = np.asarray(control_points, dtype=np.float64)
    if control.ndim != 2 or control.shape[0] < 2:
        raise ValueError("a NURBS needs at least two control points, as rows of (x, y) or (x, y, z)")
    count = control.shape[0]
    w = np.ones(count) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape[0] != count:
        raise ValueError(f"{w.shape[0]} weights for {count} control points")
    degree = max(1, min(int(degree), count - 1))
    if closed:
        # A loop: repeat the first `degree` control points at the end and use
        # evenly spaced knots, so the curve joins itself smoothly.
        control = np.concatenate((control, control[:degree]))
        w = np.concatenate((w, w[:degree]))
        count += degree
        knots = tuple(float(i) for i in range(count + degree + 1))
    else:
        knots = clamped_knots(count, degree) if knots is None else tuple(float(v) for v in knots)
        if len(knots) != count + degree + 1:
            raise ValueError(
                f"{len(knots)} knots: a curve of degree {degree} with {count} control points needs {count + degree + 1}"
            )
    return control, w, degree, knots


def nurbs_points(
    control_points,
    degree: int = 3,
    weights: Sequence[float] | None = None,
    knots: Sequence[float] | None = None,
    samples: int = 64,
    closed: bool = False,
) -> np.ndarray:
    """``samples`` points along a NURBS curve, as a ``(samples, dim)`` float32 array."""
    control, w, degree, knots = _definition(control_points, degree, weights, knots, closed)
    table = basis(len(control), degree, knots, max(int(samples), 2), bool(closed)) * w
    return ((table @ control) / table.sum(axis=1, keepdims=True)).astype(np.float32)


def nurbs_point(
    control_points,
    t: float,
    degree: int = 3,
    weights: Sequence[float] | None = None,
    knots: Sequence[float] | None = None,
    closed: bool = False,
) -> tuple[float, ...]:
    """The point of a NURBS curve at parameter ``t``: 0 is its start, 1 its end.

    On a closed curve ``t`` wraps around: 1 is the start again, 1.25 the same
    point as 0.25.
    """
    control, w, degree, knots = _definition(control_points, degree, weights, knots, closed)
    t = float(t) % 1.0 if closed else min(max(float(t), 0.0), 1.0)
    low, high = knots[degree], knots[len(control)]
    row = basis_at(len(control), degree, knots, np.array([low + (high - low) * t]))[0] * w
    return tuple(float(v) for v in (row @ control) / row.sum())


class Nurbs:
    """A NURBS curve you keep and edit.

    ``control_points`` is a list of ``(x, y)`` or ``(x, y, z)``; the other
    arguments are described at the top of this file. Everything can be changed
    afterwards — ``curve.control_points[2] = (3, 1, 0)``, ``curve.weights[2] =
    4`` — and the next ``points()`` follows.

    Drawn with ``ui.draw_nurbs(curve, color, pick=True)``, the control points
    can be picked and moved with the mouse; ``active``, ``busy`` and
    ``changed`` then tell what the user is doing (see ``curve_edit3d``).
    """

    def __init__(self, control_points, degree: int = 3, weights=None, knots=None, closed: bool = False) -> None:
        self.control_points = [tuple(float(v) for v in p) for p in control_points]
        self.degree = int(degree)
        self.weights = [1.0] * len(self.control_points) if weights is None else [float(v) for v in weights]
        self.knots = None if knots is None else [float(v) for v in knots]
        self.closed = bool(closed)
        # Mouse editing (``draw_nurbs(curve, ..., pick=True)``):
        self.active: int | None = None  # the control point that shows the gizmo
        self.busy = False  # the mouse is on, or dragging, one of the handles
        self.changed = False  # a control point moved during the last frame
        self._edit: dict = {}

    def points(self, samples: int = 64) -> np.ndarray:
        """``samples`` points along the curve: more samples, smoother picture."""
        return nurbs_points(self.control_points, self.degree, self.weights, self.knots, samples, self.closed)

    def point(self, t: float) -> tuple[float, ...]:
        """The point of the curve at parameter ``t``, as ``(x, y, z)`` (or ``(x, y)``).

        ``t`` goes from 0, the start of the curve, to 1, its end; values
        outside are brought back to the nearest end (on a closed curve they
        wrap around instead). To place something on the curve, or move it
        along::

            x, y, z = curve.point(0.5)          # the middle of the parameter range

        ``t`` is the curve's own parameter, not a distance: equal steps of
        ``t`` are not exactly equal steps along the curve, they are shorter
        where the control points are close together.
        """
        return nurbs_point(self.control_points, t, self.degree, self.weights, self.knots, self.closed)

    @classmethod
    def circle(cls, center, radius: float) -> Nurbs:
        """An exact circle in the xy plane — what plain polynomial curves cannot do.

        Nine control points on a square around the circle; the corner points
        weigh less (0.707), which is exactly what rounds the corners off.
        """
        cx, cy = float(center[0]), float(center[1])
        rest = tuple(float(v) for v in center[2:])
        ring = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
        control = [(cx + radius * x, cy + radius * y, *rest) for x, y in ring]
        corner = 2.0**-0.5
        weights = [1, corner, 1, corner, 1, corner, 1, corner, 1]
        knots = [0, 0, 0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1, 1, 1]
        return cls(control, degree=2, weights=weights, knots=knots)
