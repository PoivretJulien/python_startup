"""Editing a 3D curve with the mouse: pickable control points and a move gizmo.

Used by ``draw_nurbs(curve, color, pick=True)``; nothing here needs to be
called directly. What the user gets:

* every control point of the curve is a dot; the one under the cursor lights up;
* a click on a dot makes it *active*: three arrows — red x, green y, blue z —
  pop up on it (the gizmo);
* dragging an arrow slides the point along that axis only; dragging one of
  the three small squares between the arrows moves it in the plane of those
  two axes; dragging the dot itself moves it freely, in the plane facing the
  camera;
* a click on the background (without dragging) puts the gizmo away.

The state lives on the curve: ``curve.active`` (the index of the active point,
or ``None``), ``curve.busy`` (the mouse is on, or dragging, one of its handles
— the moment not to move the camera) and ``curve.changed`` (a point moved
during this frame).

How a drag along an axis works, since the mouse only moves in two dimensions:
at the moment of the click, the axis is projected on the screen, which gives
"so many pixels on screen per unit along the axis, in this direction". Mouse
movement measured along that direction, divided by that scale, is the distance
to slide the point.
"""

from __future__ import annotations

import glfw
import gpu3d
import numpy as np
import screen2d

AXES = np.eye(3)
AXIS_COLORS = ((0.95, 0.30, 0.30, 1.0), (0.35, 0.85, 0.35, 1.0), (0.35, 0.55, 1.00, 1.0))
HOT = (1.00, 0.85, 0.10, 1.0)  # under the cursor, or being dragged
DOT = (0.85, 0.85, 0.92, 1.0)
POLYGON = (0.50, 0.50, 0.58, 0.7)
DOT_RADIUS = 5.0  # pixels
PICK_RADIUS = 10.0  # how close the cursor must be to a dot
GIZMO_LENGTH = 70.0  # length of the arrows on screen, pixels
AXIS_GRAB = 14.0  # thickness of the zone that grabs an arrow
CLICK_SLOP = 4.0  # a press that moves less than this is a click, not a drag
# The three small squares: each lies between two arrows and moves the point in
# the plane of those two axes. As fractions of the arrows' length.
SQUARE_NEAR, SQUARE_FAR = 0.30, 0.58
# Plane `n` is the one that faces axis n: 0 = yz, 1 = zx, 2 = xy. It takes the
# colour of the axis it does NOT move along, as in modelling programs.
PLANE_AXES = ((1, 2), (2, 0), (0, 1))


def to_screen(renderer, points) -> tuple[np.ndarray, np.ndarray]:
    """World points -> window pixels: ``(N, 2)`` positions and ``(N,)`` "in front"."""
    width, height = (max(v, 1) for v in renderer.gpu.window_size())
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))[:, :3]
    clip = np.concatenate((p, np.ones((len(p), 1))), axis=1) @ np.asarray(
        renderer.camera.mvp(width / height), dtype=np.float64
    ).T
    w = clip[:, 3]
    safe = np.where(np.abs(w) > 1e-9, w, 1e-9)
    x = (clip[:, 0] / safe * 0.5 + 0.5) * width
    y = (0.5 - clip[:, 1] / safe * 0.5) * height
    return np.stack((x, y), axis=1), w > 1e-6


def plane_squares(renderer, origin: np.ndarray, reach: float) -> list[np.ndarray | None]:
    """The three squares of the gizmo on screen: four corners each, or ``None``."""
    squares: list[np.ndarray | None] = []
    for a, b in PLANE_AXES:
        corners = origin + reach * np.array(
            [
                SQUARE_NEAR * AXES[a] + SQUARE_NEAR * AXES[b],
                SQUARE_FAR * AXES[a] + SQUARE_NEAR * AXES[b],
                SQUARE_FAR * AXES[a] + SQUARE_FAR * AXES[b],
                SQUARE_NEAR * AXES[a] + SQUARE_FAR * AXES[b],
            ]
        )
        spots, visible = to_screen(renderer, corners)
        squares.append(spots if visible.all() else None)
    return squares


def in_quad(cursor: np.ndarray, quad: np.ndarray) -> bool:
    """Is the cursor inside a four-sided shape (corners in order)?"""
    edges = np.roll(quad, -1, axis=0) - quad
    to_cursor = cursor - quad
    turn = edges[:, 0] * to_cursor[:, 1] - edges[:, 1] * to_cursor[:, 0]
    return bool((turn >= 0).all() or (turn <= 0).all())  # always on the same side


def on_plane(renderer, cursor: np.ndarray, point: np.ndarray, normal: np.ndarray) -> np.ndarray | None:
    """Where the line of sight through the cursor meets a plane, or ``None``."""
    width, height = renderer.gpu.window_size()
    near, direction = renderer.camera.screen_ray(cursor[0], cursor[1], width, height)
    facing = float(direction @ normal)
    if abs(facing) < 0.03:  # the plane is seen edge-on: no usable crossing
        return None
    distance = float((point - near) @ normal) / facing
    return near + direction * distance if distance > 0.0 else None


def edit(renderer, curve) -> None:
    """Read the mouse and move the curve's points (``draw_handles`` draws them)."""
    window = renderer.window
    cursor = np.asarray(glfw.get_cursor_pos(window), dtype=np.float64)
    down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    state = curve._edit
    pressed = down and not state.get("was_down", False)
    released = not down and state.get("was_down", False)
    state["was_down"] = down
    curve.changed = False

    control = np.asarray(curve.control_points, dtype=np.float64)
    if curve.active is not None and curve.active >= len(control):
        curve.active = None
    spots, visible = to_screen(renderer, control)
    height = max(renderer.gpu.window_size()[1], 1)

    # --- what is under the cursor: an arrow of the gizmo first, then a dot ----
    hot_axis = -1
    tips = None
    if curve.active is not None and visible[curve.active]:
        origin = control[curve.active]
        # The arrows keep the same length on screen whatever the distance.
        scale = renderer.camera.pixels_per_world_unit(origin, height)
        reach = GIZMO_LENGTH / scale if scale > 0.0 else 0.0
        tips, tips_visible = to_screen(renderer, origin + AXES * reach)
        if not tips_visible.all():
            tips = None
        elif state.get("drag") is None:
            start = spots[curve.active]
            for axis in range(3):
                # From a little off the dot, so the dot itself stays clickable.
                near = start + (tips[axis] - start) * 0.2
                if screen2d.in_line(cursor, near, tips[axis], AXIS_GRAB):
                    hot_axis = axis
    hot_plane = -1
    if tips is not None and hot_axis < 0 and state.get("drag") is None:
        for plane, quad in enumerate(plane_squares(renderer, control[curve.active], reach)):
            if quad is not None and in_quad(cursor, quad):
                hot_plane = plane
    hot_dot = -1
    if hot_axis < 0 and hot_plane < 0 and state.get("drag") is None:
        distance = np.where(visible, np.linalg.norm(spots - cursor, axis=1), np.inf)
        nearest = int(np.argmin(distance))
        if distance[nearest] <= PICK_RADIUS:
            hot_dot = nearest

    # --- the click: start a drag --------------------------------------------
    if pressed:
        state["press"] = cursor.copy()
        if hot_axis >= 0:
            origin = control[curve.active]
            ahead, _ok = to_screen(renderer, origin + AXES[hot_axis])
            state["drag"] = {
                "axis": hot_axis,
                "start": origin.copy(),
                "cursor": cursor.copy(),
                "per_unit": ahead[0] - spots[curve.active],  # pixels per unit along the axis
            }
        elif hot_plane >= 0:
            origin = control[curve.active]
            normal = AXES[hot_plane]
            state["drag"] = {
                "axis": -2,
                "plane": hot_plane,
                "start": origin.copy(),
                "cursor": cursor.copy(),
                "grab": on_plane(renderer, cursor, origin, normal),  # where the plane was caught
            }
        elif hot_dot >= 0:
            curve.active = hot_dot
            right, up, _forward = renderer.camera.basis()
            state["drag"] = {
                "axis": -1,
                "start": control[hot_dot].copy(),
                "cursor": cursor.copy(),
                "right": right,
                "up": up,
                "scale": renderer.camera.pixels_per_world_unit(control[hot_dot], height),
            }
        else:
            state["drag"] = None

    # --- the drag: move the active point --------------------------------------
    drag = state.get("drag")
    if drag is not None and down and curve.active is not None:
        moved = cursor - drag["cursor"]
        if drag["axis"] >= 0:
            per_unit = drag["per_unit"]
            size = float(per_unit @ per_unit)
            # An axis pointing straight at the camera has no length on screen:
            # nothing sensible to do, the point stays.
            amount = float(moved @ per_unit) / size if size > 1e-6 else 0.0
            target = drag["start"] + AXES[drag["axis"]] * amount
        elif drag["axis"] == -2:
            # The point follows the spot of the plane that is under the cursor.
            here = on_plane(renderer, cursor, drag["start"], AXES[drag["plane"]])
            target = drag["start"] if here is None or drag["grab"] is None else drag["start"] + here - drag["grab"]
            target[drag["plane"]] = drag["start"][drag["plane"]]  # exactly in the plane
        elif drag["scale"] > 0.0:
            target = drag["start"] + (drag["right"] * moved[0] - drag["up"] * moved[1]) / drag["scale"]
        else:
            target = drag["start"]
        before = curve.control_points[curve.active]
        after = tuple(float(v) for v in target)
        if after != tuple(before):
            curve.control_points[curve.active] = after
            curve.changed = True
            control[curve.active] = target
            spots, visible = to_screen(renderer, control)

    # --- the release: a click on the background puts the gizmo away -----------
    if released:
        press = state.get("press")
        was_click = press is not None and float(np.linalg.norm(cursor - press)) < CLICK_SLOP
        if drag is None and was_click:
            curve.active = None
        state["drag"] = None
        drag = None

    curve.busy = drag is not None or hot_axis >= 0 or hot_plane >= 0 or hot_dot >= 0

    state["hot_plane"] = hot_plane
    state["hot_axis"] = hot_axis
    state["hot_dot"] = hot_dot


def draw_handles(renderer, curve) -> None:
    """Draw the control polygon, the dots and the gizmo, on top of the scene."""
    state = curve._edit
    drag = state.get("drag")
    hot_axis = state.get("hot_axis", -1)
    hot_dot = state.get("hot_dot", -1)
    control = np.asarray(curve.control_points, dtype=np.float64)
    spots, visible = to_screen(renderer, control)
    height = max(renderer.gpu.window_size()[1], 1)

    if visible.all():
        renderer.draw_screen_polyline(spots, POLYGON, 1.0, closed=False)
    for index, spot in enumerate(spots):
        if not visible[index]:
            continue
        lit = index == hot_dot or (index == curve.active and drag is not None and drag["axis"] == -1)
        renderer.draw_screen_circle(spot, DOT_RADIUS + (1.5 if index == curve.active else 0.0), HOT if lit else DOT)

    if curve.active is None or not visible[curve.active]:
        return
    # The gizmo is recomputed here so it follows the point while it is dragged.
    origin = control[curve.active]
    scale = renderer.camera.pixels_per_world_unit(origin, height)
    if scale <= 0.0:
        return
    tips, tips_visible = to_screen(renderer, origin + AXES * (GIZMO_LENGTH / scale))
    start = spots[curve.active]
    dragged_axis = drag["axis"] if drag is not None else -1

    # The three squares first, so the arrows are drawn over them.
    hot_plane = drag["plane"] if dragged_axis == -2 else state.get("hot_plane", -1)
    for plane, quad in enumerate(plane_squares(renderer, origin, GIZMO_LENGTH / scale)):
        if quad is None:
            continue
        r, g, b, _a = HOT if plane == hot_plane else AXIS_COLORS[plane]
        rows = np.empty((2, screen2d.TRIANGLE_FLOATS), dtype=np.float32)  # a square = two triangles
        rows[0, 0:6] = quad[[0, 1, 2]].ravel()
        rows[1, 0:6] = quad[[0, 2, 3]].ravel()
        rows[:, 6:10] = (r, g, b, 0.75 if plane == hot_plane else 0.35)
        renderer.gpu.submit(
            gpu3d.DrawCall(
                pipeline=renderer.sharp_triangle_pipelines[True],
                uniforms=renderer._screen_uniforms(0.0),
                vertex_count=3,
                instance_count=2,
                vertices=rows,
            )
        )
        renderer.draw_screen_polyline(quad, (r, g, b, 1.0), 1.5, closed=True)

    for axis in range(3):
        if not tips_visible[axis]:
            continue
        color = HOT if axis in (hot_axis, dragged_axis) else AXIS_COLORS[axis]
        along = tips[axis] - start
        length = float(np.linalg.norm(along))
        if length < 6.0:  # the axis points at the camera: a dot is all there is to see
            renderer.draw_screen_circle(start, 3.0, color)
            continue
        direction = along / length
        renderer.draw_screen_line(start, tips[axis] - direction * 9.0, color, 3.0)
        renderer.draw_screen_triangle(tips[axis] - direction * 11.0, 11.0, 12.0, direction, color)
