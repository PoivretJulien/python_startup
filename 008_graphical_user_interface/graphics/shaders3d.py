"""WGSL shaders: line-for-line ports of the former GLSL 330 programs.

Every program shares one uniform block (``Frame``, packed by
``gpu3d.pack_uniforms``) and the ``to_wgpu_depth`` helper. The camera still
builds OpenGL-style matrices (clip depth in ``[-1, 1]``); WebGPU clips depth to
``[0, 1]``, so each vertex shader remaps ``z`` right before writing the
position. Nothing on the Python side changes because of it.

What had no WebGPU equivalent and how it is replaced:

* ``GL_POINTS`` + ``gl_PointSize`` (WebGPU points are always 1 px): every
  particle is an **instance** of a 6-vertex quad generated from
  ``vertex_index``. The particle buffer keeps its one-row-per-particle layout
  and is still one upload and one draw call; ``gl_PointCoord`` becomes the
  ``corner`` varying.
* ``noperspective`` / ``flat`` qualifiers become ``@interpolate(linear)`` /
  ``@interpolate(flat)``.
* ``discard`` stays ``discard``.
"""

from __future__ import annotations

COMMON = """
struct Frame {
    mvp: mat4x4<f32>,
    eye: vec4<f32>,
    forward: vec4<f32>,
    tint: vec4<f32>,
    viewport: vec2<f32>,      // framebuffer size, pixels
    screen: vec2<f32>,        // window size, points (cursor coordinates)
    pixel_scale: f32,         // framebuffer height / (2 tan(fov / 2))
    min_radius_px: f32,
    max_radius_px: f32,
    feather: f32,             // extra margin (pixels) that holds the AA fade
    flat_depth: f32,          // 1 = pin to the middle depth layer (AA sprites)
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Frame;

// OpenGL clip depth [-w, w] -> WebGPU [0, w].
fn to_wgpu_depth(p: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(p.x, p.y, 0.5 * (p.z + p.w), p.w);
}

// A position outside every clip plane: the whole primitive is discarded.
const CULLED = vec4<f32>(2.0, 2.0, 2.0, 1.0);
"""

# ------------------------------------------------------------ plain renderer
LINES = (
    COMMON
    + """
struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.position = to_wgpu_depth(u.mvp * vec4<f32>(v.position, 1.0));
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.color, 1.0);
}
"""
)

# ------------------------------------------------------------- AA strokes
_STROKE_IO = """
struct VIn {
    @location(0) segment_a: vec3<f32>,   // fixed first endpoint of the segment
    @location(1) segment_b: vec3<f32>,   // fixed second endpoint
    @location(2) color: vec3<f32>,
    @location(3) end: f32,               // 0 = corner sits at A, 1 = at B
    @location(4) side: f32,              // +1 / -1: which side of the line
    @location(5) half_width: f32,        // stroke half width in pixels
};

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) @interpolate(linear) px: vec2<f32>,
    @location(2) @interpolate(flat) segment_a: vec2<f32>,
    @location(3) @interpolate(flat) segment_b: vec2<f32>,
    @location(4) @interpolate(flat) half_width: f32,
};

// Exact Euclidean distance from p to segment a-b, in pixels.
fn distance_to_segment(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let pa = p - a;
    let ab = b - a;
    let t = clamp(dot(pa, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    return length(pa - ab * t);
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    let d = distance_to_segment(i.px, i.segment_a, i.segment_b);
    // 50% coverage exactly on the geometric edge, fading out over one pixel.
    let coverage = clamp(i.half_width + 0.5 - d, 0.0, 1.0);
    if (coverage <= 0.001) {
        discard;
    }
    return vec4<f32>(i.color, coverage);
}

// The quad corner around the projected segment a_px -> b_px (pixels).
fn stroke_corner(v: VIn, a_px: vec2<f32>, b_px: vec2<f32>) -> vec2<f32> {
    let delta = b_px - a_px;
    let len = length(delta);
    var dir = vec2<f32>(1.0, 0.0);
    if (len > 1e-6) {
        dir = delta / len;
    }
    let normal = vec2<f32>(-dir.y, dir.x);
    let anchor_px = mix(a_px, b_px, v.end);
    let cap = v.end * 2.0 - 1.0;   // A extends backwards, B forwards
    return anchor_px + normal * v.side * (v.half_width + u.feather) + dir * cap * u.feather;
}
"""

STROKES = (
    COMMON
    + _STROKE_IO
    + """
const NEAR_W: f32 = 0.05;  // clip-space near plane used for segment clipping

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    var ca = u.mvp * vec4<f32>(v.segment_a, 1.0);
    var cb = u.mvp * vec4<f32>(v.segment_b, 1.0);

    // Clip the 3D segment against the near plane in homogeneous space, so
    // every corner of the quad agrees on the clipped segment.
    let da = ca.w - NEAR_W;
    let db = cb.w - NEAR_W;
    if (da < 0.0 && db < 0.0) {
        out.position = CULLED;
        out.color = v.color;
        out.px = vec2<f32>(0.0);
        out.segment_a = vec2<f32>(0.0);
        out.segment_b = vec2<f32>(0.0);
        out.half_width = 0.0;
        return out;
    }
    if (da < 0.0) {
        ca = mix(ca, cb, da / (da - db));
    } else if (db < 0.0) {
        cb = mix(cb, ca, db / (db - da));
    }

    let a_px = (ca.xy / ca.w * 0.5 + 0.5) * u.viewport;
    let b_px = (cb.xy / cb.w * 0.5 + 0.5) * u.viewport;
    let corner = stroke_corner(v, a_px, b_px);

    // Back to clip space: multiplying the NDC by the anchor's w makes the
    // perspective divide land exactly on the corner.
    let anchor_clip = mix(ca, cb, v.end);
    let ndc = corner / u.viewport * 2.0 - 1.0;
    out.position = to_wgpu_depth(vec4<f32>(ndc * anchor_clip.w, anchor_clip.z, anchor_clip.w));
    out.color = v.color;
    out.px = corner;
    out.segment_a = a_px;
    out.segment_b = b_px;
    out.half_width = v.half_width;
    return out;
}
"""
)

# Screen-space (HUD) strokes: pixel input, top-left origin, y down.
SCREEN_STROKES = (
    COMMON
    + _STROKE_IO
    + """
@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    let a_px = v.segment_a.xy;
    let b_px = v.segment_b.xy;
    let corner = stroke_corner(v, a_px, b_px);
    let ndc = vec2<f32>(
        corner.x / u.screen.x * 2.0 - 1.0,
        1.0 - corner.y / u.screen.y * 2.0
    );
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.color = v.color;
    out.px = corner;
    out.segment_a = a_px;
    out.segment_b = b_px;
    out.half_width = v.half_width;
    return out;
}
"""
)

# ---------------------------------------------------------------- particles
# One instance per particle, six generated corners per instance. ``COLOR`` is
# vec3 for the opaque layout and vec4 for the coverage-channel twin.
_POINT_VERTEX = """
struct PIn {
    @location(0) position: vec3<f32>,
    @location(1) color: COLOR,
    @location(2) radius_world: f32,      // particle radius, world units
};

struct POut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: COLOR,
    @location(1) @interpolate(linear) corner: vec2<f32>,   // -1..1 across the sprite
    @location(2) @interpolate(flat) radius_px: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, p: PIn) -> POut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    var out: POut;
    out.color = p.color;
    let clip = u.mvp * vec4<f32>(p.position, 1.0);
    if (clip.w <= 0.0) {
        // Behind the eye: GL clipped the point, here the quad is dropped.
        out.position = CULLED;
        out.corner = vec2<f32>(0.0);
        out.radius_px = 0.0;
        return out;
    }
    // The caller states a world-space radius; the perspective divide is here.
    let depth = max(dot(p.position - u.eye.xyz, u.forward.xyz), 1e-6);
    let radius_px = clamp(p.radius_world * u.pixel_scale / depth, u.min_radius_px, u.max_radius_px);
    let half_px = radius_px + u.feather;

    let corner = corners[vertex];
    let offset_ndc = corner * half_px * 2.0 / u.viewport;
    if (u.flat_depth > 0.5) {
        // AA: every sprite sits exactly on the middle depth layer (GL's NDC
        // z = 0). Emitting w = 1 keeps that depth bit-exact: 0.5 * w / w can
        // round differently per particle and z-fight with its neighbours.
        out.position = vec4<f32>(clip.xy / clip.w + offset_ndc, 0.5, 1.0);
    } else {
        out.position = to_wgpu_depth(vec4<f32>(clip.xy + offset_ndc * clip.w, clip.z, clip.w));
    }
    out.corner = corner;
    out.radius_px = radius_px;
    return out;
}
"""


def _points(color_type: str, fragment: str) -> str:
    return COMMON + _POINT_VERTEX.replace("COLOR", color_type) + fragment.replace(
        "COLOR", color_type
    )


# Plain renderer: hard squares (opaque) ...
POINTS_PLAIN = _points(
    "vec3<f32>",
    """
@fragment
fn fs_main(i: POut) -> @location(0) vec4<f32> {
    return vec4<f32>(i.color, 1.0);
}
""",
)
# ... and the same square with the particle's own alpha written through.
POINTS_PLAIN_RGBA = _points(
    "vec4<f32>",
    """
@fragment
fn fs_main(i: POut) -> @location(0) vec4<f32> {
    return i.color;
}
""",
)

# AA renderer: a feathered disc from an exact pixel distance field.
_DISC = """
@fragment
fn fs_main(i: POut) -> @location(0) vec4<f32> {
    let d = length(i.corner) * (i.radius_px + u.feather);
    let coverage = clamp(i.radius_px + 0.5 - d, 0.0, 1.0);
    if (coverage <= 0.001) {
        discard;
    }
    return OUTPUT;
}
"""
POINTS_AA = _points("vec3<f32>", _DISC.replace("OUTPUT", "vec4<f32>(i.color, coverage)"))
POINTS_AA_RGBA = _points(
    "vec4<f32>", _DISC.replace("OUTPUT", "vec4<f32>(i.color.rgb, i.color.a * coverage)")
)

# ------------------------------------------------------------------ pictures
IMAGE = (
    COMMON
    + """
@group(1) @binding(0) var image_texture: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;

struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) texcoord: vec2<f32>,
};

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    out.position = to_wgpu_depth(u.mvp * vec4<f32>(v.position, 1.0));
    out.uv = v.texcoord;
    return out;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    // u.tint multiplies the texel; fully transparent texels are killed so a
    // cut-out picture can still write depth.
    let texel = textureSample(image_texture, image_sampler, i.uv) * u.tint;
    if (texel.a <= 0.003) {
        discard;
    }
    return texel;
}
"""
)

# ------------------------------------------------- screen-space shapes (SDF)
# Rounded rectangles and circles in window pixels (top-left origin, y down).
# One instance per shape, six generated corners per instance — the same trick as
# the particles. Sizes arrive in window points and are converted to framebuffer
# pixels here, so the one-pixel fade stays one *physical* pixel on a HiDPI
# (Retina) surface.
_SHAPE_VERTEX = """
struct SIn {
    @location(0) center: vec2<f32>,      // window points, top-left origin
    @location(1) half_size: vec2<f32>,   // half width / half height, points
    @location(2) color: vec4<f32>,
    @location(3) radius: f32,            // corner radius, points
};

struct SOut {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) color: vec4<f32>,
    @location(1) @interpolate(linear) local: vec2<f32>,    // from the centre, pixels
    @location(2) @interpolate(flat) half_px: vec2<f32>,
    @location(3) @interpolate(flat) radius_px: f32,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, s: SIn) -> SOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0),
    );
    let scale = u.viewport / u.screen;          // points -> framebuffer pixels
    let half_px = s.half_size * scale;
    // u.feather is the margin that holds the fade: 1 px for the SDF program,
    // 0 for the sharp one (the quad then is exactly the rectangle).
    let local = corners[vertex] * (half_px + vec2<f32>(u.feather));
    let pixel = s.center * scale + local;
    var out: SOut;
    out.position = vec4<f32>(
        pixel.x / u.viewport.x * 2.0 - 1.0,
        1.0 - pixel.y / u.viewport.y * 2.0,
        0.0,
        1.0
    );
    out.color = s.color;
    out.local = local;
    out.half_px = half_px;
    out.radius_px = s.radius * min(scale.x, scale.y);
    return out;
}
"""

# Rounded corners / circles: one signed distance per fragment, faded over one
# pixel (50% coverage exactly on the geometric edge, like the strokes).
SCREEN_SHAPES = (
    COMMON
    + _SHAPE_VERTEX
    + """
@fragment
fn fs_main(i: SOut) -> @location(0) vec4<f32> {
    // Signed distance to a rounded box: negative inside, 0 on the edge.
    let q = abs(i.local) - i.half_px + vec2<f32>(i.radius_px);
    let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - i.radius_px;
    let coverage = clamp(0.5 - d, 0.0, 1.0);
    if (coverage <= 0.001) {
        discard;
    }
    return vec4<f32>(i.color.rgb, i.color.a * coverage);
}
"""
)

# radius == 0: no distance field, no discard, no fade — the rasteriser's own
# edges are the rectangle. This is the cheapest fragment shader there is.
SCREEN_RECT_SHARP = (
    COMMON
    + _SHAPE_VERTEX
    + """
@fragment
fn fs_main(i: SOut) -> @location(0) vec4<f32> {
    return i.color;
}
"""
)

# --------------------------------------------------------------- vertex layouts
# (location, components, float offset) — the same triples the GL VAOs used.
SCREEN_SHAPE_LAYOUT = (9, ((0, 2, 0), (1, 2, 2), (2, 4, 4), (3, 1, 8)))
LINE_LAYOUT = (6, ((0, 3, 0), (1, 3, 3)))
QUAD_LAYOUT = (5, ((0, 3, 0), (1, 2, 3)))
STROKE_LAYOUT = (12, ((0, 3, 0), (1, 3, 3), (2, 3, 6), (3, 1, 9), (4, 1, 10), (5, 1, 11)))
POINT_LAYOUT = (7, ((0, 3, 0), (1, 3, 3), (2, 1, 6)))
POINT_RGBA_LAYOUT = (8, ((0, 3, 0), (1, 4, 3), (2, 1, 7)))

ALL_SHADERS = {
    "lines": LINES,
    "strokes": STROKES,
    "screen_strokes": SCREEN_STROKES,
    "points_plain": POINTS_PLAIN,
    "points_plain_rgba": POINTS_PLAIN_RGBA,
    "points_aa": POINTS_AA,
    "points_aa_rgba": POINTS_AA_RGBA,
    "image": IMAGE,
    "screen_shapes": SCREEN_SHAPES,
    "screen_rect_sharp": SCREEN_RECT_SHARP,
}
