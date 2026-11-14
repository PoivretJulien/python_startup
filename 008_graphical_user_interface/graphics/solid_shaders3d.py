"""The shaders of the solid shapes, and the small shader system on top of them.

A *shader* here is one WGSL function that decides the colour of one point of a
surface. Everything around it — placing the shape, finding the point, the
normal, the shadow — is done for you and handed over in a ``Surface``::

    fn shade(s: Surface) -> vec3<f32> {
        return s.color * max(dot(s.normal, s.light), 0.0);
    }

Pick a ready-made one by name (``shader="toon"``) or write your own::

    from solid_shaders3d import Shader

    stripes = Shader('''
    fn shade(s: Surface) -> vec3<f32> {
        let band = step(0.5, fract(s.position.z * 4.0 + s.time));
        return lit(s) * mix(0.4, 1.0, band);
    }
    ''')
    ui.draw_sphere((0, 0, 1), 1.0, ORANGE, shader=stripes)

What a ``Surface`` holds:

==================  =========================================================
``position``        the point, in world coordinates
``normal``          the direction the surface faces there (length 1)
``view``            the direction from the point to the camera (length 1)
``color``           the colour given to the draw call
``light``           the direction from the point to the light (length 1)
``light_color``     the colour of the light
``ambient``         how bright the unlit side stays, 0..1
``shadow``          1 = lit, 0 = in shadow
``occlusion``       1 = open, lower in creases (always 1 without ``--raymarch``)
``specular``        strength of the shiny highlight, 0..1
``shininess``       how tight the highlight is (higher = smaller)
``time``            seconds since the program started, for animation
``edge``            pixels to the nearest edge of the triangle being drawn
==================  =========================================================

Helpers a shader may call: ``lit(s)`` (the default look, whole), ``diffuse(s)``
(how much light falls on the point, 0..1, shadow included),
``highlight(s)`` (the shiny spot, 0..1) and ``wire(s, width)`` (1 on the edges
of the triangles, 0 inside, for lines ``width`` pixels wide). A shader may
``discard;`` a point to leave a hole there. A solid whose shader uses the
triangles' edges is always drawn as triangles, even with ``--raymarch``.

The same shader source is compiled for triangles and for the ray marcher, so
a shader works in both modes without change. On a see-through solid (a colour
with an alpha below 1) the colour your shader returns is then mixed with what
is behind the solid.
"""

from __future__ import annotations

MAX_SOLIDS = 64  # solids the ray marcher can hold in one frame
SCENE_HEADER = 32  # info, shadow, light matrix, lamp, extra: floats before the first solid
SCENE_TINTS = SCENE_HEADER + 16 * MAX_SOLIDS  # then one (r, g, b, opacity) per solid
SCENE_FLOATS = SCENE_TINTS + 4 * MAX_SOLIDS
SHADOW_MAP_SIZE = 2048  # pixels on each side of the shadow map

# Position + normal, six floats per vertex.
SOLID_LAYOUT = (6, ((0, 3, 0), (1, 3, 3)))

_COMMON = """
struct Frame {
    mvp: mat4x4<f32>,
    eye: vec4<f32>,
    light: vec4<f32>,         // xyz: toward the light, w: ambient
    tint: vec4<f32>,          // the colour of the solid; a = opacity
    viewport: vec2<f32>,
    screen: vec2<f32>,
    material: vec4<f32>,      // specular, shininess, time, 1 = ray-marched extras
    solid: vec4<f32>,         // index in the scene (ray marcher), refraction index, thickness
    model: mat4x4<f32>,       // unit shape -> world
    light_color: vec4<f32>,
    bounds: vec4<f32>,        // half size of the box around the solid, its own axes
};

@group(0) @binding(0) var<uniform> u: Frame;

// Every solid of the frame: [kind, p0, p1, p2] then three axes, each with one
// coordinate of the centre in w. info = (count of the solids that cast full
// shadows, detail size, reach, count of the see-through ones). The first are
// stored from the start of `rows`, the second from its end.
struct Scene {
    info: vec4<f32>,
    shadow: vec4<f32>,        // 1 = shadow map in use, normal offset, depth bias, texel
    light_matrix: mat4x4<f32>,  // world -> the shadow map
    lamp: vec4<f32>,          // an area light: xyz its position, w its radius (0 = the sun)
    extra: vec4<f32>,         // spread of the sun's shadows, contact strength, 1 = glass map in use
    rows: array<vec4<f32>, ROWS>,
    tints: array<vec4<f32>, SOLIDS>,   // see-through solids: their colour and opacity
};

const MAX_SOLIDS: i32 = SOLIDS;

@group(1) @binding(0) var<uniform> scene: Scene;
@group(1) @binding(1) var shadow_map: texture_depth_2d;
@group(1) @binding(2) var shadow_sampler: sampler_comparison;
@group(1) @binding(3) var glass_depth: texture_depth_2d;
@group(1) @binding(4) var glass_tint: texture_2d<f32>;

// The light as one point of the scene sees it. `spread` is how fast the edge
// of a shadow widens: its width is spread x the distance to what casts it.
struct Light {
    toward: vec3<f32>,        // unit vector from the point to the light
    spread: f32,
    reach: f32,               // how far a shadow ray has to travel
};

fn light_at(position: vec3<f32>) -> Light {
    var light: Light;
    if (scene.lamp.w > 0.0) {
        // An area light is somewhere: its direction and its apparent size
        // change from point to point.
        let to_lamp = scene.lamp.xyz - position;
        let distance = max(length(to_lamp), 1e-4);
        light.toward = to_lamp / distance;
        light.spread = scene.lamp.w / distance;
        light.reach = max(distance - scene.lamp.w, 0.0);
    } else {
        light.toward = u.light.xyz;
        light.spread = scene.extra.x;
        light.reach = scene.info.z;
    }
    return light;
}

fn random(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Point `n` of `count` on a spiral that fills a disc of radius 1 evenly.
fn spiral(n: i32, count: i32, turn: f32) -> vec2<f32> {
    let r = sqrt((f32(n) + 0.5) / f32(count));
    let a = f32(n) * 2.3999632 + turn;
    return r * vec2<f32>(cos(a), sin(a));
}

// The shadow map holds, for each direction from the light, the distance to the
// first surface. A point farther than that is in shadow.
//
// To soften the edge like the ray-marched shadows — sharp where the shadow
// touches its object, wider farther away — the map is read in two rounds:
// first "how far in front of me is what blocks the light?", then an average
// over a disc whose size grows with that distance.
fn map_shadow(position: vec3<f32>, normal: vec3<f32>, spread: f32) -> f32 {
    let p = scene.light_matrix * vec4<f32>(position + normal * scene.shadow.y, 1.0);
    let uv = p.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 1.0; }
    // A point beyond the far end of the map is still behind whatever the map
    // holds in its direction: a low light throws shadows far past the casters.
    let depth = min(p.z, 1.0) - scene.shadow.z;
    let texel = scene.shadow.w;
    let size = 1.0 / texel;
    let turn = random(position.xy * 37.0 + position.z * 11.0) * 6.2831853;

    // Round 1: the average depth of what is nearer to the light around here.
    let search = clamp(spread * 0.35 * size, 2.0, 24.0);
    var blocker = 0.0;
    var blockers = 0.0;
    for (var n = 0; n < 10; n++) {
        let at = clamp((uv + spiral(n, 10, turn) * search * texel) * size, vec2<f32>(0.0), vec2<f32>(size - 1.0));
        let z = textureLoad(shadow_map, vec2<i32>(at), 0);
        if (z < depth) {
            blocker += z;
            blockers += 1.0;
        }
    }
    if (blockers < 0.5) { return 1.0; }

    // Round 2: average over a disc as wide as the edge should be there.
    let gap = max(depth - blocker / blockers, 0.0);
    let radius = clamp(spread * gap * size, 1.0, 40.0);
    var lit = 0.0;
    for (var n = 0; n < 16; n++) {
        lit += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + spiral(n, 16, turn) * radius * texel, depth);
    }
    return lit / 16.0;
}

// The shadow of see-through solids drawn as triangles: a second map, seen
// from the light like the first, that holds for each direction the nearest
// see-through solid and the colour of the light it lets pass. A point behind
// it receives that colour; averaged over a few places for a soft edge.
fn map_glass(position: vec3<f32>, normal: vec3<f32>, spread: f32) -> vec3<f32> {
    let p = scene.light_matrix * vec4<f32>(position + normal * scene.shadow.y, 1.0);
    let uv = p.xy * vec2<f32>(0.5, -0.5) + 0.5;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return vec3<f32>(1.0); }
    let depth = min(p.z, 1.0) - 2.0 * scene.shadow.z;
    let size = 1.0 / scene.shadow.w;
    let turn = random(position.xy * 41.0 + position.z * 7.0) * 6.2831853;
    let radius = clamp(spread * 0.06 * size, 1.5, 14.0);
    var through = vec3<f32>(0.0);
    for (var n = 0; n < 8; n++) {
        let at = vec2<i32>(clamp(uv * size + spiral(n, 8, turn) * radius, vec2<f32>(0.0), vec2<f32>(size - 1.0)));
        if (textureLoad(glass_depth, at, 0) < depth) {
            through += textureLoad(glass_tint, at, 0).rgb;
        } else {
            through += vec3<f32>(1.0);
        }
    }
    return through / 8.0;
}

fn to_wgpu_depth(p: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(p.x, p.y, 0.5 * (p.z + p.w), p.w);
}

// ---- distance functions: how far is point q from the surface of a shape
// centred on the origin, with its axis along z. Negative = inside.
fn sd_shape(kind: i32, q: vec3<f32>, s: vec3<f32>) -> f32 {
    switch kind {
        case 0 {                                    // sphere: s.x = radius
            return length(q) - s.x;
        }
        case 1 {                                    // box: s = half size
            let d = abs(q) - s;
            return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
        }
        case 2 {                                    // cylinder: radius, half height
            let d = vec2<f32>(length(q.xy) - s.x, abs(q.z) - s.y);
            return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
        }
        case 3 {                                    // cone: base radius, half height
            let p = vec2<f32>(length(q.xy), q.z);
            let k1 = vec2<f32>(0.0, s.y);
            let k2 = vec2<f32>(-s.x, 2.0 * s.y);
            let ca = vec2<f32>(p.x - min(p.x, select(0.0, s.x, p.y < 0.0)), abs(p.y) - s.y);
            let cb = p - k1 + k2 * clamp(dot(k1 - p, k2) / dot(k2, k2), 0.0, 1.0);
            let inside = select(1.0, -1.0, cb.x < 0.0 && ca.y < 0.0);
            return inside * sqrt(min(dot(ca, ca), dot(cb, cb)));
        }
        case 4 {                                    // capsule: radius, half length
            let p = vec3<f32>(q.xy, q.z - clamp(q.z, -s.y, s.y));
            return length(p) - s.x;
        }
        default {                                   // torus: ring radius, tube radius
            return length(vec2<f32>(length(q.xy) - s.x, q.z)) - s.y;
        }
    }
}

// World point -> the solid's own axes.
fn to_local(index: i32, p: vec3<f32>) -> vec3<f32> {
    let ax = scene.rows[index * 4 + 1];
    let ay = scene.rows[index * 4 + 2];
    let az = scene.rows[index * 4 + 3];
    let d = p - vec3<f32>(ax.w, ay.w, az.w);
    return vec3<f32>(dot(ax.xyz, d), dot(ay.xyz, d), dot(az.xyz, d));
}

fn solid_distance(index: i32, p: vec3<f32>) -> f32 {
    let head = scene.rows[index * 4];
    return sd_shape(i32(head.x), to_local(index, p), head.yzw);
}

// ---- which solids matter here
// A shadow ray, or an occlusion probe, only needs the few solids that are near
// it. They are listed once, with a cheap test on a ball around each solid, and
// the expensive exact distances are then computed for those alone. Most pixels
// of a floor list nothing at all and are finished at once.
const NEAR_MAX: i32 = 16;
var<private> near_count: i32;
var<private> near_ids: array<i32, 16>;

fn solid_center(index: i32) -> vec3<f32> {
    return vec3<f32>(scene.rows[index * 4 + 1].w, scene.rows[index * 4 + 2].w, scene.rows[index * 4 + 3].w);
}

// The radius of a ball that contains the solid (the sum of its sizes is
// always enough, and costs nothing to compute).
fn solid_bound(index: i32) -> f32 {
    let head = scene.rows[index * 4];
    return head.y + head.z + head.w;
}

// The solids a ray from `origin` could touch, or pass close enough to for
// the soft edge of their shadow. `skip` is left out.
fn solids_along(origin: vec3<f32>, toward: vec3<f32>, spread: f32, reach: f32, skip: i32) {
    near_count = 0;
    let count = i32(scene.info.x);
    for (var i = 0; i < count; i++) {
        if (i == skip) { continue; }
        let bound = solid_bound(i);
        let to_center = solid_center(i) - origin;
        let along = dot(to_center, toward);
        if (along < -bound || along > reach + bound) { continue; }      // behind, or beyond the light
        let aside = length(to_center - toward * along);
        if (aside - bound > spread * (max(along, 0.0) + bound)) { continue; }  // outside the cone of the soft edge
        if (near_count < NEAR_MAX) {
            near_ids[near_count] = i;
            near_count++;
        }
    }
}

// The solids within `radius` of a point.
fn solids_around(p: vec3<f32>, radius: f32, skip: i32) {
    near_count = 0;
    let count = i32(scene.info.x);
    for (var i = 0; i < count; i++) {
        if (i == skip) { continue; }
        if (length(solid_center(i) - p) - solid_bound(i) > radius) { continue; }
        if (near_count < NEAR_MAX) {
            near_ids[near_count] = i;
            near_count++;
        }
    }
}

// Distance from p to the nearest of the listed solids.
fn near_distance(p: vec3<f32>) -> f32 {
    var nearest = 1e9;
    for (var k = 0; k < near_count; k++) {
        nearest = min(nearest, solid_distance(near_ids[k], p));
    }
    return nearest;
}

// Distance from p to the nearest solid of the frame (all of them).
fn scene_distance(p: vec3<f32>) -> f32 {
    var nearest = 1e9;
    let count = i32(scene.info.x);
    for (var i = 0; i < count; i++) {
        nearest = min(nearest, solid_distance(i, p));
    }
    return nearest;
}

// Walk from the point toward the light. At each step, compare the distance to
// the nearest solid with the width the light's cone has there: far outside
// the cone = fully lit, deep inside a solid = fully dark, and everything in
// between is the soft edge. Walking on *through* the solids (the distance
// goes negative) is what gives the edge its inner half, so a shadow fades in
// on both sides of its outline instead of starting abruptly.
fn soft_shadow(origin: vec3<f32>, light: Light, skip: i32) -> f32 {
    solids_along(origin, light.toward, light.spread, light.reach, skip);
    if (near_count == 0) { return 1.0; }
    let detail = scene.info.y;
    let spread = max(light.spread, 0.004);
    // Even right against its object a shadow keeps an edge this wide, so the
    // place where a shape meets the floor is a soft fade and not a hard line.
    let least = 0.05 * detail;
    var open = 1.0;
    // Each pixel starts its walk at a slightly different place, so the steps
    // of neighbouring pixels do not line up into visible bands.
    var t = (0.02 + 0.03 * random(origin.xy * 53.0 + origin.z * 17.0)) * detail;
    var before = 1e9;         // the distance at the previous step
    for (var i = 0; i < 48; i++) {
        let h = near_distance(origin + light.toward * t);
        if (h > 0.0 && before < 1e8) {
            // Between two steps the ray may have passed closer than either
            // of them saw: estimate that closest approach from the two.
            let back = h * h / (2.0 * before);
            let closest = sqrt(max(h * h - back * back, 0.0));
            open = min(open, closest / (spread * max(t - back, 1e-4) + least));
        } else {
            open = min(open, h / (spread * t + least));
        }
        if (open < -1.0) { break; }
        before = max(h, 1e-4);
        t += clamp(0.8 * h, 0.012 * detail, 0.4 * detail);
        if (t > light.reach) { break; }
    }
    open = clamp(open, -1.0, 1.0);
    return 0.25 * (1.0 + open) * (1.0 + open) * (2.0 - open);          // -1..1 -> a smooth 0..1
}

// ---- shadows of see-through solids
// How much of the light's cone one solid covers, seen from `origin`: 0 = none,
// 1 = all of it. The same walk as soft_shadow, along one solid only.
fn glass_cover(index: i32, origin: vec3<f32>, light: Light) -> f32 {
    let detail = scene.info.y;
    let spread = max(light.spread, 0.004);
    let least = 0.05 * detail;
    var open = 1.0;
    var t = (0.02 + 0.03 * random(origin.xy * 29.0 + origin.z * 13.0)) * detail;
    for (var i = 0; i < 28; i++) {
        let h = solid_distance(index, origin + light.toward * t);
        open = min(open, h / (spread * t + least));
        if (open < -1.0) { break; }
        t += clamp(0.8 * h, 0.012 * detail, 1e9);
        if (t > light.reach) { break; }
    }
    open = clamp(open, -1.0, 1.0);
    return 1.0 - 0.25 * (1.0 + open) * (1.0 + open) * (2.0 - open);
}

// The light that reaches a point after passing through the see-through solids
// in its way: white = nothing in the way. Glass does not block light, it
// filters it — the shadow is lighter than a solid's and takes the colour of
// the glass. Each solid in the way filters what the previous ones let pass.
fn glass_shadow(origin: vec3<f32>, light: Light, own: i32) -> vec3<f32> {
    var through = vec3<f32>(1.0);
    let first = MAX_SOLIDS - i32(scene.info.w);
    for (var g = first; g < MAX_SOLIDS; g++) {
        let tint = scene.tints[g];
        if (g == own || tint.a <= 0.0) { continue; }
        // The cheap test first: is this solid anywhere near the ray?
        let bound = solid_bound(g);
        let to_center = solid_center(g) - origin;
        let along = dot(to_center, light.toward);
        if (along < -bound || along > light.reach + bound) { continue; }
        let aside = length(to_center - light.toward * along);
        if (aside - bound > light.spread * (max(along, 0.0) + bound) + 0.05 * scene.info.y) { continue; }
        let covered = glass_cover(g, origin, light);
        if (covered <= 0.0) { continue; }
        // Even clear glass stops a little light; coloured glass keeps its colour.
        let stopped = 0.15 + 0.85 * tint.a;
        let passes = (1.0 - stopped) * mix(vec3<f32>(1.0), tint.rgb, 0.7);
        through *= mix(vec3<f32>(1.0), passes, covered);
    }
    return through;
}

// Creases and contact areas receive less ambient light. Five probes along the
// normal ask "is something closer than it would be in the open?"; the nearer
// probes count more. scene.extra.y is the strength (0 = off).
fn ambient_occlusion(p: vec3<f32>, n: vec3<f32>, skip: i32) -> f32 {
    let strength = scene.extra.y;
    if (strength <= 0.0) { return 1.0; }
    let detail = scene.info.y;
    solids_around(p, 0.6 * detail, skip);
    if (near_count == 0) { return 1.0; }
    var blocked = 0.0;
    var weight = 1.0;
    var total = 0.0;
    for (var i = 0; i < 5; i++) {
        let reach = detail * (0.04 + 0.13 * f32(i));
        blocked += clamp((reach - near_distance(p + n * reach)) / reach, 0.0, 1.0) * weight;
        total += weight;
        weight *= 0.75;
    }
    // 0 = open, 1 = fully enclosed; eased so the darkening starts gently, and
    // never darker than a floor value: a contact is a shade, not a black line.
    let enclosed = smoothstep(0.0, 1.0, blocked / total);
    return max(1.0 - strength * enclosed, 0.25);
}

// ---- what a shader receives
struct Surface {
    position: vec3<f32>,
    normal: vec3<f32>,
    view: vec3<f32>,
    color: vec3<f32>,
    light: vec3<f32>,
    light_color: vec3<f32>,
    ambient: f32,
    shadow: f32,
    occlusion: f32,
    specular: f32,
    shininess: f32,
    time: f32,
    edge: f32,                // pixels to the nearest edge of the triangle
};

fn make_surface(position: vec3<f32>, normal: vec3<f32>) -> Surface {
    var s: Surface;
    s.position = position;
    s.normal = normal;
    s.view = normalize(u.eye.xyz - position);
    s.color = u.tint.rgb;
    let light = light_at(position);
    s.light = light.toward;
    s.light_color = u.light_color.rgb;
    s.ambient = u.light.w;
    s.shadow = 1.0;
    s.occlusion = 1.0;
    if (u.material.w > 0.5 && scene.info.x + scene.info.w > 0.5) {
        // A shape without hollows cannot shadow itself on its lit side, so it
        // is left out of its own shadow ray (all shapes but the torus).
        var skip = i32(u.solid.x);
        if (skip >= 0 && i32(scene.rows[skip * 4].x) == 5) { skip = -1; }
        // The shadow ray starts a little off the surface. A torus stays in its
        // own shadow test (it can shade itself through its hole), so it starts
        // farther out: beyond the soft band every contact has.
        let itself = i32(u.solid.x) >= 0 && skip < 0;
        let lifted = position + normal * (select(0.01, 0.07, itself) * scene.info.y);
        if (dot(normal, s.light) > 0.0) {
            s.shadow = soft_shadow(lifted, light, skip);
        }
        s.occlusion = ambient_occlusion(position, normal, skip);
        if (scene.info.w > 0.5 && s.shadow > 0.0 && dot(normal, s.light) > 0.0) {
            // What the see-through solids let pass: its brightness dims the
            // light, its colour tints it.
            let through = glass_shadow(lifted, light, i32(u.solid.x));
            let brightness = max(through.r, max(through.g, through.b));
            s.shadow *= brightness;
            s.light_color *= through / max(brightness, 1e-3);
        }
    }
    if (scene.shadow.x > 0.5 && dot(normal, s.light) > 0.0) {
        s.shadow = min(s.shadow, map_shadow(position, normal, light.spread));
    }
    if (scene.extra.z > 0.5 && s.shadow > 0.0 && dot(normal, s.light) > 0.0) {
        let through = map_glass(position, normal, light.spread);
        let brightness = max(through.r, max(through.g, through.b));
        s.shadow *= brightness;
        s.light_color *= through / max(brightness, 1e-3);
    }
    s.specular = u.material.x;
    s.shininess = u.material.y;
    s.time = u.material.z;
    s.edge = 1e6;             // far from any edge, unless the caller knows better
    return s;
}

// ---- helpers for shaders
fn diffuse(s: Surface) -> f32 {
    return max(dot(s.normal, s.light), 0.0) * s.shadow;
}

fn highlight(s: Surface) -> f32 {
    let facing = step(0.0, dot(s.normal, s.light));
    let halfway = normalize(s.light + s.view);
    return pow(max(dot(s.normal, halfway), 0.0), s.shininess) * s.specular * s.shadow * facing;
}

// 1 on the edges of the triangles, 0 inside: lines `width` pixels wide.
fn wire(s: Surface, width: f32) -> f32 {
    return 1.0 - smoothstep(width - 0.75, width + 0.75, s.edge);
}

fn lit(s: Surface) -> vec3<f32> {
    // The sky side of a shape gets a little more ambient light than its underside.
    let sky = mix(0.65, 1.0, 0.5 + 0.5 * s.normal.z);
    let ambient = s.ambient * sky * s.occlusion;
    return s.color * (ambient + (1.0 - s.ambient) * diffuse(s) * s.light_color)
        + highlight(s) * s.light_color;
}
""".replace("ROWS", str(4 * MAX_SOLIDS)).replace("SOLIDS", str(MAX_SOLIDS))

# ---------------------------------------------------------- the last step
# Opaque solids: the shader's colour is the pixel.
_OPAQUE = """
fn finish(s: Surface, shaded: vec3<f32>) -> vec3<f32> {
    return shaded;
}
"""

# See-through solids: mix the shader's colour with the picture behind, read
# where the line of sight lands after being bent by the surface.
_GLASS = """
@group(2) @binding(0) var backdrop: texture_2d<f32>;
@group(2) @binding(1) var backdrop_sampler: sampler;

fn finish(s: Surface, shaded: vec3<f32>) -> vec3<f32> {
    let opacity = u.tint.a;
    // Snell's law: entering a denser material bends the ray toward the normal.
    var bent = refract(-s.view, s.normal, 1.0 / max(u.solid.y, 1.0));
    if (dot(bent, bent) < 1e-6) { bent = -s.view; }
    // Follow the bent ray through the solid, and look up that place on screen.
    var clip = u.mvp * vec4<f32>(s.position + bent * u.solid.z, 1.0);
    if (clip.w < 1e-4) { clip = u.mvp * vec4<f32>(s.position, 1.0); }
    let uv = clamp(clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let behind = textureSampleLevel(backdrop, backdrop_sampler, uv, 0.0).rgb;
    let tinted = behind * mix(vec3<f32>(1.0), s.color, 0.5);       // coloured glass
    // Glass reflects more at a grazing angle: a faint bright edge.
    let edge = pow(1.0 - max(dot(s.normal, s.view), 0.0), 4.0);
    let shine = (highlight(s) + 0.25 * edge) * (1.0 - opacity);
    return mix(tinted, shaded, opacity) + shine * s.light_color;
}
"""

# ------------------------------------------------------------------ triangles
_MESH = """
struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) corner: vec3<f32>,
};

@vertex
fn vs_main(v: VIn, @builtin(vertex_index) number: u32) -> VOut {
    var out: VOut;
    // Each corner of a triangle gets its own axis: (1,0,0), (0,1,0), (0,0,1).
    // Inside the triangle the three numbers say how far each side is.
    let which = number % 3u;
    out.corner = vec3<f32>(f32(which == 0u), f32(which == 1u), f32(which == 2u));
    let world = u.model * vec4<f32>(v.position, 1.0);
    // A stretched shape needs its normals un-stretched: divide by the squared
    // scale of each axis, then rotate like the positions.
    let m = mat3x3<f32>(u.model[0].xyz, u.model[1].xyz, u.model[2].xyz);
    let squared = vec3<f32>(dot(m[0], m[0]), dot(m[1], m[1]), dot(m[2], m[2]));
    out.normal = m * (v.normal / max(squared, vec3<f32>(1e-20)));
    out.world = world.xyz;
    out.clip = to_wgpu_depth(u.mvp * world);
    return out;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    // Distance to the nearest side of the triangle, in pixels.
    let sides = i.corner / max(fwidth(i.corner), vec3<f32>(1e-6));
    let edge = min(sides.x, min(sides.y, sides.z));
    var normal = normalize(i.normal);
    // Files do not always agree on which side is "out": light the side we see.
    if (dot(normal, u.eye.xyz - i.world) < 0.0) { normal = -normal; }
    var surface = make_surface(i.world, normal);
    surface.edge = edge;
    return vec4<f32>(finish(surface, shade(surface)), 1.0);
}
"""

# ---------------------------------------------------------------- ray marcher
# The box around the solid is drawn (its far faces, so it also works with the
# camera inside); each of its pixels walks a ray through the box until it
# touches the exact mathematical surface, then writes that point's real depth
# so solids, meshes, lines and particles still hide each other correctly.
_MARCH = """
struct VIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
};

struct FOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_main(v: VIn) -> VOut {
    var out: VOut;
    let world = u.model * vec4<f32>(v.position, 1.0);
    out.world = world.xyz;
    out.clip = to_wgpu_depth(u.mvp * world);
    return out;
}

@fragment
fn fs_main(i: VOut) -> FOut {
    let index = i32(u.solid.x);
    let head = scene.rows[index * 4];
    let kind = i32(head.x);
    let ax = scene.rows[index * 4 + 1].xyz;
    let ay = scene.rows[index * 4 + 2].xyz;
    let az = scene.rows[index * 4 + 3].xyz;

    // The ray of this pixel, in the solid's own axes.
    let ray = normalize(i.world - u.eye.xyz);
    let origin = to_local(index, u.eye.xyz);
    let direction = vec3<f32>(dot(ax, ray), dot(ay, ray), dot(az, ray));

    // Where the ray enters and leaves the box around the solid.
    let inv = 1.0 / direction;
    let t0 = (-u.bounds.xyz - origin) * inv;
    let t1 = (u.bounds.xyz - origin) * inv;
    let lo = min(t0, t1);
    let hi = max(t0, t1);
    var t = max(max(lo.x, max(lo.y, lo.z)), 0.0);
    let t_end = min(hi.x, min(hi.y, hi.z));

    // How many pixels one unit covers at one unit of distance (from the
    // projection), to measure near misses in pixels.
    let per_unit = 0.5 * u.viewport.y * length(vec3<f32>(u.mvp[0][1], u.mvp[1][1], u.mvp[2][1]));

    // Sphere tracing: the distance to the surface is a step that cannot overshoot.
    // The closest approach is remembered: a ray that misses the shape by less
    // than a pixel still covers part of that pixel — the smooth outline.
    var hit = false;
    var closest = 1e9;        // smallest distance to the surface seen, in pixels
    var t_closest = t;
    for (var n = 0; n < 96; n++) {
        let d = sd_shape(kind, origin + direction * t, head.yzw);
        if (d < 0.0004 * t + 1e-5) { hit = true; break; }
        let pixels = d * per_unit / max(t, 1e-4);
        if (pixels < closest) {
            closest = pixels;
            t_closest = t;
        }
        t += d;
        if (t > t_end) { break; }
    }
    var coverage = 1.0;
    if (!hit) {
        coverage = 1.0 - closest;
        if (coverage <= 0.01) { discard; }
        t = t_closest;        // shade the point where the ray passed closest
    }

    // The normal is the direction in which the distance grows fastest.
    let p = origin + direction * t;
    let e = vec2<f32>(max(0.0005 * t, 1e-4), 0.0);
    let local_normal = normalize(vec3<f32>(
        sd_shape(kind, p + e.xyy, head.yzw) - sd_shape(kind, p - e.xyy, head.yzw),
        sd_shape(kind, p + e.yxy, head.yzw) - sd_shape(kind, p - e.yxy, head.yzw),
        sd_shape(kind, p + e.yyx, head.yzw) - sd_shape(kind, p - e.yyx, head.yzw)));
    let normal = ax * local_normal.x + ay * local_normal.y + az * local_normal.z;
    let world = u.eye.xyz + ray * t;

    var out: FOut;
    let clip = to_wgpu_depth(u.mvp * vec4<f32>(world, 1.0));
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    let surface = make_surface(world, normal);
    out.color = vec4<f32>(finish(surface, shade(surface)), coverage);
    return out;
}
"""


# The shadow map pass: positions only, seen from the light, depths kept.
SHADOW_PASS = """
struct Caster {
    matrix: mat4x4<f32>,
    passes: vec4<f32>,        // see-through solids: the colour of the light they let through
};
@group(0) @binding(0) var<uniform> caster: Caster;

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
    return caster.matrix * vec4<f32>(position, 1.0);
}

// Only used by the pass of the see-through solids.
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return caster.passes;
}
"""


# ------------------------------------------------------------- shader system
class Shader:
    """A look for solid shapes: WGSL source that defines ``shade``.

    ``source`` must contain ``fn shade(s: Surface) -> vec3<f32>`` and may
    define other functions next to it. Make it once, outside ``draw_frame``,
    and pass it to any solid: ``ui.draw_box(..., shader=my_shader)``.
    A mistake in the source is reported by the graphics driver, with the line,
    the first time the shader is used.
    """

    def __init__(self, source: str, name: str = "custom") -> None:
        if "fn shade" not in source:
            raise ValueError("a shader must define: fn shade(s: Surface) -> vec3<f32>")
        self.source = source
        self.name = name
        # A shader that looks at the triangles' edges needs triangles: solids
        # that use it are not ray-marched (there are no triangles in a formula).
        self.triangles = "s.edge" in source or "wire(" in source
        # Four programs from one source: triangles or ray-marched, opaque or see-through.
        self.sources = {
            (marched, glass): _COMMON + source + (_GLASS if glass else _OPAQUE) + (_MARCH if marched else _MESH)
            for marched in (False, True)
            for glass in (False, True)
        }

    def __repr__(self) -> str:
        return f"Shader({self.name!r})"


SHADERS: dict[str, Shader] = {
    # The default: ambient + light + a shiny highlight (and shadows when ray-marched).
    "lit": Shader("fn shade(s: Surface) -> vec3<f32> { return lit(s); }", "lit"),
    # The plain colour, no light at all.
    "flat": Shader("fn shade(s: Surface) -> vec3<f32> { return s.color; }", "flat"),
    # Cartoon: the light in three flat steps and a hard highlight.
    "toon": Shader(
        """
fn shade(s: Surface) -> vec3<f32> {
    let level = floor(diffuse(s) * 3.0 + 0.5) / 3.0;
    let spark = step(0.5, highlight(s));
    return s.color * (s.ambient + (1.0 - s.ambient) * level) * s.occlusion + spark * s.light_color;
}
""",
        "toon",
    ),
    # The normal as a colour: the classic way to check a shape or a file.
    "normals": Shader(
        "fn shade(s: Surface) -> vec3<f32> { return 0.5 + 0.5 * s.normal; }", "normals"
    ),
    # The default look plus a glowing outline where the surface turns away.
    "rim": Shader(
        """
fn shade(s: Surface) -> vec3<f32> {
    let edge = pow(1.0 - max(dot(s.normal, s.view), 0.0), 3.0);
    return lit(s) + edge * s.light_color * 0.8;
}
""",
        "rim",
    ),
}


SHADERS["wireframe"] = Shader(
    # Only the edges of the triangles are drawn; the rest is thrown away, so
    # the far side of the shape shows through.
    """
fn shade(s: Surface) -> vec3<f32> {
    if (wire(s, 1.0) < 0.5) { discard; }
    return s.color;
}
""",
    "wireframe",
)
SHADERS["glow"] = Shader(
    # Lit from inside, like a neon tube: white-hot where it faces you, the
    # colour toward the rim. Ignores the light and the shadows; it breathes
    # slowly when frames are drawn (--poll).
    """
fn shade(s: Surface) -> vec3<f32> {
    let facing = max(dot(s.normal, s.view), 0.0);
    let core = mix(s.color, vec3<f32>(1.0), 0.75 * pow(facing, 3.0));
    let halo = s.color * pow(1.0 - facing, 2.0) * 0.6;
    return (core + halo) * (0.9 + 0.1 * sin(s.time * 2.5));
}
""",
    "glow",
)


def resolve(shader) -> Shader:
    """A :class:`Shader` from a name, a Shader or ``None`` (the default look)."""
    if shader is None:
        return SHADERS["lit"]
    if isinstance(shader, Shader):
        return shader
    try:
        return SHADERS[shader]
    except KeyError:
        raise ValueError(
            f"unknown shader {shader!r}: use one of {', '.join(SHADERS)} or a Shader"
        ) from None
