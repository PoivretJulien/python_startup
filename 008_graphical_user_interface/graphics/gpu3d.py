"""wgpu device, window surface and the per-frame draw recorder.

This module replaces every piece of OpenGL state the app used to rely on:

* **context creation** — instead of an OpenGL 3.3 core context, the GLFW
  window is created with ``CLIENT_API = NO_API`` and wgpu (the Rust
  ``wgpu-native`` library behind ``wgpu.backends.wgpu_native``) builds a
  native surface on it;
* **backend choice** — Metal on macOS, Vulkan on Linux (GL/GLES as a last
  resort when no Vulkan driver is installed), DX12 then Vulkan on Windows.
  ``--backend`` or the ``WGPU_BACKEND`` environment variable overrides it;
* **immediate-mode drawing** — OpenGL executed every ``glDrawArrays`` at once.
  WebGPU records commands into a render pass instead, so every renderer call
  becomes a :class:`DrawCall` appended to the frame; :meth:`GPU.end_frame`
  uploads all of them, encodes **one** render pass in the same order and
  presents. Draw order is preserved, so the scene code does not notice.

The public drawing API (``draw_line``, ``draw_points`` ...) lives in the
renderers and is unchanged; only ``main.py`` sees this module, to create the
``GPU`` object and hand it to them.

Conventions kept from the OpenGL version, so ``math3d`` / ``camera3d`` stay
untouched:

* matrices are built row-major with OpenGL clip depth ``[-1, 1]``; they are
  transposed at upload (``math3d.to_gl``) and every vertex shader remaps depth
  to WebGPU's ``[0, 1]`` with ``to_wgpu_depth``;
* the swap-chain uses a **non-sRGB** format when the surface offers one, which
  is what the GL default framebuffer was: colours look exactly as before.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from collections.abc import Callable, Sequence

import glfw
import numpy as np
import wgpu
from wgpu.backends.wgpu_native import extras as wgpu_extras

# --------------------------------------------------------------------- backends
# Order of preference per platform. The first backend that has an adapter wins.
# Metal is the native, non-deprecated API on macOS; Vulkan is the native API on
# Linux (and the one Mesa / NVIDIA / AMD ship). GL is kept as a Linux fallback
# for machines without a Vulkan ICD (old Intel iGPUs, some VMs): it is wgpu's
# GLES backend, not the deprecated desktop OpenGL path this port moves away from.
PLATFORM_BACKENDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("darwin", ("Metal",)),
    ("linux", ("Vulkan", "GL")),
    ("freebsd", ("Vulkan", "GL")),
    ("win32", ("DX12", "Vulkan", "GL")),
)

# User spelling -> wgpu-native InstanceBackend flag name.
BACKEND_NAMES = {
    "auto": "auto",
    "metal": "Metal",
    "vulkan": "Vulkan",
    "vk": "Vulkan",
    "dx12": "DX12",
    "d3d12": "DX12",
    "gl": "GL",
    "gles": "GL",
    "opengl": "GL",
}

# adapter.info["backend_type"] -> the same flag names as above.
_ADAPTER_BACKENDS = {
    "metal": "Metal",
    "vulkan": "Vulkan",
    "d3d12": "DX12",
    "dx12": "DX12",
    "opengl": "GL",
    "opengles": "GL",
    "gl": "GL",
}

_ADAPTER_TYPE_RANK = {"DiscreteGPU": 0, "IntegratedGPU": 1, "VirtualGPU": 2, "CPU": 3}

ENV_BACKEND = "WGPU_BACKEND"


def backend_preference(requested: str | None = "auto") -> tuple[str, ...]:
    """The backends to try, best first: an explicit request or the platform default."""
    requested = (requested or os.environ.get(ENV_BACKEND) or "auto").strip().lower()
    if requested not in BACKEND_NAMES:
        raise ValueError(
            f"unknown wgpu backend {requested!r}; choose one of "
            f"{', '.join(sorted(set(BACKEND_NAMES)))}"
        )
    if requested == "auto":
        env = os.environ.get(ENV_BACKEND, "").strip().lower()
        if env and env != "auto":
            return backend_preference(env)
        for prefix, order in PLATFORM_BACKENDS:
            if sys.platform.startswith(prefix):
                return order
        return ("Vulkan", "GL")
    return (BACKEND_NAMES[requested],)


def adapter_backend(adapter: wgpu.GPUAdapter) -> str:
    raw = str(adapter.info.get("backend_type", "")).lower()
    return _ADAPTER_BACKENDS.get(raw, raw or "?")


def pick_adapter(
    order: Sequence[str],
    power_preference: str = "high-performance",
) -> wgpu.GPUAdapter:
    """Choose the best adapter for the first backend in ``order`` that has one.

    ``set_instance_extras`` restricts the wgpu instance to ``order`` before it
    exists; then the adapters are ranked by backend preference, and by GPU type
    (discrete first for "high-performance", integrated first for "low-power").
    """
    try:
        wgpu_extras.set_instance_extras(backends=list(order))
    except Exception as exc:  # pragma: no cover - the instance already exists
        warnings.warn(f"wgpu instance already created, backend filter ignored: {exc}")

    adapters = wgpu.gpu.enumerate_adapters_sync()
    if not adapters:
        hint = (
            "install a Vulkan driver (Debian/Ubuntu: sudo apt install mesa-vulkan-drivers "
            "libvulkan1; Fedora: sudo dnf install mesa-vulkan-drivers vulkan-loader) "
            "or try --backend gl"
            if sys.platform.startswith("linux")
            else "update the GPU driver / OS"
        )
        raise RuntimeError(f"no wgpu adapter found for backends {list(order)}: {hint}")

    type_rank = dict(_ADAPTER_TYPE_RANK)
    if power_preference == "low-power":
        type_rank["IntegratedGPU"], type_rank["DiscreteGPU"] = 0, 1

    def rank(adapter: wgpu.GPUAdapter) -> tuple[int, int]:
        backend = adapter_backend(adapter)
        backend_rank = order.index(backend) if backend in order else len(order)
        return backend_rank, type_rank.get(str(adapter.info.get("adapter_type")), 9)

    return min(adapters, key=rank)


# --------------------------------------------------------------------- uniforms
# One uniform block shared by every pipeline (see shaders3d.FRAME_UNIFORMS).
# Each draw gets its own 256-byte slot in one per-frame buffer, selected with a
# dynamic offset, so a frame with hundreds of draws is still one upload.
UNIFORM_FLOATS = 40
UNIFORM_BYTES = UNIFORM_FLOATS * 4
# The slot every draw owns in the per-frame buffer. Most shaders read the first
# UNIFORM_BYTES of it; the solids (solid_shaders3d) use all of it.
UNIFORM_SLOT_FLOATS = 64
UNIFORM_SLOT_BYTES = UNIFORM_SLOT_FLOATS * 4


def pack_uniforms(
    *,
    mvp: np.ndarray | None = None,
    eye: Sequence[float] = (0.0, 0.0, 0.0),
    forward: Sequence[float] = (0.0, 0.0, -1.0),
    tint: Sequence[float] = (1.0, 1.0, 1.0, 1.0),
    viewport: Sequence[float] = (1.0, 1.0),
    screen: Sequence[float] = (1.0, 1.0),
    pixel_scale: float = 1.0,
    min_radius_px: float = 0.0,
    max_radius_px: float = 1.0,
    feather: float = 0.0,
    flat_depth: bool = False,
) -> np.ndarray:
    """Pack the shared ``Frame`` uniform struct as ``UNIFORM_FLOATS`` float32."""
    out = np.zeros(UNIFORM_FLOATS, dtype=np.float32)
    if mvp is None:
        out[0:16] = np.eye(4, dtype=np.float32).ravel()
    else:
        # Row-major math -> column-major WGSL mat4x4, the same single transpose
        # math3d.to_gl does for OpenGL.
        out[0:16] = np.asarray(mvp, dtype=np.float32).T.ravel()
    out[16:19] = eye[:3]
    out[20:23] = forward[:3]
    out[24:28] = (*tint[:3], tint[3] if len(tint) > 3 else 1.0)
    out[28:30] = viewport[:2]
    out[30:32] = screen[:2]
    out[32] = pixel_scale
    out[33] = min_radius_px
    out[34] = max_radius_px
    out[35] = feather
    out[36] = 1.0 if flat_depth else 0.0
    return out


# --------------------------------------------------------------------- pipeline
BLEND_ALPHA = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
}


def vertex_layout(
    stride_floats: int,
    attributes: Sequence[tuple[int, int, int]],
    *,
    instanced: bool = False,
) -> dict:
    """A vertex-buffer layout from ``(location, components, float_offset)`` triples.

    The same triples the GL ``Strokes`` / ``Points`` classes fed to
    ``glVertexAttribPointer``; ``instanced`` advances the buffer once per
    instance (one particle) instead of once per vertex.
    """
    formats = {1: "float32", 2: "float32x2", 3: "float32x3", 4: "float32x4"}
    return {
        "array_stride": stride_floats * 4,
        "step_mode": "instance" if instanced else "vertex",
        "attributes": [
            {"format": formats[size], "offset": offset * 4, "shader_location": location}
            for location, size, offset in attributes
        ],
    }


@dataclasses.dataclass(frozen=True)
class PipelineSpec:
    """Everything that is baked into an immutable wgpu render pipeline.

    OpenGL toggled blending / depth writes / depth test as global state around
    each draw; WebGPU bakes them into the pipeline, so every combination the
    renderers use is one cached pipeline.
    """

    shader: str  # WGSL source
    layout: tuple  # vertex layout, frozen (see GPU.pipeline)
    topology: str = "triangle-list"
    blend: bool = False
    depth_write: bool = True
    depth_compare: str = "less-equal"
    textured: bool = False
    scene: bool = False  # group 1 = the solids' scene buffer (see solid_renderer3d)
    cull_mode: str = "none"
    backdrop: bool = False  # group 2 = the picture behind (see-through solids)


@dataclasses.dataclass(slots=True)
class DrawCall:
    """One recorded draw: what ``glDrawArrays`` used to execute immediately."""

    pipeline: wgpu.GPURenderPipeline
    uniforms: np.ndarray
    vertex_count: int
    instance_count: int = 1
    vertices: np.ndarray | None = None  # transient data, uploaded at end_frame
    buffer: wgpu.GPUBuffer | None = None  # or a persistent buffer (the grid)
    texture_group: wgpu.GPUBindGroup | None = None
    # See-through solids: drawn in a second pass, where the picture made by
    # everything before them can be read. `order` sorts them, far ones first.
    backdrop: bool = False
    order: float = 0.0


_BLIT = """
@group(1) @binding(0) var picture: texture_2d<f32>;
@group(1) @binding(1) var picture_sampler: sampler;

struct VOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VOut {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: VOut;
    out.position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

@fragment
fn fs_main(i: VOut) -> @location(0) vec4<f32> {
    return textureSampleLevel(picture, picture_sampler, i.uv, 0.0);
}
"""

_current: GPU | None = None


def current() -> GPU:
    """The live :class:`GPU` (textures created without an explicit device use it)."""
    if _current is None:
        raise RuntimeError("no GPU yet: create gpu3d.GPU(window) before drawing")
    return _current


def _present_info(window, vsync: bool) -> dict:
    """The native window handles wgpu needs, from rendercanvas' GLFW helper."""
    from rendercanvas.glfw import get_glfw_present_info

    info = dict(get_glfw_present_info(window))
    info["vsync"] = bool(vsync)
    return info


class GPU:
    """The wgpu device, the window surface and the frame being recorded."""

    DEPTH_FORMAT = "depth24plus"

    def __init__(
        self,
        window,
        *,
        backend: str | None = "auto",
        vsync: bool = True,
        power_preference: str = "high-performance",
    ) -> None:
        global _current
        self.window = window
        self.backend_order = backend_preference(backend)
        self.adapter = pick_adapter(self.backend_order, power_preference)
        self.backend = adapter_backend(self.adapter)

        limits = self.adapter.limits
        self.device = self.adapter.request_device_sync(
            required_limits={
                # A 1 M particle frame is ~28 MB of vertices: ask for what the
                # adapter really has instead of WebGPU's 256 MB default floor.
                "max-buffer-size": limits["max-buffer-size"],
            },
        )
        self.queue = self.device.queue
        self.uniform_stride = max(256, int(limits["min-uniform-buffer-offset-alignment"]))

        self.context = wgpu.gpu.get_canvas_context(_present_info(window, vsync))
        self.format = self._configure_surface()

        # Shared layouts: group 0 = per-draw uniforms (dynamic offset),
        # group 1 = a picture's texture + sampler.
        visibility = wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT
        self.uniform_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": visibility,
                    "buffer": {
                        "type": "uniform",
                        "has_dynamic_offset": True,
                        "min_binding_size": UNIFORM_SLOT_BYTES,
                    },
                }
            ]
        )
        self.texture_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "float", "view_dimension": "2d"},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "filtering"},
                },
            ]
        )
        # Solids: group 1 = one uniform buffer describing every solid of the
        # frame, read by the fragment shader for shadows (ray-marched mode).
        self.scene_layout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": "uniform"},
                },
                # The shadow map: the scene's depths seen from the light.
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {"sample_type": "depth", "view_dimension": "2d"},
                },
                {
                    "binding": 2,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {"type": "comparison"},
                },
            ]
        )
        self.shadow_sampler = self.device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
            compare="less-equal",
        )
        self._layouts = {
            False: self.device.create_pipeline_layout(bind_group_layouts=[self.uniform_layout]),
            True: self.device.create_pipeline_layout(
                bind_group_layouts=[self.uniform_layout, self.texture_layout]
            ),
            "scene": self.device.create_pipeline_layout(
                bind_group_layouts=[self.uniform_layout, self.scene_layout]
            ),
            "backdrop": self.device.create_pipeline_layout(
                bind_group_layouts=[self.uniform_layout, self.scene_layout, self.texture_layout]
            ),
        }
        self.sampler = self.device.create_sampler(
            address_mode_u="clamp-to-edge",
            address_mode_v="clamp-to-edge",
            mag_filter="linear",
            min_filter="linear",
            mipmap_filter="linear",
        )

        self._modules: dict[str, wgpu.GPUShaderModule] = {}
        self._pipelines: dict[PipelineSpec, wgpu.GPURenderPipeline] = {}

        # Per-frame arenas, grown on demand and reused across frames.
        self._vertex_arena: wgpu.GPUBuffer | None = None
        self._uniform_arena: wgpu.GPUBuffer | None = None
        self._uniform_group: wgpu.GPUBindGroup | None = None
        self._depth: wgpu.GPUTexture | None = None
        self._depth_view = None
        # The picture behind see-through solids; made only when one is drawn.
        self._backdrop: wgpu.GPUTexture | None = None
        self._backdrop_view = None
        self._backdrop_group: wgpu.GPUBindGroup | None = None
        self._blit_pipeline: wgpu.GPURenderPipeline | None = None

        self.clear_color: Sequence[float] = (0.0, 0.0, 0.0, 1.0)
        self._calls: list[DrawCall] = []
        self._frame_open = False
        # A number that changes with every frame, and functions called just
        # before a frame is encoded: how a renderer uploads per-frame data that
        # is only complete once every draw call of the frame has been made.
        self.frame_serial = 0
        self.before_end_frame: list[Callable[[], None]] = []
        # Functions given the frame's command encoder before the main pass:
        # how a renderer adds a pass of its own (the shadow map).
        self.before_main_pass: list[Callable[[wgpu.GPUCommandEncoder], None]] = []
        self.frames_presented = 0
        self.frames_skipped = 0
        _current = self

    # ----------------------------------------------------------------- setup
    def _configure_surface(self) -> str:
        """Configure the swap-chain, preferring a non-sRGB twin of the preferred format."""
        preferred = self.context.get_preferred_format(self.adapter)
        candidates = [preferred.removesuffix("-srgb"), preferred, "bgra8unorm", "rgba8unorm"]
        last_error: Exception | None = None
        for fmt in dict.fromkeys(candidates):
            try:
                self.context.configure(device=self.device, format=fmt, alpha_mode="opaque")
                return fmt
            except ValueError as exc:
                last_error = exc
        raise RuntimeError(f"could not configure the window surface: {last_error}")

    @property
    def summary(self) -> str:
        info = self.adapter.info
        return (
            f"{info.get('device', '?')} ({info.get('adapter_type', '?')}) via {self.backend}"
            f" · surface {self.format}"
        )

    def framebuffer_size(self) -> tuple[int, int]:
        width, height = glfw.get_framebuffer_size(self.window)
        return int(width), int(height)

    def window_size(self) -> tuple[int, int]:
        width, height = glfw.get_window_size(self.window)
        return int(width), int(height)

    def shader(self, source: str) -> wgpu.GPUShaderModule:
        module = self._modules.get(source)
        if module is None:
            module = self.device.create_shader_module(code=source)
            self._modules[source] = module
        return module

    def pipeline(
        self,
        shader: str,
        layout: dict,
        *,
        topology: str = "triangle-list",
        blend: bool = False,
        depth_write: bool = True,
        depth_compare: str = "less-equal",
        textured: bool = False,
        scene: bool = False,
        cull_mode: str = "none",
        backdrop: bool = False,
    ) -> wgpu.GPURenderPipeline:
        """A cached render pipeline: the WebGPU form of "program + GL state"."""
        frozen = (
            layout["array_stride"],
            layout["step_mode"],
            tuple(tuple(sorted(a.items())) for a in layout["attributes"]),
        )
        spec = PipelineSpec(
            shader, frozen, topology, blend, depth_write, depth_compare, textured, scene, cull_mode,
            backdrop,
        )
        cached = self._pipelines.get(spec)
        if cached is not None:
            return cached

        module = self.shader(shader)
        target: dict = {"format": self.format}
        if blend:
            target["blend"] = BLEND_ALPHA
        pipeline = self.device.create_render_pipeline(
            layout=self._layouts["backdrop" if backdrop else "scene" if scene else textured],
            vertex={"module": module, "entry_point": "vs_main", "buffers": [layout]},
            primitive={"topology": topology, "cull_mode": cull_mode},
            depth_stencil={
                "format": self.DEPTH_FORMAT,
                "depth_write_enabled": depth_write,
                "depth_compare": depth_compare,
            },
            multisample={"count": 1},
            fragment={"module": module, "entry_point": "fs_main", "targets": [target]},
        )
        self._pipelines[spec] = pipeline
        return pipeline

    def create_vertex_buffer(self, vertices: np.ndarray) -> wgpu.GPUBuffer:
        """A persistent vertex buffer (the grid): uploaded once, drawn every frame."""
        data = np.ascontiguousarray(vertices, dtype=np.float32)
        buffer = self.device.create_buffer(
            size=max(_align(data.nbytes, 16), 16),
            usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
        )
        if data.nbytes:
            self.queue.write_buffer(buffer, 0, data)
        return buffer

    # ----------------------------------------------------------------- frame
    def begin_frame(self) -> None:
        """Start recording; replaces ``glClear`` (the pass clears on load)."""
        self._calls.clear()
        self._frame_open = True
        self.frame_serial += 1

    def submit(self, call: DrawCall) -> None:
        """Queue one draw in API order. Outside a frame it is dropped."""
        if self._frame_open and call.vertex_count and call.instance_count:
            self._calls.append(call)

    def end_frame(self) -> bool:
        """Upload, encode one render pass, submit and present. False = frame skipped."""
        calls, self._calls = self._calls, []
        if not self._frame_open:
            return False
        self._frame_open = False

        width, height = self.framebuffer_size()
        if width <= 0 or height <= 0:  # minimised: nothing to present into
            self.frames_skipped += 1
            return False
        self.context.set_physical_size(width, height)
        try:
            target = self.context.get_current_texture()
        except wgpu.DrawCancelled:  # occluded window: skip this frame
            self.frames_skipped += 1
            return False

        for flush in self.before_end_frame:
            flush()
        vertex_offsets = self._upload_vertices(calls)
        self._upload_uniforms(calls)

        encoder = self.device.create_command_encoder()
        for extra_pass in self.before_main_pass:
            extra_pass(encoder)
        clear = (*self.clear_color[:3], self.clear_color[3] if len(self.clear_color) > 3 else 1.0)
        depth_view = self._depth_target(width, height)
        numbered = list(enumerate(calls))

        def draw(render_pass, items) -> None:
            current_pipeline = None
            for index, call in items:
                if call.pipeline is not current_pipeline:
                    render_pass.set_pipeline(call.pipeline)
                    current_pipeline = call.pipeline
                render_pass.set_bind_group(0, self._uniform_group, [index * self.uniform_stride])
                if call.texture_group is not None:
                    render_pass.set_bind_group(1, call.texture_group)
                if call.backdrop:
                    render_pass.set_bind_group(2, self._backdrop_group)
                if call.buffer is not None:
                    render_pass.set_vertex_buffer(0, call.buffer)
                else:
                    offset, size = vertex_offsets[index]
                    render_pass.set_vertex_buffer(0, self._vertex_arena, offset, size)
                render_pass.draw(call.vertex_count, call.instance_count)

        def begin(view, first: bool, last: bool):
            return encoder.begin_render_pass(
                color_attachments=[
                    {"view": view, "clear_value": clear, "load_op": "clear", "store_op": "store"}
                ],
                depth_stencil_attachment={
                    "view": depth_view,
                    "depth_clear_value": 1.0,
                    "depth_load_op": "clear" if first else "load",
                    "depth_store_op": "discard" if last else "store",
                },
            )

        see_through = [item for item in numbered if item[1].backdrop]
        if not see_through:
            # The usual frame: one pass, straight to the window.
            render_pass = begin(target.create_view(), True, True)
            draw(render_pass, numbered)
            render_pass.end()
        else:
            # 1. Everything drawn before the last see-through solid goes to a
            #    texture: "the picture behind".
            # 2. That picture is copied to the window, then the see-through
            #    solids are drawn reading it (far ones first), then whatever
            #    came after them (the interface) on top.
            last = see_through[-1][0]
            self._backdrop_target(width, height)
            render_pass = begin(self._backdrop_view, True, False)
            draw(render_pass, [item for item in numbered[:last] if not item[1].backdrop])
            render_pass.end()

            render_pass = begin(target.create_view(), False, True)
            render_pass.set_pipeline(self._blit())
            render_pass.set_bind_group(0, self._uniform_group, [0])
            render_pass.set_bind_group(1, self._backdrop_group)
            render_pass.draw(3, 1)
            draw(render_pass, sorted(see_through, key=lambda item: -item[1].order))
            draw(render_pass, numbered[last + 1 :])
            render_pass.end()

        self.queue.submit([encoder.finish()])
        self.context.present()
        self.frames_presented += 1
        return True

    def _upload_vertices(self, calls: list[DrawCall]) -> dict[int, tuple[int, int]]:
        sizes = [
            (i, c.vertices) for i, c in enumerate(calls) if c.buffer is None and c.vertices is not None
        ]
        total = sum(_align(v.nbytes, 16) for _i, v in sizes)
        if total == 0:
            return {}
        if self._vertex_arena is None or self._vertex_arena.size < total:
            if self._vertex_arena is not None:
                self._vertex_arena.destroy()
            self._vertex_arena = self.device.create_buffer(
                size=_grow(total),
                usage=wgpu.BufferUsage.VERTEX | wgpu.BufferUsage.COPY_DST,
            )
        offsets: dict[int, tuple[int, int]] = {}
        cursor = 0
        for index, vertices in sizes:
            data = np.ascontiguousarray(vertices, dtype=np.float32)
            self.queue.write_buffer(self._vertex_arena, cursor, data)
            offsets[index] = (cursor, data.nbytes)
            cursor += _align(data.nbytes, 16)
        return offsets

    def _upload_uniforms(self, calls: list[DrawCall]) -> None:
        stride_floats = self.uniform_stride // 4
        block = np.zeros((max(len(calls), 1), stride_floats), dtype=np.float32)
        for index, call in enumerate(calls):
            block[index, : call.uniforms.size] = call.uniforms
        if self._uniform_arena is None or self._uniform_arena.size < block.nbytes:
            if self._uniform_arena is not None:
                self._uniform_arena.destroy()
            self._uniform_arena = self.device.create_buffer(
                size=_grow(block.nbytes),
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
            )
            self._uniform_group = self.device.create_bind_group(
                layout=self.uniform_layout,
                entries=[
                    {
                        "binding": 0,
                        "resource": {
                            "buffer": self._uniform_arena,
                            "offset": 0,
                            "size": UNIFORM_SLOT_BYTES,
                        },
                    }
                ],
            )
        self.queue.write_buffer(self._uniform_arena, 0, block)

    def _backdrop_target(self, width: int, height: int) -> None:
        if self._backdrop is None or self._backdrop.size[:2] != (width, height):
            if self._backdrop is not None:
                self._backdrop.destroy()
            self._backdrop = self.device.create_texture(
                size=(width, height, 1),
                format=self.format,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            )
            self._backdrop_view = self._backdrop.create_view()
            self._backdrop_group = self.device.create_bind_group(
                layout=self.texture_layout,
                entries=[
                    {"binding": 0, "resource": self._backdrop_view},
                    {"binding": 1, "resource": self.sampler},
                ],
            )

    def _blit(self) -> wgpu.GPURenderPipeline:
        """Copies the backdrop to the window: one triangle that covers the screen."""
        if self._blit_pipeline is None:
            module = self.shader(_BLIT)
            self._blit_pipeline = self.device.create_render_pipeline(
                layout=self._layouts[True],
                vertex={"module": module, "entry_point": "vs_main", "buffers": []},
                primitive={"topology": "triangle-list", "cull_mode": "none"},
                depth_stencil={
                    "format": self.DEPTH_FORMAT,
                    "depth_write_enabled": False,
                    "depth_compare": "always",
                },
                multisample={"count": 1},
                fragment={"module": module, "entry_point": "fs_main", "targets": [{"format": self.format}]},
            )
        return self._blit_pipeline

    def _depth_target(self, width: int, height: int):
        if self._depth is None or self._depth.size[:2] != (width, height):
            if self._depth is not None:
                self._depth.destroy()
            self._depth = self.device.create_texture(
                size=(width, height, 1),
                format=self.DEPTH_FORMAT,
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
            )
            self._depth_view = self._depth.create_view()
        return self._depth_view

    # --------------------------------------------------------------- cleanup
    def close(self) -> None:
        global _current
        for buffer in (self._vertex_arena, self._uniform_arena):
            if buffer is not None:
                buffer.destroy()
        if self._depth is not None:
            self._depth.destroy()
        if self._backdrop is not None:
            self._backdrop.destroy()
        self._vertex_arena = self._uniform_arena = self._depth = self._backdrop = None
        try:
            self.context.unconfigure()
        except Exception:  # pragma: no cover - best effort at shutdown
            pass
        if _current is self:
            _current = None

    def __enter__(self) -> GPU:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _grow(size: int) -> int:
    """Next power of two >= size (min 64 KiB): few reallocations as batches grow."""
    return max(1 << 16, 1 << (int(size) - 1).bit_length())


def run_once(make_frame: Callable[[], None], gpu: GPU) -> bool:
    """Record and present one frame (handy for tests and screenshots)."""
    gpu.begin_frame()
    make_frame()
    return gpu.end_frame()
