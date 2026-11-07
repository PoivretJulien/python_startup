"""Pictures (textures) for the 3D drawing API.

The scene is stroke geometry; a picture is the one primitive that needs
something else: decoded pixels, a UV-mapped quad and a GPU texture. This module
owns those pieces so both renderers share them:

* :func:`load_pixels` normalises a file path, a Pillow image or a NumPy array
  to an ``(H, W, 4)`` uint8 RGBA array;
* :class:`Texture` uploads that array once to wgpu (mipmapped, clamped to the edge);
* :class:`TextureStore` keeps uploaded textures between frames — a file is
  re-read only when it changes on disk, while an in-memory image is cached by
  identity (pass the same array/image every frame, or a :class:`Texture`);
* :class:`ImageBatch` is the textured-quad wgpu pipeline, used by
  ``renderer3d.GridArrowRenderer.draw_image`` and
  ``renderer_aa3d.AAGridArrowRenderer.draw_image``.

The quad itself is pure math and lives in
``geometry3d.image_quad_vertices``: ``x, y, z, u, v`` per vertex, two
triangles. ``anchor`` is the picture's bottom-left corner and ``right`` / ``up``
are its own axes, which is the same convention ``draw_text`` uses.

Pillow is imported only when a picture actually has to be decoded (or converted
from a Pillow image), so the rest of the app keeps working without it::

    system.draw_image(
        anchor=(-3.0, -1.5, 0.02),
        source="picture.png",
        right=(1.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        width=1.8,
        height=1.2,
    )
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import wgpu

import gpu3d
import shaders3d

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    # A path, a Pillow image, or an RGBA-ish NumPy array.
    ImageSource = str | bytes | os.PathLike[str] | np.ndarray | PILImage
else:
    ImageSource = Any

Color4 = tuple[float, float, float, float]
Tint = Sequence[float]  # (r, g, b) or (r, g, b, a)

# The textured-quad shader is shaders3d.IMAGE (WGSL).


# ------------------------------------------------------------------- decoding
def _pillow():
    """Import Pillow on demand, with an actionable error if it is missing."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "draw_image needs Pillow to decode pictures: pip install pillow"
        ) from exc
    return Image


def _rgba(array: np.ndarray) -> np.ndarray:
    """Normalise an ``(H, W)``, ``(H, W, 3)`` or ``(H, W, 4)`` array to RGBA8.

    Integer arrays are clipped to ``0..255``; floating arrays are read as
    ``0..1`` and scaled. Grayscale is expanded to RGB with an opaque alpha, and
    RGB gets an opaque alpha channel.
    """
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(
            f"a picture array must be (H, W), (H, W, 3) or (H, W, 4); got shape {array.shape}"
        )

    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            array = np.clip(array, 0.0, 1.0) * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] == 3:
        alpha = np.full((*array.shape[:2], 1), 255, dtype=np.uint8)
        array = np.concatenate((array, alpha), axis=2)
    return np.ascontiguousarray(array, dtype=np.uint8)


def load_pixels(source: ImageSource) -> np.ndarray:
    """Decode `source` to a contiguous ``(H, W, 4)`` uint8 RGBA array.

    Row 0 is the **top** row of the picture, exactly like a Pillow image and
    the image file itself. ``geometry3d.image_quad_vertices`` maps that row to
    ``v = 0``, so a picture drawn with ``up`` pointing up appears upright.
    """
    if isinstance(source, np.ndarray):
        return _rgba(source)
    if isinstance(source, Texture):
        return source.pixels

    Image = _pillow()
    if isinstance(source, (str, bytes, os.PathLike)):
        with Image.open(os.fsdecode(source)) as image:
            return _rgba(np.asarray(image.convert("RGBA"), dtype=np.uint8))
    if isinstance(source, Image.Image):
        return _rgba(np.asarray(source.convert("RGBA"), dtype=np.uint8))
    raise TypeError(
        "source must be a file path, a Pillow image, a NumPy array, or a "
        f"texture3d.Texture; got {type(source).__name__}"
    )


# -------------------------------------------------------------------- texture
def mip_chain(pixels: np.ndarray) -> list[np.ndarray]:
    """Level 0 plus every 2x2 box-filtered level down to 1x1.

    OpenGL had ``glGenerateMipmap``; WebGPU has no built-in equivalent, so the
    levels are averaged on the CPU once, at upload. Sizes follow the GPU rule
    ``max(1, size // 2)``, so an odd size drops its last row/column first.
    """
    levels = [pixels]
    current = pixels.astype(np.float32)
    while current.shape[0] > 1 or current.shape[1] > 1:
        if current.shape[0] % 2 and current.shape[0] > 1:
            current = current[:-1]
        if current.shape[1] % 2 and current.shape[1] > 1:
            current = current[:, :-1]
        h = max(current.shape[0] // 2, 1)
        w = max(current.shape[1] // 2, 1)
        rows = current.reshape(h, current.shape[0] // h, w, current.shape[1] // w, 4)
        current = rows.mean(axis=(1, 3))
        levels.append(np.ascontiguousarray(np.clip(current + 0.5, 0, 255).astype(np.uint8)))
    return levels


class Texture:
    """One image uploaded to the GPU once; safe to draw many times.

    The constructor takes anything :func:`load_pixels` accepts, so a texture can
    be created from a file, a Pillow image or an array and then reused. A
    ``Texture`` is also the explicit escape hatch for in-memory images: pass it
    to ``draw_image`` and no per-frame cache lookup is needed. Call
    :meth:`destroy` (or pass it to a store) when it is no longer used — the GPU
    object lives until then. ``gpu`` defaults to the live ``gpu3d.GPU``.
    """

    def __init__(self, source: ImageSource, gpu: gpu3d.GPU | None = None) -> None:
        pixels = load_pixels(source)
        height, width = pixels.shape[:2]
        if width <= 0 or height <= 0:
            raise ValueError("a picture must have at least one pixel")
        self.pixels = pixels
        self.width = int(width)
        self.height = int(height)
        # A picture with any non-opaque texel must not write depth: the
        # fragments behind it still have to be composited.
        self.has_alpha = bool(np.any(pixels[:, :, 3] != 255))

        self.gpu = gpu or gpu3d.current()
        levels = mip_chain(pixels)
        # rgba8unorm (not -srgb): sampled values are the file's bytes / 255,
        # exactly what the GL_RGBA8 texture returned.
        self.texture: wgpu.GPUTexture | None = self.gpu.device.create_texture(
            size=(self.width, self.height, 1),
            format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            mip_level_count=len(levels),
        )
        for level, data in enumerate(levels):
            h, w = data.shape[:2]
            self.gpu.queue.write_texture(
                {"texture": self.texture, "mip_level": level, "origin": (0, 0, 0)},
                data,
                {"offset": 0, "bytes_per_row": w * 4, "rows_per_image": h},
                (w, h, 1),
            )
        # Mipmaps keep a picture lying flat on the floor from shimmering when
        # the camera looks across it at a grazing angle.
        self.bind_group = self.gpu.device.create_bind_group(
            layout=self.gpu.texture_layout,
            entries=[
                {"binding": 0, "resource": self.texture.create_view()},
                {"binding": 1, "resource": self.gpu.sampler},
            ],
        )

    @property
    def aspect(self) -> float:
        """Width / height of the picture in pixels."""
        return self.width / max(self.height, 1)

    def destroy(self) -> None:
        """Release the GPU texture. Drawing it afterwards is an error."""
        if self.texture is not None:
            self.texture.destroy()
            self.texture = None
            self.bind_group = None

    def __enter__(self) -> Texture:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.destroy()

class TextureStore:
    """Uploaded textures for one renderer, kept between frames.

    Two caches, because the two source kinds need different keys:

    * **files** are keyed by absolute path and re-decoded only when the mtime or
      size changes, so editing a picture while the app runs shows the new one;
    * **in-memory images** (arrays, Pillow images) are keyed by identity, so the
      same object drawn every frame uploads once. An LRU cap bounds the cache
      when a scene keeps producing new images (a live camera feed, say); the
      oldest texture is deleted rather than leaked.

    Pass a :class:`Texture` to ``draw_image`` to bypass both caches.
    """

    MAX_IN_MEMORY = 16

    def __init__(self) -> None:
        self._files: dict[str, tuple[tuple[int, int], Texture]] = {}
        self._objects: OrderedDict[int, tuple[object, Texture]] = OrderedDict()

    def get(self, source: ImageSource) -> Texture:
        if isinstance(source, Texture):
            return source
        if isinstance(source, (str, bytes, os.PathLike)):
            return self._file_texture(os.path.abspath(os.fsdecode(source)))
        return self._object_texture(source)

    def _file_texture(self, path: str) -> Texture:
        try:
            stat = os.stat(path)
        except OSError as exc:
            raise FileNotFoundError(f"draw_image could not read picture file {path!r}") from exc
        version = (stat.st_mtime_ns, stat.st_size)
        cached = self._files.get(path)
        if cached is not None:
            cached_version, texture = cached
            if cached_version == version:
                return texture
            texture.destroy()  # the file changed on disk: re-upload it
        texture = Texture(path)
        self._files[path] = (version, texture)
        return texture

    def _object_texture(self, source: ImageSource) -> Texture:
        key = id(source)
        cached = self._objects.get(key)
        # Holding the source in the entry pins its id, so an evicted object can
        # never be mistaken for a new one that reuses the same address.
        if cached is not None and cached[0] is source:
            self._objects.move_to_end(key)
            return cached[1]
        texture = Texture(source)
        self._objects[key] = (source, texture)
        self._objects.move_to_end(key)
        while len(self._objects) > self.MAX_IN_MEMORY:
            _key, (_source, evicted) = self._objects.popitem(last=False)
            evicted.destroy()
        return texture

    def close(self) -> None:
        """Delete every texture this store uploaded (not user-owned Textures)."""
        for _version, texture in self._files.values():
            texture.destroy()
        for _source, texture in self._objects.values():
            texture.destroy()
        self._files.clear()
        self._objects.clear()


# ---------------------------------------------------------------- draw program
class ImageBatch:
    """The textured-quad pipeline shared by both renderers.

    :meth:`draw` records one blended draw of the quad with the camera MVP, the
    texture and the tint. OpenGL toggled ``GL_BLEND`` / ``glDepthMask`` around
    the call; here those are baked into two cached pipelines (depth write on /
    off), so nothing leaks into the stroke and point batches around it.
    """

    def __init__(self, gpu: gpu3d.GPU) -> None:
        self.gpu = gpu
        layout = gpu3d.vertex_layout(*shaders3d.QUAD_LAYOUT)
        self.pipelines = {
            depth_write: gpu.pipeline(
                shaders3d.IMAGE, layout, blend=True, depth_write=depth_write, textured=True
            )
            for depth_write in (True, False)
        }

    def draw(
        self,
        vertices: np.ndarray,
        texture: Texture,
        mvp: np.ndarray,
        tint: Tint = (1.0, 1.0, 1.0, 1.0),
        *,
        depth_write: bool | None = None,
    ) -> None:
        """Draw an ``x, y, z, u, v`` triangle batch with `texture` applied.

        ``mvp`` is the camera's row-major projection * view matrix (the same one
        the stroke shader receives). ``tint`` multiplies the texel; pass an
        ``(r, g, b)`` colour to tint the picture or an RGBA to fade it.

        ``depth_write`` defaults to ``True`` for an opaque picture and ``False``
        for one with transparency (blending needs the fragments behind it). A
        cut-out picture still writes depth, because fully transparent texels are
        discarded in the fragment shader.
        """
        if not vertices.shape[0] or texture.bind_group is None:
            return
        tint4 = (
            float(tint[0]),
            float(tint[1]),
            float(tint[2]),
            float(tint[3]) if len(tint) > 3 else 1.0,
        )
        if depth_write is None:
            depth_write = not texture.has_alpha and tint4[3] >= 1.0
        self.gpu.submit(
            gpu3d.DrawCall(
                pipeline=self.pipelines[bool(depth_write)],
                uniforms=gpu3d.pack_uniforms(mvp=mvp, tint=tint4),
                vertex_count=int(vertices.shape[0]),
                vertices=np.ascontiguousarray(vertices, dtype=np.float32),
                texture_group=texture.bind_group,
            )
        )

    def close(self) -> None:
        """Pipelines are owned by the GPU cache; nothing to release here."""

    def __enter__(self) -> ImageBatch:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
