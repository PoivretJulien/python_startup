"""Pictures (textures) for the 3D drawing API.

The scene is stroke geometry; a picture is the one primitive that needs
something else: decoded pixels, a UV-mapped quad and a GL texture. This module
owns those pieces so both renderers share them:

* :func:`load_pixels` normalises a file path, a Pillow image or a NumPy array
  to an ``(H, W, 4)`` uint8 RGBA array;
* :class:`Texture` uploads that array once (mipmapped, clamped to the edge);
* :class:`TextureStore` keeps uploaded textures between frames — a file is
  re-read only when it changes on disk, while an in-memory image is cached by
  identity (pass the same array/image every frame, or a :class:`Texture`);
* :class:`ImageBatch` is the textured-quad program and its VAO, used by
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
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from gl_utils3d import Quads, build_program
from math3d import to_gl
from OpenGL.GL import (
    GL_BLEND,
    GL_CLAMP_TO_EDGE,
    GL_DEPTH_WRITEMASK,
    GL_FALSE,
    GL_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGBA,
    GL_RGBA8,
    GL_SRC_ALPHA,
    GL_TEXTURE0,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S,
    GL_TEXTURE_WRAP_T,
    GL_TRUE,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    glBindTexture,
    glBlendFunc,
    glDeleteProgram,
    glDeleteTextures,
    glDepthMask,
    glDisable,
    glEnable,
    glGenerateMipmap,
    glGenTextures,
    glGetBooleanv,
    glGetUniformLocation,
    glIsEnabled,
    glPixelStorei,
    glTexImage2D,
    glTexParameteri,
    glUniform1i,
    glUniform4f,
    glUniformMatrix4fv,
    glUseProgram,
)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    # A path, a Pillow image, or an RGBA-ish NumPy array.
    ImageSource = str | bytes | os.PathLike[str] | np.ndarray | PILImage
else:
    ImageSource = Any

Color4 = tuple[float, float, float, float]
Tint = Sequence[float]  # (r, g, b) or (r, g, b, a)

# --------------------------------------------------------------------- shaders
# The quad is real 3D geometry (it is depth-tested and written like the
# strokes), so the vertex shader is one MVP transform and a UV pass-through.
IMAGE_VERTEX_SHADER = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;

uniform mat4 mvp;

out vec2 v_uv;

void main() {
    gl_Position = mvp * vec4(position, 1.0);
    v_uv = texcoord;
}
"""

# The texture already carries straight alpha; blending is enabled around the
# draw by ImageBatch. Fully transparent texels are killed so a cut-out picture
# can still write depth (the alpha test is free at this fragment count).
IMAGE_FRAGMENT_SHADER = """#version 330 core
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D image;
uniform vec4 tint;      // rgba multiplier; (1, 1, 1, 1) = the picture as-is

void main() {
    vec4 texel = texture(image, v_uv) * tint;
    if (texel.a <= 0.003) {
        discard;
    }
    frag_color = texel;
}
"""


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
class Texture:
    """One image uploaded to the GPU once; safe to draw many times.

    The constructor takes anything :func:`load_pixels` accepts, so a texture can
    be created from a file, a Pillow image or an array and then reused. A
    ``Texture`` is also the explicit escape hatch for in-memory images: pass it
    to ``draw_image`` and no per-frame cache lookup is needed. Call
    :meth:`destroy` (or pass it to a store) when it is no longer used — the GL
    object lives until then.
    """

    def __init__(self, source: ImageSource) -> None:
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

        self.texture = int(glGenTextures(1))
        if not self.texture:
            raise RuntimeError("could not create an OpenGL texture")
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA8,
            self.width,
            self.height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            pixels.tobytes(),
        )
        # Mipmaps keep a picture lying flat on the floor from shimmering when
        # the camera looks across it at a grazing angle.
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    @property
    def aspect(self) -> float:
        """Width / height of the picture in pixels."""
        return self.width / max(self.height, 1)

    def destroy(self) -> None:
        """Delete the GL texture. Drawing it afterwards is a GL error."""
        if self.texture:
            glDeleteTextures(1, [self.texture])
            self.texture = 0

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
    """The textured-quad program + VAO shared by both renderers.

    One instance per renderer (the program and texture unit are GL state).
    :meth:`draw` binds the camera MVP, the texture and the tint, turns blending
    on just for the quad, and restores the depth-write and blend flags it
    changed — so it composes with the stroke and point batches around it.
    """

    # Any unit works: no other batch in this app samples a texture.
    TEXTURE_UNIT = 3

    def __init__(self) -> None:
        self.program = build_program(IMAGE_VERTEX_SHADER, IMAGE_FRAGMENT_SHADER)
        self.mvp_location = glGetUniformLocation(self.program, "mvp")
        self.tint_location = glGetUniformLocation(self.program, "tint")
        self.sampler_location = glGetUniformLocation(self.program, "image")
        self.quads = Quads()
        glUseProgram(self.program)
        glUniform1i(self.sampler_location, self.TEXTURE_UNIT)

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
        if not vertices.shape[0]:
            return
        tint4 = (
            float(tint[0]),
            float(tint[1]),
            float(tint[2]),
            float(tint[3]) if len(tint) > 3 else 1.0,
        )
        if depth_write is None:
            depth_write = not texture.has_alpha and tint4[3] >= 1.0

        glUseProgram(self.program)
        glUniformMatrix4fv(self.mvp_location, 1, GL_FALSE, to_gl(mvp))
        glUniform4f(self.tint_location, *tint4)
        glActiveTexture(cast(int, GL_TEXTURE0) + self.TEXTURE_UNIT)
        glBindTexture(GL_TEXTURE_2D, texture.texture)

        # Both renderers disagree about blending being on by default (the AA one
        # enables it in __init__), so save and restore instead of assuming.
        blend_was_enabled = bool(glIsEnabled(GL_BLEND))
        if not blend_was_enabled:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        depth_mask = bool(np.asarray(glGetBooleanv(GL_DEPTH_WRITEMASK)).ravel()[0])
        if not depth_write:
            glDepthMask(GL_FALSE)

        self.quads.upload(vertices)
        self.quads.draw()

        if not depth_write and depth_mask:
            glDepthMask(GL_TRUE)
        if not blend_was_enabled:
            glDisable(GL_BLEND)
        glBindTexture(GL_TEXTURE_2D, 0)

    def close(self) -> None:
        self.quads.destroy()
        glDeleteProgram(self.program)
        self.program = 0

    def __enter__(self) -> ImageBatch:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
