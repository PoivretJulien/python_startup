"""Render one frame of main.py's scene off-screen and save it as a PNG.

    .venv/bin/python graphics/tools/snapshot.py out.png [--backend vulkan]

The window surface is swapped for an off-screen texture that can be copied
back, so this checks the real renderers and shaders without a screenshot tool.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glfw  # noqa: E402
import numpy as np  # noqa: E402
import wgpu  # noqa: E402
from PIL import Image  # noqa: E402

import main as app  # noqa: E402
from camera3d import OrbitCamera  # noqa: E402
from gpu3d import GPU  # noqa: E402
from renderer3d import GridArrowRenderer  # noqa: E402
from renderer_aa3d import AAGridArrowRenderer  # noqa: E402
from settings3d import BACKGROUND, CLOCK_COLOR  # noqa: E402
from window3d import close_window, create_window  # noqa: E402


def capture(gpu: GPU, draw) -> np.ndarray:
    width, height = gpu.framebuffer_size()
    target = gpu.device.create_texture(
        size=(width, height, 1),
        format=gpu.format,
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )
    gpu.context.get_current_texture = lambda: target
    gpu.context.present = lambda: None
    gpu.begin_frame()
    draw()
    gpu.end_frame()
    data = gpu.queue.read_texture(
        {"texture": target, "mip_level": 0, "origin": (0, 0, 0)},
        {"offset": 0, "bytes_per_row": width * 4, "rows_per_image": height},
        (width, height, 1),
    )
    pixels = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    if gpu.format.startswith("bgra"):
        pixels = pixels[:, :, [2, 1, 0, 3]]
    return pixels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--plain", action="store_true", help="use the aliased renderer")
    args = parser.parse_args()

    window = create_window()
    gpu = GPU(window, backend=args.backend)
    gpu.clear_color = (*BACKGROUND, 1.0)
    camera = OrbitCamera()
    plain = GridArrowRenderer(gpu, camera)
    aa = AAGridArrowRenderer(gpu, camera)
    system = plain if args.plain else aa

    def draw() -> None:
        system.draw_grid()
        app.draw_2D_rectangle(system, (3, 2), (5, 3), (1, 1, 0))
        system.draw_image(
            anchor=(-3.7, -0.4, 0.05), source=app.DEMO_PICTURE,
            right=(1.0, 0.0, 0.0), up=(0.0, 0.0, 1.0), width=1.8, height=1.15,
        )
        app.draw_clock(system, aa, 4000.0)
        system.draw_line((2.0, -1.0, 0.02), (3.0, -1.5, 0.02), CLOCK_COLOR, 0.05)
        app.draw_particle_wave(system, (0.0, -3.5, 0.45), 200, 200, 0.7,
                               spacing=0.011, size=0.03, amplitude=0.35)
        app.draw_particle_lifetime(system, (-3.0, 3.0, 0.75), 24, 24, 0.7, spacing=0.16,
                                   size=0.07, amplitude=0.18, wavelength=0.9,
                                   speed=0.25, fade_rate=0.30, power=1.6)
        app.draw_vector_array(system, (1.5, 1.5, 0.02))
        aa.draw_screen_text(16.0, 24.0, f"wgpu {gpu.summary}", 14.0, (1.0, 0.85, 0.40))

    try:
        pixels = capture(gpu, draw)
        Image.fromarray(pixels[:, :, :3]).save(args.out)
        print(f"saved {args.out} {pixels.shape[1]}x{pixels.shape[0]} · {gpu.summary}")
    finally:
        aa.close()
        plain.close()
        gpu.close()
        close_window(window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
