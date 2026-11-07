#!/usr/bin/env python3
"""Cross-platform probe for the GLFW + wgpu stack (Metal / Vulkan / DX12).

    python check_env.py              # packages, native libs, GPU adapters
    python check_env.py --window     # also open a window and present frames
    python check_env.py --backend vulkan

Exit code 0 when everything needed to run main.py is present.
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib
import importlib.metadata
import platform
import sys

GREEN, YELLOW, RED, ORANGE, RST = "\033[32m", "\033[33m", "\033[31m", "\033[38;5;208m", "\033[0m"
failures = 0


def say(text: str) -> None:
    print(f"{ORANGE}▸{RST} {text}")


def ok(text: str) -> None:
    print(f"  {GREEN}✓{RST} {text}")


def warn(text: str) -> None:
    print(f"  {YELLOW}!{RST} {text}")


def bad(text: str, fix: str = "") -> None:
    global failures
    failures += 1
    print(f"  {RED}✗{RST} {text}" + (f"\n      fix: {fix}" if fix else ""))


PACKAGES = (
    ("glfw", "glfw", "window + input"),
    ("wgpu", "wgpu", "WebGPU (wgpu-native, Rust)"),
    ("rendercanvas", "rendercanvas", "GLFW -> wgpu surface handles"),
    ("numpy", "numpy", "math + vertex packing"),
    ("PIL", "Pillow", "picture decoding"),
)


def check_python() -> None:
    say("interpreter")
    if sys.version_info < (3, 11):
        bad(f"Python {platform.python_version()} < 3.11", "install Python 3.11+")
    else:
        ok(f"Python {platform.python_version()} ({sys.executable})")


def check_packages() -> None:
    say("python packages")
    for module, dist, why in PACKAGES:
        try:
            importlib.import_module(module)
            version = importlib.metadata.version(dist)
            ok(f"{dist} {version}  ({why})")
        except Exception as exc:  # noqa: BLE001
            bad(f"{dist} missing: {exc}", "pip install -r requirements.txt")


def check_native() -> None:
    say("native GPU layer")
    if sys.platform == "darwin":
        ok("Metal ships with macOS (no driver to install)")
    elif sys.platform.startswith("linux"):
        if ctypes.util.find_library("vulkan"):
            ok("libvulkan (Vulkan loader) found")
        else:
            bad(
                "no libvulkan: wgpu will fall back to its GL backend",
                "Debian/Ubuntu: sudo apt install libvulkan1 mesa-vulkan-drivers  |  "
                "Fedora: sudo dnf install vulkan-loader mesa-vulkan-drivers  |  "
                "Arch: sudo pacman -S vulkan-icd-loader vulkan-radeon/vulkan-intel/nvidia-utils",
            )
    elif sys.platform == "win32":
        ok("DX12 ships with Windows 10+")


def check_adapters(backend: str) -> None:
    say("wgpu adapters")
    try:
        import gpu3d
        import wgpu

        order = gpu3d.backend_preference(backend)
        ok(f"backend preference on {sys.platform}: {' > '.join(order)}")
        for adapter in wgpu.gpu.enumerate_adapters_sync():
            info = adapter.info
            print(f"      - {info.get('device')} ({info.get('adapter_type')}) via {info.get('backend_type')}")
        chosen = gpu3d.pick_adapter(order)
        ok(f"selected: {chosen.summary}")
    except Exception as exc:  # noqa: BLE001
        bad(f"no usable adapter: {exc}")


def check_window(backend: str) -> None:
    say("live window + surface (1 s)")
    try:
        import time

        import glfw
        from gpu3d import GPU
        from window3d import close_window, create_window

        window = create_window("check_env — wgpu", 480, 320)
        gpu = GPU(window, backend=backend)
        gpu.clear_color = (1.0, 0.5, 0.0, 1.0)
        frames = 0
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and not glfw.window_should_close(window):
            glfw.poll_events()
            gpu.begin_frame()
            frames += gpu.end_frame()
        gpu.close()
        close_window(window)
        ok(f"presented {frames} frames via {gpu.backend}")
    except Exception as exc:  # noqa: BLE001
        bad(f"window probe failed: {exc}", "run from a desktop session (DISPLAY / WAYLAND_DISPLAY)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe the GLFW + wgpu stack.")
    parser.add_argument("--window", action="store_true", help="open a window and present frames")
    parser.add_argument("--backend", default="auto", help="auto | metal | vulkan | dx12 | gl")
    args = parser.parse_args()

    check_python()
    check_packages()
    check_native()
    check_adapters(args.backend)
    if args.window:
        check_window(args.backend)

    print()
    if failures:
        print(f"{RED}{failures} problem(s) found{RST}")
        return 1
    print(f"{GREEN}all good: ./.venv/bin/python main.py{RST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
