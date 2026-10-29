#!/usr/bin/env python3
"""Cross-platform probe for the GLFW + OpenGL (core profile) stack.

    python check_env.py             # hidden window: checks libs, packages, live GL context
    python check_env.py --window    # same, but shows a real window for ~2 s
    python check_env.py --quiet     # only failures + one summary line

Exit code 0 = the stack is usable; 1 = something fatal is missing.

Deliberately network-free, sudo-free and import-light: it works on Linux, macOS
and Windows with nothing but the Python standard library, then it creates an
actual 3.3-core context, compiles a GLSL 330 shader pair and links it. If that
passes, the tutorial scripts will run.
"""

from __future__ import annotations

import argparse
import ctypes.util
import importlib
import importlib.metadata
import os
import platform
import sys

MIN_PY = (3, 11)
MIN_GL = (3, 3)
SYSTEM = platform.system()  # 'Linux' | 'Darwin' | 'Windows'

_COLOR = False


def _enable_colors() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.name == "nt":
        # Turns on VT sequences in modern Windows consoles (no-op elsewhere).
        os.system("")
        return os.environ.get("WT_SESSION") is not None or bool(os.environ.get("TERM"))
    return sys.stdout.isatty()


_COLOR = _enable_colors()
C = {
    "reset": "\033[0m" if _COLOR else "",
    "grey": "\033[90m" if _COLOR else "",
    "orange": "\033[38;5;208m" if _COLOR else "",
    "green": "\033[32m" if _COLOR else "",
    "yellow": "\033[33m" if _COLOR else "",
    "red": "\033[31m" if _COLOR else "",
    "bold": "\033[1m" if _COLOR else "",
}

PROBLEMS: list[str] = []
HINTS: list[str] = []


def say(msg: str) -> None:
    print(f"{C['orange']}\u25b8{C['reset']} {msg}")


def ok(msg: str) -> None:
    print(f"  {C['green']}\u2713{C['reset']} {msg}")


def warn(msg: str) -> None:
    print(f"  {C['yellow']}!{C['reset']} {msg}")


def bad(msg: str, hint: str | None = None) -> None:
    print(f"  {C['red']}\u2717{C['reset']} {msg}")
    PROBLEMS.append(msg)
    if hint:
        HINTS.append(hint)


def detail(msg: str) -> None:
    print(f"    {C['grey']}{msg}{C['reset']}")


# --------------------------------------------------------------- 1. interpreter
def check_interpreter() -> None:
    say("interpreter")
    v = platform.python_version()
    exe = sys.executable
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if sys.version_info[:2] < MIN_PY:
        bad(f"Python {v} is too old (need >= {MIN_PY[0]}.{MIN_PY[1]})", "install a newer Python or set PYTHON=/path/to/python3")
    else:
        ok(f"Python {v}")
    detail(f"exe    : {exe}")
    detail(f"venv   : {'yes' if in_venv else 'no (python -m venv .venv && activate)'}")
    if os.name == "nt" and not in_venv:
        warn(r"expected .venv\Scripts\python.exe")


# ------------------------------------------------------------- 2. native layers
_LINUX_LIBS = [
    ("GL", "libGL.so.1", "OpenGL driver (Mesa or vendor)", True,
     "config/GL: dnf install mesa-libGL | apt install libgl1 libglx-mesa0 | pacman -S mesa"),
    ("X11", "libX11.so.6", "X11 backend for GLFW", False,
     "config/X11: dnf install libX11 | apt install libx11-6 | pacman -S libx11"),
    ("wayland-client", "libwayland-client.so.0", "Wayland backend (optional)", False,
     "config/wayland: dnf install libwayland-client | apt install libwayland-client0 | pacman -S wayland"),
    ("xkbcommon", "libxkbcommon.so.0", "keyboard mapping (Wayland)", False,
     "config/xkbcommon: dnf install libxkbcommon | apt install libxkbcommon0 | pacman -S libxkbcommon"),
]
_MAC_FRAMEWORKS = [
    ("/System/Library/Frameworks/OpenGL.framework", "OpenGL.framework (deprecated but shipped)", True),
    ("/System/Library/Frameworks/Cocoa.framework", "Cocoa.framework (window/session)", True),
    ("/System/Library/Frameworks/IOKit.framework", "IOKit.framework (input)", False),
]
_WIN_DLLS = [
    ("opengl32.dll", "opengl32.dll (system OpenGL 1.1 loader)", True),
    ("glu32.dll", "glu32.dll (legacy helpers, optional)", False),
]


def check_native() -> None:
    say("native OpenGL / windowing layer")
    if SYSTEM == "Linux":
        found_x11 = found_wayland = False
        for name, filename, why, required, hint in _LINUX_LIBS:
            got = ctypes.util.find_library(name)
            if got:
                ok(f"{filename}  [{got}]  {why}")
                found_x11 |= name == "X11"
                found_wayland |= name == "wayland-client"
            elif required:
                bad(f"{filename} missing  ({why})", hint)
            else:
                warn(f"{filename} missing  ({why})")
        if not (found_x11 or found_wayland):
            bad("GLFW needs at least one of X11 or Wayland",
                "install libX11 (X11) or libwayland-client + libxkbcommon (Wayland)")
        check_session()
    elif SYSTEM == "Darwin":
        for path, why, required in _MAC_FRAMEWORKS:
            if os.path.isdir(path):
                ok(why)
            elif required:
                bad(f"{why} missing", "unexpected: OpenGL/Cocoa ship with macOS; reinstall the command line tools")
            else:
                warn(f"{why} missing")
        ok(f"darwin {platform.machine()} / macOS {platform.mac_ver()[0]}")
        detail("no packages to install: OpenGL, Cocoa and IOKit are part of the OS")
        detail("a logged-in GUI (Aqua) session is required to open a window")
    elif SYSTEM == "Windows":
        sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
        for dll, why, required in _WIN_DLLS:
            if os.path.exists(os.path.join(sys32, dll)):
                ok(why)
            elif required:
                bad(f"{dll} missing", "system OpenGL is broken; run 'sfc /scannow'")
            else:
                warn(f"{why} missing")
        detail(f"win32 {platform.machine()} / {platform.release()} ({platform.version()})")
        detail("no packages to install: opengl32.dll ships with Windows")
        if os.environ.get("SESSIONNAME", "").upper().startswith("RDP"):
            warn("Remote Desktop session: GPU contexts are usually limited to GL 1.1; run locally")
    else:
        warn(f"unrecognised platform {SYSTEM!r}: skipping native checks")


def check_session() -> None:
    """Linux: a window needs a display server. Headless still gets lib/package checks."""
    has_x = bool(os.environ.get("DISPLAY"))
    has_wl = bool(os.environ.get("WAYLAND_DISPLAY"))
    if has_x or has_wl:
        ok(f"session: {os.environ.get('XDG_SESSION_TYPE') or 'unknown'}"
           f" (DISPLAY={os.environ.get('DISPLAY') or '-'}, WAYLAND_DISPLAY={os.environ.get('WAYLAND_DISPLAY') or '-'})")
    else:
        warn("no DISPLAY / WAYLAND_DISPLAY: window probe will be skipped")
        HINTS.append("headless: xvfb-run -a python check_env.py   (dnf/apt install xvfb)")


# ----------------------------------------------------------------- 3. packages
# (module, distribution, label, why) -- distribution names, not import names
_PKGS = [
    ("glfw", "glfw", "PyGLFW", "window, input, GL context"),
    ("OpenGL", "PyOpenGL", "PyOpenGL", "OpenGL bindings"),
    ("numpy", "numpy", "numpy", "matrix/vector math"),
    ("PIL", "Pillow", "Pillow", "PNG texture loading"),
]
_PKGS_OPT = [
    ("OpenGL_accelerate", "PyOpenGL-accelerate", "PyOpenGL-accelerate", "C fast path (optional)"),
    ("pyrr", "pyrr", "pyrr", "legacy 2016 scripts only (--legacy)"),
]


def _version(dist: str) -> str:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "?"


def check_packages() -> None:
    say("python packages")
    for module, dist, label, why in _PKGS:
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report anything, keep going
            bad(f"{label} not importable ({why}): {type(exc).__name__}: {exc}",
                "install: python -m pip install -r requirements.txt")
        else:
            ok(f"{label} {_version(dist)}  ({why})")
    for module, dist, label, why in _PKGS_OPT:
        try:
            importlib.import_module(module)
        except Exception:
            warn(f"{label} absent  ({why})")
        else:
            ok(f"{label} {_version(dist)}  ({why})")


# ------------------------------------------------------------ 4. live GL probe
VERT = """#version 330 core
layout(location = 0) in vec2 position;
void main() { gl_Position = vec4(position, 0.0, 1.0); }
"""
FRAG = """#version 330 core
out vec4 color;
void main() { color = vec4(1.0, 0.5, 0.0, 1.0); }
"""


def _gl_version(text: str) -> tuple[int, ...]:
    head = text.split("(")[0].strip().split()
    nums = head[0].split(".") if head else []
    out = []
    for part in nums[:2]:
        try:
            out.append(int(part))
        except ValueError:
            out.append(0)
    return tuple(out)


def check_gl(visible: bool) -> None:
    say("live OpenGL context")
    if SYSTEM == "Linux" and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        warn("skipped (no display server)")
        return
    try:
        import glfw
    except Exception:
        bad("cannot probe: glfw not importable")
        return

    errors: list[str] = []

    def on_error(code: int, desc: str) -> None:  # noqa: ARG001
        errors.append(desc)

    glfw.set_error_callback(on_error)
    if not glfw.init():
        bad(f"glfw.init() failed: {'; '.join(errors) or 'no detail'}",
            "Linux: install a display server / set DISPLAY; Windows: run locally (not as a service); macOS: log in to the GUI")
        return
    ok("glfw.init()")

    window = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, MIN_GL[0])
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, MIN_GL[1])
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)  # required on macOS
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
        window = glfw.create_window(640, 480, "check_env", None, None)
        if not window:
            bad(f"create_window failed: {'; '.join(errors) or 'no detail'}",
                "needs an OpenGL 3.3 core driver (software fallback: LIBGL_ALWAYS_SOFTWARE=1 on Mesa)")
            return
        glfw.make_context_current(window)

        import OpenGL.GL as GL

        version = GL.glGetString(GL.GL_VERSION)
        vendor = GL.glGetString(GL.GL_VENDOR)
        renderer = GL.glGetString(GL.GL_RENDERER)
        glsl = GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)
        decode = lambda b: b.decode("utf-8", "replace") if isinstance(b, bytes) else str(b)  # noqa: E731
        detail(f"vendor   : {decode(vendor)}")
        detail(f"renderer : {decode(renderer)}")
        detail(f"GL       : {decode(version)}")
        detail(f"GLSL     : {decode(glsl)}")

        got = _gl_version(decode(version))
        if got and got < MIN_GL:
            bad(f"OpenGL {'.'.join(map(str, got))} < required {'.'.join(map(str, MIN_GL))}",
                "install/update the GPU driver; Mesa software: LIBGL_ALWAYS_SOFTWARE=1")
        else:
            ok(f"OpenGL {'.'.join(map(str, got)) if got else '?'} core profile")

        # Core-profile sanity: a VAO must exist, and a GLSL 330 program must link.
        GL.glGenVertexArrays(1)
        ok("core profile: glGenVertexArrays available")

        def compile_shader(kind: int, source: str, label: str) -> int:
            shader = GL.glCreateShader(kind)
            GL.glShaderSource(shader, source)
            GL.glCompileShader(shader)
            if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
                raise RuntimeError(f"{label} compile error: {GL.glGetShaderInfoLog(shader)}")
            return shader

        vs = compile_shader(GL.GL_VERTEX_SHADER, VERT, "vertex")
        fs = compile_shader(GL.GL_FRAGMENT_SHADER, FRAG, "fragment")
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vs)
        GL.glAttachShader(program, fs)
        GL.glLinkProgram(program)
        if not GL.glGetProgramiv(program, GL.GL_LINK_STATUS):
            raise RuntimeError(f"link error: {GL.glGetProgramInfoLog(program)}")
        ok("GLSL #version 330 core: shader pair compiled + linked")

        if visible:
            GL.glClearColor(1.0, 0.5, 0.0, 1.0)
            frames = 0
            while not glfw.window_should_close(window) and frames < 120:
                GL.glClear(GL.GL_COLOR_BUFFER_BIT)
                glfw.swap_buffers(window)
                glfw.poll_events()
                frames += 1
            ok(f"presented {frames} frames + swap_buffers")
    except Exception as exc:  # noqa: BLE001
        bad(f"GL probe raised {type(exc).__name__}: {exc}",
            "check that the driver supports OpenGL 3.3 core (glxinfo -B / GPU driver update)")
    finally:
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


# --------------------------------------------------------------------- summary
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the GLFW + OpenGL tutorial stack.")
    parser.add_argument("--window", action="store_true", help="open a visible 640x480 window for ~2 s")
    parser.add_argument("--quiet", action="store_true", help="only print failures and the verdict")
    args = parser.parse_args(argv)

    global say, ok, warn, detail  # noqa: PLW0603
    if args.quiet:
        quiet = lambda *a, **k: None  # noqa: E731
        say = ok = warn = detail = quiet

    print(f"{C['bold']}check_env{C['reset']} {C['grey']}({SYSTEM}, {platform.machine()}){C['reset']}\n")
    check_interpreter()
    check_native()
    check_packages()
    check_gl(args.window)

    print()
    if PROBLEMS:
        print(f"{C['red']}\u2717 {len(PROBLEMS)} problem(s){C['reset']}")
        for hint in dict.fromkeys(HINTS):
            print(f"  {C['grey']}hint:{C['reset']} {hint}")
        return 1
    print(f"{C['green']}\u2713 stack OK{C['reset']} \u2014 "
          f"activate and run: {C['grey']}{os.path.join('.venv', 'bin', 'python') if os.name != 'nt' else r'.venv\Scripts\python.exe'}{C['reset']} <script>.py")
    for hint in dict.fromkeys(HINTS):
        print(f"  {C['grey']}note:{C['reset']} {hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())