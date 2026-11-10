#!/usr/bin/env bash
# Linux bootstrap of the whole project: GLFW + wgpu (Vulkan) + OpenCV.
#
#   ./toolchain/setup-linux.sh
#   ./toolchain/setup-linux.sh --no-opencv     # skip the optional OpenCV step
#   PYTHON=/usr/bin/python3.12 ./toolchain/setup-linux.sh
#
# wgpu picks Vulkan automatically on Linux (falls back to its GL/GLES backend
# when no Vulkan driver is present). Only the Vulkan loader + a Mesa/vendor ICD
# are needed system-wide; everything Python-side is a prebuilt wheel.
#
# It can be started from any folder: it always works in the project root (the
# parent of toolchain/), which is where .venv is created.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python3}"
VENV=".venv"
REQ="toolchain/requirements.txt"
REQ_CV="toolchain/requirements-opencv.txt"
OPENCV=1

for arg in "$@"; do
  case "$arg" in
    --no-opencv) OPENCV=0 ;;
    -h|--help)   echo "usage: ./toolchain/setup-linux.sh [--no-opencv]   (env: PYTHON=/path/to/python3 >= 3.11)"; exit 0 ;;
    *)           echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\033[38;5;208m▸\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux (use ./toolchain/setup-macos.sh on macOS)"

say "checking interpreter"
command -v "$PY" >/dev/null 2>&1 || die "$PY not found (install python3 >= 3.11)"
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "Python >= 3.11 required ($("$PY" -V))"
ok "$("$PY" -V)"

say "checking the Vulkan driver stack"
if "$PY" -c 'import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library("vulkan") else 1)'; then
  ok "libvulkan found"
else
  warn "no Vulkan loader: wgpu will use its GL fallback. Install one of:"
  echo "      Debian/Ubuntu: sudo apt install libvulkan1 mesa-vulkan-drivers"
  echo "      Fedora:        sudo dnf install vulkan-loader mesa-vulkan-drivers"
  echo "      Arch:          sudo pacman -S vulkan-icd-loader vulkan-radeon|vulkan-intel|nvidia-utils"
fi
if [[ -n "${WAYLAND_DISPLAY:-}" ]]; then ok "Wayland session"; elif [[ -n "${DISPLAY:-}" ]]; then ok "X11 session"; else warn "no DISPLAY / WAYLAND_DISPLAY: the window probe will fail"; fi

say "virtualenv + packages"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PY" -m venv "$VENV" || die "could not create $VENV (Debian/Ubuntu: sudo apt install python3-venv)"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r "$REQ"
ok "packages installed (glfw, wgpu, rendercanvas, numpy, Pillow)"

# OpenCV is optional, so a missing wheel (very new Python) is a warning only.
if [[ $OPENCV -eq 1 ]]; then
  say "OpenCV (optional: camera + video_link)"
  if "$VENV/bin/python" -m pip install --quiet -r "$REQ_CV"; then
    ok "OpenCV installed"
  else
    warn "OpenCV could not be installed (no wheel for this Python yet?): only network/video_link needs it"
  fi
fi

say "probing wgpu (Vulkan adapter)"
"$VENV/bin/python" toolchain/check_env.py

cat <<'DONE'

Ready. From the project root:
  .venv/bin/python toolchain/check_env.py --window     # surface + present test
  .venv/bin/python examples/01_hover_shapes/main.py    # hover two shapes
  .venv/bin/python examples/02_udp_drag/main.py        # two windows over UDP
  .venv/bin/python examples/03_udp_video/main.py       # video between two windows
  .venv/bin/python examples/04_linked_circles/main.py  # two circles and a dashed line
  .venv/bin/python examples/05_keyboard/main.py        # the keyboard, on screen
  .venv/bin/python examples/06_slider/main.py          # sliders that set values
  .venv/bin/python examples/07_gradient_gauge/main.py  # a gradient gauge and the arrow keys
  .venv/bin/python graphics/main.py                    # the full 3D scene (--backend gl forces the GL fallback)
  xdg-open guide.html                                  # how to use the API
DONE
