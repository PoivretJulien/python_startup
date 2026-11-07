#!/usr/bin/env bash
# Linux bootstrap of the Python GLFW + wgpu (Vulkan) toolchain.
#
#   ./setup-linux.sh
#   PYTHON=/usr/bin/python3.12 ./setup-linux.sh
#
# wgpu picks Vulkan automatically on Linux (falls back to its GL/GLES backend
# when no Vulkan driver is present). Only the Vulkan loader + a Mesa/vendor ICD
# are needed system-wide; everything Python-side is a prebuilt wheel.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"

say()  { printf '\033[38;5;208m▸\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

[[ "$(uname -s)" == "Linux" ]] || die "this script is for Linux (use ./setup-macos.sh on macOS)"

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
[[ -x "$VENV/bin/python" ]] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet -r requirements.txt
ok "packages installed"

say "probing wgpu (Vulkan adapter)"
"$VENV/bin/python" check_env.py

cat <<'DONE'

Ready:
  ./.venv/bin/python main.py                  # Vulkan, auto-selected
  ./.venv/bin/python main.py --backend gl     # force wgpu's GL fallback
  ./.venv/bin/python check_env.py --window    # surface + present test
DONE
