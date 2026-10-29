#!/usr/bin/env bash
# Linux bootstrap of the Python GLFW + OpenGL (core profile) toolchain.
#
#   ./setup-linux.sh            # .venv + modern tutorial stack + verify
#   ./setup-linux.sh --legacy   # also install pyrr, so the original 2016 scripts run
#   PYTHON=python3.12 ./setup-linux.sh
#
# Distro-agnostic (dnf / apt / pacman / zypper / apk). Never calls sudo: if a
# system library is missing it prints the exact command to run for *your* distro.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"
LEGACY=0
[[ "${1:-}" == "--legacy" ]] && LEGACY=1

say()  { printf '\033[38;5;208m▸\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# Cross-platform lib probe: ctypes works without ldconfig (absent on some images).
# Takes the *link name* ("GL", "X11", ...) not the soname ("libGL.so.1"):
# ctypes.util.find_library() resolves the platform-correct file itself.
have_lib() {
  "$PY" -c 'import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library(sys.argv[1]) else 1)' "$1" 2>/dev/null
}

# ------------------------------------------------------------ 0. distro mapping
detect_pm() {
  for pm in dnf apt-get pacman zypper apk; do
    command -v "$pm" >/dev/null 2>&1 && { printf '%s' "$pm"; return; }
  done
  printf 'unknown'
}
PM="$(detect_pm)"
pkg_hint() { # $1 = logical lib
  case "$PM:$1" in
    dnf:gl)          echo "sudo dnf install mesa-libGL libglvnd-glx" ;;
    dnf:x11)         echo "sudo dnf install libX11 libXrandr libXinerama libXcursor libXi" ;;
    dnf:wayland)     echo "sudo dnf install libwayland-client libwayland-cursor libxkbcommon" ;;
    apt-get:gl)      echo "sudo apt install libgl1 libglx-mesa0 libglvnd0" ;;
    apt-get:x11)     echo "sudo apt install libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6" ;;
    apt-get:wayland) echo "sudo apt install libwayland-client0 libwayland-cursor0 libxkbcommon0" ;;
    pacman:gl)       echo "sudo pacman -S --needed mesa libglvnd" ;;
    pacman:x11)      echo "sudo pacman -S --needed libx11 libxrandr libxinerama libxcursor libxi" ;;
    pacman:wayland)  echo "sudo pacman -S --needed wayland libxkbcommon" ;;
    zypper:gl)       echo "sudo zypper install Mesa-libGL1 libglvnd" ;;
    zypper:x11)      echo "sudo zypper install libX11-6 libXrandr2 libXinerama1 libXcursor1 libXi6" ;;
    zypper:wayland)  echo "sudo zypper install libwayland-client0 libxkbcommon0" ;;
    apk:gl)          echo "sudo apk add mesa-gl mesa-egl" ;;
    apk:x11)         echo "sudo apk add libx11 libxrandr libxinerama libxcursor libxi" ;;
    apk:wayland)     echo "sudo apk add wayland libxkbcommon" ;;
    *)               echo "install the packages providing $1 (unknown package manager '$PM')" ;;
  esac
}

# ---------------------------------------------------------------- 1. interpreter
say "checking interpreter"
[[ -x "$(command -v "$PY" || true)" ]] || die "$PY not found (install python3 >= 3.11 or set PYTHON=...)"
"$PY" - <<'PYCHECK'
import sys
if sys.version_info < (3, 11):
    sys.exit(f"Python >= 3.11 required, found {sys.version.split()[0]}")
print(f"    Python {sys.version.split()[0]}  ({sys.executable})")
PYCHECK

# ---------------------------------------------------- 2. native libs (check only)
say "checking native OpenGL / windowing libraries"
have_lib GL && ok "libGL.so.1 (Mesa / vendor GL driver)" \
  || warn "libGL.so.1 missing  ->  $(pkg_hint gl)"
if have_lib X11; then
  ok "libX11.so.6 (X11 backend for GLFW)"
  HAVE_X11=1
else
  warn "libX11.so.6 missing  ->  $(pkg_hint x11)"
  HAVE_X11=0
fi
if have_lib wayland-client; then
  ok "libwayland-client.so.0 (Wayland backend, optional)"
  HAVE_WL=1
else
  warn "no wayland client lib (fine on X11)  ->  $(pkg_hint wayland)"
  HAVE_WL=0
fi
have_lib xkbcommon && ok "libxkbcommon.so.0 (keyboard mapping)" \
  || warn "libxkbcommon.so.0 missing (only needed on Wayland)"

if [[ $HAVE_X11 -eq 0 && $HAVE_WL -eq 0 ]]; then
  die "GLFW needs at least one of X11 or Wayland -- install one of the sets above and re-run"
fi

if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  warn "no DISPLAY / WAYLAND_DISPLAY set: the probe below needs a session (or Xvfb + xvfb-run)"
fi

# ------------------------------------------------------------------- 3. venv
create_venv() {
  if command -v uv >/dev/null 2>&1; then
    say "creating virtualenv with uv"
    uv venv --python "$PY" "$VENV" >/dev/null
  else
    say "creating virtualenv with stdlib venv (uv not found)"
    "$PY" -m venv "$VENV"
  fi
}

if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c 'import sys' 2>/dev/null; then
  say "reusing existing virtualenv ($VENV)"
else
  [[ -e "$VENV" ]] && warn "$VENV is broken, recreating it" && rm -rf "$VENV"
  create_venv
fi
ok "venv -> $VENV/bin/python"

VPY="$VENV/bin/python"
if command -v uv >/dev/null 2>&1; then
  say "installing requirements.txt"
  uv pip install --quiet --python "$VPY" -r requirements.txt
  if [[ $LEGACY -eq 1 ]]; then
    say "installing pyrr for the legacy scripts"
    uv pip install --quiet --python "$VPY" -r requirements-legacy.txt
  fi
else
  say "installing requirements.txt"
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -r requirements.txt
  if [[ $LEGACY -eq 1 ]]; then
    say "installing pyrr for the legacy scripts"
    "$VPY" -m pip install --quiet -r requirements-legacy.txt
  fi
fi
ok "packages installed"

# ------------------------------------------------------------------- 4. verify
say "probing the live OpenGL context"
"$VPY" check_env.py

cat <<'DONE'

Ready. Activate and run things with:
  source .venv/bin/activate
  python check_env.py --window
DONE