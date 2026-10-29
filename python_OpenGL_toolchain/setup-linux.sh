#!/usr/bin/env bash
# Linux bootstrap of the Python GLFW + OpenGL (core profile) toolchain.
#
#   ./setup-linux.sh                 # .venv + modern tutorial stack + verify
#   ./setup-linux.sh --legacy        # also install pyrr, so the 2016 scripts run
#   ./setup-linux.sh --system-glfw   # also install/require the distro libglfw.so.3
#   ./setup-linux.sh --bundled-glfw  # force the GLFW that ships inside the wheel
#   PYTHON=python3.12 ./setup-linux.sh
#
# Distro-agnostic (dnf / apt / pacman / zypper / apk). Never calls sudo on its own:
# when a system library is missing it prints the exact command for *your* distro,
# and only runs it after you answer "y" (--system-glfw).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"
LEGACY=0
GLFW_MODE="auto"   # auto | system | bundled
PIN_GLFW=""        # PYGLFW_LIBRARY value exported for the verification step

usage() {
  cat <<EOF
usage: ./setup-linux.sh [--legacy] [--system-glfw | --bundled-glfw]

  --legacy         also install pyrr (original 2016 scripts)
  --system-glfw    require the distro's libglfw.so.3 (installs it on request) and
                   pin PYGLFW_LIBRARY to it instead of the wheel's bundled copy
  --bundled-glfw   pin PYGLFW_LIBRARY to the copy bundled in the glfw wheel
                   (immune to a system libglfw silently taking precedence)

env: PYTHON=/path/to/python3 (>= 3.11)   PYGLFW_LIBRARY_VARIANT=x11|wayland
EOF
}

for arg in "$@"; do
  case "$arg" in
    --legacy)        LEGACY=1 ;;
    --system-glfw)   GLFW_MODE="system" ;;
    --bundled-glfw)  GLFW_MODE="bundled" ;;
    -h|--help)       usage; exit 0 ;;
    *)               printf 'unknown option: %s\n\n' "$arg" >&2; usage; exit 2 ;;
  esac
done

say()  { printf '\033[38;5;208m▸\033[0m %s\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '  \033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

# Cross-platform lib probe: ctypes works without ldconfig (absent on some images).
# Takes the *link name* ("GL", "X11", "glfw"), not the soname ("libGL.so.1").
have_lib() {
  "$PY" -c 'import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library(sys.argv[1]) else 1)' "$1" 2>/dev/null
}
lib_path() {
  "$PY" -c 'import ctypes.util,sys; print(ctypes.util.find_library(sys.argv[1]) or "")' "$1" 2>/dev/null
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
    dnf:glfw)        echo "sudo dnf install glfw glfw-devel" ;;
    apt-get:gl)      echo "sudo apt install libgl1 libglx-mesa0 libglvnd0" ;;
    apt-get:x11)     echo "sudo apt install libx11-6 libxrandr2 libxinerama1 libxcursor1 libxi6" ;;
    apt-get:wayland) echo "sudo apt install libwayland-client0 libwayland-cursor0 libxkbcommon0" ;;
    apt-get:glfw)    echo "sudo apt install libglfw3 libglfw3-dev" ;;
    pacman:gl)       echo "sudo pacman -S --needed mesa libglvnd" ;;
    pacman:x11)      echo "sudo pacman -S --needed libx11 libxrandr libxinerama libxcursor libxi" ;;
    pacman:wayland)  echo "sudo pacman -S --needed wayland libxkbcommon" ;;
    pacman:glfw)     echo "sudo pacman -S --needed glfw" ;;
    zypper:gl)       echo "sudo zypper install Mesa-libGL1 libglvnd" ;;
    zypper:x11)      echo "sudo zypper install libX11-6 libXrandr2 libXinerama1 libXcursor1 libXi6" ;;
    zypper:wayland)  echo "sudo zypper install libwayland-client0 libxkbcommon0" ;;
    zypper:glfw)     echo "sudo zypper install libglfw3 glfw-devel" ;;
    apk:gl)          echo "sudo apk add mesa-gl mesa-egl" ;;
    apk:x11)         echo "sudo apk add libx11 libxrandr libxinerama libxcursor libxi" ;;
    apk:wayland)     echo "sudo apk add wayland libxkbcommon" ;;
    apk:glfw)        echo "sudo apk add glfw glfw-dev" ;;
    *)               echo "install the packages providing $1 (unknown package manager '$PM')" ;;
  esac
}

# ---------------------------------------------------------------- 1. interpreter
say "checking interpreter"
command -v "$PY" >/dev/null 2>&1 || die "$PY not found (install python3 >= 3.11 or set PYTHON=...)"
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

# ------------------------------------------- 2b. native GLFW (libglfw.so.3)
# The glfw wheel ships its own GLFW (x11/ + wayland/, both 3.4.0), so a system
# libglfw is OPTIONAL. Beware: PyGLFW scans /usr/lib*, /usr/local/lib ... and
# picks the highest version, so a distro libglfw silently wins over the bundled
# one (and a distro build may lack the Wayland backend). Pin with --bundled-glfw.
say "checking native GLFW library (optional: the wheel bundles 3.4.0)"
SYS_GLFW="$(lib_path glfw)"
if [[ -n "$SYS_GLFW" ]]; then
  ok "system libglfw  [$SYS_GLFW]  (will be preferred if its version >= the bundled one)"
else
  warn "no system libglfw.so.3  ->  $(pkg_hint glfw)"
fi

if [[ $GLFW_MODE == "system" ]]; then
  if [[ -z "$SYS_GLFW" ]]; then
    CMD="$(pkg_hint glfw)"
    printf '  \033[33m!\033[0m --system-glfw requested but the distro package is missing\n'
    printf '    run:  %s\n' "$CMD"
    if [[ -t 0 ]]; then
      read -r -p "    run it now? [y/N] " reply
      if [[ "$reply" =~ ^[Yy]$ ]]; then
        if eval "$CMD"; then
          ok "system GLFW installed"
        else
          warn "install failed; falling back to the wheel's bundled GLFW"
        fi
        hash -r
        SYS_GLFW="$(lib_path glfw)"
      fi
    else
      warn "non-interactive shell: not installing (re-run from a terminal to be prompted)"
    fi
    [[ -n "$SYS_GLFW" ]] || die "system GLFW still unavailable; drop --system-glfw or install it with: $CMD"
  fi
  PIN_GLFW="$( "$PY" -c 'import ctypes.util,sys,glob,os; n=ctypes.util.find_library("glfw"); print(sorted(glob.glob(f"/usr/lib*/{n}*"))[0] if n and glob.glob(f"/usr/lib*/{n}*") else (n or ""))' )"
  ok "using system GLFW: $PIN_GLFW"
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
  say "installing requirements.txt (includes the glfw wheel)"
  uv pip install --quiet --python "$VPY" -r requirements.txt
  if [[ $LEGACY -eq 1 ]]; then
    say "installing pyrr for the legacy scripts"
    uv pip install --quiet --python "$VPY" -r requirements-legacy.txt
  fi
else
  say "installing requirements.txt (includes the glfw wheel)"
  "$VPY" -m pip install --quiet --upgrade pip
  "$VPY" -m pip install --quiet -r requirements.txt
  if [[ $LEGACY -eq 1 ]]; then
    say "installing pyrr for the legacy scripts"
    "$VPY" -m pip install --quiet -r requirements-legacy.txt
  fi
fi
ok "packages installed"

# ------------------------------------------- 3b. pin the GLFW the venv will use
glfw_bundled() { # path of the copy inside the wheel, honouring the backend variant
  "$1" - <<'PY'
import glob, os, glfw
root = os.path.dirname(glfw.__file__)
variant = os.environ.get("PYGLFW_LIBRARY_VARIANT") or ("wayland" if os.environ.get("XDG_SESSION_TYPE") == "wayland" else "x11")
fname = "libglfw.so"
for cand in (os.path.join(root, variant, fname),
             os.path.join(root, fname),
             *sorted(glob.glob(os.path.join(root, "**", fname), recursive=True))):
    if os.path.exists(cand):
        print(cand)
        break
PY
}

if [[ $GLFW_MODE == "bundled" ]]; then
  BUNDLED="$(glfw_bundled "$VPY")"
  [[ -n "$BUNDLED" ]] || die "the glfw wheel contains no bundled libglfw.so -- reinstall it: uv pip install --force-reinstall glfw"
  PIN_GLFW="$BUNDLED"
  ok "using the wheel's bundled GLFW: $PIN_GLFW"
fi

[[ -n "$PIN_GLFW" ]] && export PYGLFW_LIBRARY="$PIN_GLFW"

say "verifying the GLFW native library loaded by the venv"
"$VPY" - <<'PY' || die "GLFW native library not loaded -- fix: uv pip install --force-reinstall glfw (or ./setup-linux.sh --system-glfw)"
import os, sys
from glfw import library
import glfw
lib = getattr(library.glfw, "_name", None)
if lib is None:
    print("    GLFW native library: NOT LOADED")
    raise SystemExit(1)
try:
    major, minor, rev = glfw.get_version()
except Exception:
    major = minor = rev = 0
where = "system" if not lib.startswith(sys.prefix) else "bundled (wheel)"
pin = os.environ.get("PYGLFW_LIBRARY")
print(f"    GLFW {major}.{minor}.{rev}  [{where}]  {lib}")
if pin:
    print(f"    pinned by PYGLFW_LIBRARY={pin}")
elif where == "system":
    print("    note: a system libglfw shadows the wheel's copy; use --bundled-glfw to pin it")
PY
ok "GLFW ok"

# ------------------------------------------------------------------- 4. verify
say "probing the live OpenGL context"
"$VPY" check_env.py

cat <<'DONE'

Ready. Activate and run things with:
  source .venv/bin/activate
  python check_env.py --window
DONE

if [[ -n "$PIN_GLFW" ]]; then
  cat <<DONE

This run used PYGLFW_LIBRARY=$PIN_GLFW
To keep it for interactive shells:
  echo 'export PYGLFW_LIBRARY=$PIN_GLFW' >> ~/.bashrc
DONE
fi