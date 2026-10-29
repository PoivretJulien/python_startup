#!/usr/bin/env bash
# macOS bootstrap of the Python GLFW + OpenGL (core profile) toolchain.
#
#   ./setup-macos.sh                 # .venv + modern tutorial stack + verify
#   ./setup-macos.sh --legacy        # also install pyrr, so the 2016 scripts run
#   ./setup-macos.sh --system-glfw   # install/require Homebrew's libglfw.3.dylib
#   ./setup-macos.sh --bundled-glfw  # force the GLFW that ships inside the wheel
#   PYTHON=/opt/homebrew/bin/python3.12 ./setup-macos.sh
#
# Nothing here needs sudo by default: OpenGL.framework, Cocoa and IOKit ship with
# macOS, and the glfw / PyOpenGL / numpy / Pillow wheels are prebuilt for arm64
# and x86_64. The only prerequisite is a Python >= 3.11 (system /usr/bin/python3
# is 3.9 and is refused explicitly).
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
VENV=".venv"
LEGACY=0
GLFW_MODE="auto"   # auto | system | bundled
PIN_GLFW=""

usage() {
  cat <<EOF
usage: ./setup-macos.sh [--legacy] [--system-glfw | --bundled-glfw]

  --legacy         also install pyrr (original 2016 scripts)
  --system-glfw    require Homebrew/MacPorts libglfw.3.dylib (installs it on
                   request) and pin PYGLFW_LIBRARY to it
  --bundled-glfw   pin PYGLFW_LIBRARY to the copy bundled in the glfw wheel
                   (a Homebrew libglfw in /opt/homebrew/lib otherwise shadows it)

env: PYTHON=/path/to/python3 (>= 3.11)
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

[[ "$(uname -s)" == "Darwin" ]] || die "this script is for macOS (use ./setup-linux.sh or setup-windows.ps1)"

lib_path() { # ctypes-based probe: works without ldconfig (macOS has none)
  "$PY" -c 'import ctypes.util,sys; print(ctypes.util.find_library(sys.argv[1]) or "")' "$1" 2>/dev/null
}

# ---------------------------------------------------------------- 1. interpreter
say "checking macOS + interpreter"
ok "macOS $(sw_vers -productVersion) ($(uname -m))"

command -v "$PY" >/dev/null 2>&1 || die "$PY not found.
       install one:  brew install python@3.12
                     or:  curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12
       then re-run:  PYTHON=\"\$HOME/.local/bin/python3.12\" ./setup-macos.sh"

PY_PATH="$(command -v "$PY")"
case "$PY_PATH" in
  /usr/bin/python3|/usr/bin/python3.*)
    die "refusing the macOS system python ($PY_PATH): Apple ships 3.9.
       install one:  brew install python@3.12    (or uv python install 3.12)
       then re-run:  PYTHON=/opt/homebrew/bin/python3.12 ./setup-macos.sh" ;;
esac

"$PY" - <<'PYCHECK'
import platform, sys
if sys.version_info < (3, 11):
    sys.exit(f"Python >= 3.11 required, found {sys.version.split()[0]} at {sys.executable}")
print(f"    Python {platform.python_version()}  ({sys.executable})")
PYCHECK

# ------------------------------------------------------ 2. native libs (verify)
say "checking native OpenGL / windowing frameworks"
missing=0
for fw in OpenGL Cocoa IOKit; do
  if [[ -d "/System/Library/Frameworks/$fw.framework" ]]; then
    ok "$fw.framework"
  else
    warn "$fw.framework missing (unexpected; part of macOS)"
    missing=1
  fi
done
[[ $missing -eq 1 ]] && die "a macOS system framework is missing: reinstall the Xcode command line tools (xcode-select --install)"

if xcode-select -p >/dev/null 2>&1; then
  ok "command line tools: $(xcode-select -p)"
else
  warn "no Xcode command line tools: fine (all wheels are prebuilt), needed only to build from source"
fi

[[ "$(uname -m)" == "arm64" ]] && ok "Apple silicon: native arm64 wheels (no Rosetta needed)"

if [[ -z "${SSH_TTY:-}" && -z "${SSH_CONNECTION:-}" ]]; then
  ok "local GUI session available for the window probe"
else
  warn "running over SSH: step 4 needs a logged-in Aqua session and will likely fail"
fi

# ---------------------------------------------- 2b. native GLFW (libglfw.dylib)
# The glfw wheel ships its own libglfw.3.dylib, so a system GLFW is OPTIONAL.
# Beware: PyGLFW also scans /opt/homebrew/lib and /usr/local/lib, so a Homebrew
# libglfw of equal-or-newer version silently replaces the bundled one.
say "checking native GLFW library (optional: the wheel bundles 3.4.0)"
SYS_GLFW="$(lib_path glfw)"
for cand in /opt/homebrew/lib/libglfw*.dylib /usr/local/lib/libglfw*.dylib /opt/local/lib/libglfw*.dylib; do
  [[ -f "$cand" && -z "$SYS_GLFW" ]] && SYS_GLFW="$cand"
done
if [[ -n "$SYS_GLFW" ]]; then
  ok "system libglfw  [$SYS_GLFW]  (preferred by PyGLFW if its version >= the bundled one)"
else
  warn "no system libglfw  ->  brew install glfw   (optional)"
fi

if [[ $GLFW_MODE == "system" && -z "$SYS_GLFW" ]]; then
  if command -v brew >/dev/null 2>&1; then
    CMD="brew install glfw"
  elif command -v port >/dev/null 2>&1; then
    CMD="sudo port install glfw"
  else
    die "no Homebrew/MacPorts: install GLFW manually or drop --system-glfw
       (the wheel's bundled GLFW is what runs by default and it works fine)"
  fi
  printf '  \033[33m!\033[0m --system-glfw requested but no libglfw found\n    run:  %s\n' "$CMD"
  if [[ -t 0 ]]; then
    read -r -p "    run it now? [y/N] " reply
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      if eval "$CMD"; then
        ok "system GLFW installed"
      else
        warn "install failed; falling back to the wheel"
      fi
      SYS_GLFW="$(lib_path glfw)"
      for cand in /opt/homebrew/lib/libglfw*.dylib /usr/local/lib/libglfw*.dylib /opt/local/lib/libglfw*.dylib; do
        [[ -f "$cand" && -z "$SYS_GLFW" ]] && SYS_GLFW="$cand"
      done
    fi
  else
    warn "non-interactive shell: not installing"
  fi
  [[ -n "$SYS_GLFW" ]] || die "system GLFW still unavailable; drop --system-glfw or install it with: brew install glfw"
fi

if [[ $GLFW_MODE == "system" ]]; then
  PIN_GLFW=""
  for cand in /opt/homebrew/lib/libglfw.3*.dylib /usr/local/lib/libglfw.3*.dylib /opt/local/lib/libglfw.3*.dylib; do
    if [[ -f "$cand" ]]; then PIN_GLFW="$cand"; break; fi
  done
  [[ -n "$PIN_GLFW" ]] || PIN_GLFW="$SYS_GLFW"
  ok "using system GLFW: $PIN_GLFW"
fi

# ------------------------------------------------------------------ 3. venv
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

# ------------------------------------------ 3b. pin the GLFW the venv will use
if [[ $GLFW_MODE == "bundled" ]]; then
  BUNDLED="$("$VPY" - <<'PY'
import glob, os, glfw
root = os.path.dirname(glfw.__file__)
hits = [p for p in sorted(glob.glob(os.path.join(root, "**", "libglfw*.dylib"), recursive=True))]
print(hits[0] if hits else "")
PY
)"
  [[ -n "$BUNDLED" ]] || die "the glfw wheel contains no bundled libglfw dylib -- reinstall it: uv pip install --force-reinstall glfw"
  PIN_GLFW="$BUNDLED"
  ok "using the wheel's bundled GLFW: $PIN_GLFW"
fi

[[ -n "$PIN_GLFW" ]] && export PYGLFW_LIBRARY="$PIN_GLFW"

say "verifying the GLFW native library loaded by the venv"
"$VPY" - <<'PY' || die "GLFW native library not loaded -- fix: uv pip install --force-reinstall glfw (or ./setup-macos.sh --system-glfw)"
import os, sys
try:
    from glfw import library
    import glfw
except ImportError as exc:
    print(f"    GLFW native library: NOT LOADED ({exc})")
    raise SystemExit(1)
lib = getattr(library.glfw, "_name", None)
if lib is None:
    print("    GLFW native library: NOT LOADED")
    raise SystemExit(1)
try:
    major, minor, rev = glfw.get_version()
except Exception:
    major = minor = rev = 0
where = "bundled (wheel)" if lib.startswith(sys.prefix) else "system"
print(f"    GLFW {major}.{minor}.{rev}  [{where}]  {lib}")
pin = os.environ.get("PYGLFW_LIBRARY")
if pin:
    print(f"    pinned by PYGLFW_LIBRARY={pin}")
elif where == "system":
    print("    note: a Homebrew libglfw shadows the wheel's copy; use --bundled-glfw to pin it")
PY
ok "GLFW ok"

# ------------------------------------------------------------------- 4. verify
say "probing the live OpenGL context"
"$VPY" check_env.py

cat <<'DONE'

Ready. Activate and run things with:
  source .venv/bin/activate
  python check_env.py --window

macOS notes:
  * GLFW must create windows on the main thread; the tutorials already do.
  * OpenGL is deprecated by Apple (still functional): ignore -Wdeprecated noise.
  * Under Rosetta or an x86_64 shell, use an x86_64 Python + x86_64 wheels.
DONE

if [[ -n "$PIN_GLFW" ]]; then
  cat <<DONE

This run used PYGLFW_LIBRARY=$PIN_GLFW
To keep it for interactive shells:
  echo 'export PYGLFW_LIBRARY=$PIN_GLFW' >> ~/.zshrc
DONE
fi