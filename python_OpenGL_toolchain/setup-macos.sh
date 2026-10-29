#!/usr/bin/env bash
# macOS bootstrap of the Python GLFW + OpenGL (core profile) toolchain.
#
#   ./setup-macos.sh            # .venv + modern tutorial stack + verify
#   ./setup-macos.sh --legacy   # also install pyrr, so the original 2016 scripts run
#   PYTHON=/opt/homebrew/bin/python3.12 ./setup-macos.sh
#
# Nothing needs sudo: OpenGL.framework, Cocoa and IOKit ship with macOS, and the
# glfw / PyOpenGL / numpy / Pillow wheels are all prebuilt for arm64 + x86_64.
# The only prerequisite is a Python >= 3.11 (system /usr/bin/python3 is too old).
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

[[ "$(uname -s)" == "Darwin" ]] || die "this script is for macOS (use ./setup-linux.sh or setup-windows.ps1)"

# ---------------------------------------------------------------- 1. interpreter
say "checking macOS + interpreter"
ok "macOS $(sw_vers -productVersion) ($(uname -m))"

# The system interpreter (/usr/bin/python3) is still 3.9 on current macOS: refuse
# it explicitly instead of failing on a cryptic syntax error later.
if [[ -x "$(command -v "$PY" || true)" ]]; then
  PY_PATH="$(command -v "$PY")"
else
  die "$PY not found.
       install one:  brew install python@3.12
                     or:  curl -LsSf https://astral.sh/uv/install.sh | sh && uv python install 3.12
       then re-run:  PYTHON=\"\$HOME/.local/bin/python3.12\" ./setup-macos.sh"
fi
[[ "$PY_PATH" == "/usr/bin/python3" || "$PY_PATH" == "/usr/bin/python3."* ]] && \
  die "refusing the macOS system python ($PY_PATH): Apple ships 3.9.
       install one:  brew install python@3.12    (or uv python install 3.12)
       then re-run:  PYTHON=/opt/homebrew/bin/python3.12 ./setup-macos.sh"

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

# Apple deprecated OpenGL but still ships it (all Apple GPUs reach GL 4.1 core).
[[ "$(uname -m)" == "arm64" ]] && ok "Apple silicon: native arm64 wheels (no Rosetta needed)"

if [[ -z "${SSH_TTY:-}" && -z "${SSH_CONNECTION:-}" ]]; then
  ok "local GUI session available for the window probe"
else
  warn "running over SSH: step 4 needs a logged-in Aqua session and will likely fail"
  warn "log in on the machine itself (screen sharing also counts) before running the probe"
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

macOS notes:
  * GLFW must create windows on the main thread; the tutorials already do.
  * OpenGL is deprecated by Apple (still functional): ignore -Wdeprecated noise.
  * Under Rosetta or an x86_64 shell, use an x86_64 Python + x86_64 wheels.
DONE