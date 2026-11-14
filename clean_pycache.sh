#!/usr/bin/env bash
#
# clean_pycache.sh: remove every __pycache__ folder below the folder
# that contains this script.
#
# Usage:
#   ./clean_pycache.sh            list what will be removed, then ask
#   ./clean_pycache.sh --dry-run  only list, remove nothing
#   ./clean_pycache.sh --yes      remove without asking
#
# Safety:
#   - Works only below the script's own folder, wherever you run it from.
#   - Refuses to run if that folder is / or your home folder.
#   - Never follows symbolic links, and skips links named __pycache__.
#   - Skips .git, virtual environments (.venv, venv, env) and node_modules.
#   - Removes a __pycache__ folder only if it holds nothing but .pyc files.
#     Anything else inside means it is not a plain Python cache: it is skipped.

set -euo pipefail

DRY_RUN=0
ASSUME_YES=0
for arg in "$@"; do
    case "$arg" in
        -n|--dry-run) DRY_RUN=1 ;;
        -y|--yes)     ASSUME_YES=1 ;;
        -h|--help)    awk 'NR > 2 && !/^#/ { exit } NR > 2 { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# The folder that contains this script, as a real absolute path.
ROOT="$(cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

case "$ROOT" in
    ""|"/"|"$HOME"|"$(cd -P -- "$HOME" && pwd -P)")
        echo "Refusing to run from '$ROOT': too wide." >&2
        exit 1 ;;
esac

echo "Looking for __pycache__ folders below: $ROOT"

TO_REMOVE=()
SKIPPED=()

# -P: never follow symbolic links.
# -prune on the skipped folders: do not even look inside them.
# -type d: only real folders, so a link called __pycache__ is ignored.
while IFS= read -r -d '' dir; do
    # Anything in it that is not a .pyc file?
    other="$(find "$dir" -mindepth 1 ! -type d ! -name '*.pyc' -print -quit)"
    if [ -n "$other" ]; then
        SKIPPED+=("$dir")
    else
        TO_REMOVE+=("$dir")
    fi
done < <(find -P "$ROOT" \
            \( -name .git -o -name .venv -o -name venv -o -name env -o -name node_modules \) -prune \
            -o -type d -name __pycache__ -prune -print0)

if [ "${#SKIPPED[@]}" -gt 0 ]; then
    echo
    echo "Skipped (they contain files other than .pyc):"
    for d in "${SKIPPED[@]}"; do echo "  ${d#"$ROOT"/}"; done
fi

if [ "${#TO_REMOVE[@]}" -eq 0 ]; then
    echo
    echo "Nothing to remove."
    exit 0
fi

echo
echo "To remove (${#TO_REMOVE[@]}):"
for d in "${TO_REMOVE[@]}"; do echo "  ${d#"$ROOT"/}"; done

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "Dry run: nothing removed."
    exit 0
fi

if [ "$ASSUME_YES" -ne 1 ]; then
    echo
    read -r -p "Remove these ${#TO_REMOVE[@]} folder(s)? [y/N] " answer
    case "$answer" in
        y|Y|yes|YES) ;;
        *) echo "Cancelled: nothing removed."; exit 0 ;;
    esac
fi

removed=0
for d in "${TO_REMOVE[@]}"; do
    # Check again just before removing: still a real folder, still below ROOT.
    if [ -d "$d" ] && [ ! -L "$d" ] && [ "${d#"$ROOT"/}" != "$d" ]; then
        rm -rf -- "$d"
        removed=$((removed + 1))
    fi
done

echo "Removed $removed folder(s)."
