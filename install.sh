#!/usr/bin/env bash
# Install the deep-humanizer skill into one or more agent skill folders.
#   ./install.sh                 # Claude Code (~/.claude/skills) + shared agents folder (~/.agents/skills)
#   ./install.sh claude          # Claude Code only
#   ./install.sh agents          # ~/.agents/skills (Codex, Cursor, Gemini CLI, GitHub Copilot and others)
#   ./install.sh copilot|cursor|gemini|codex   # tool-specific user folders
#   ./install.sh --link ...      # symlink instead of copy (edits in this repo take effect immediately)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
case "$HERE" in /*) ;; *) HERE="$PWD/${HERE#./}" ;; esac
SRC="$HERE/skills/deep-humanizer"
MODE=copy
TARGETS=()
for a in "$@"; do
  case "$a" in
    --link) MODE=link ;;
    claude) TARGETS+=("$HOME/.claude/skills") ;;
    agents) TARGETS+=("$HOME/.agents/skills") ;;
    copilot) TARGETS+=("$HOME/.copilot/skills") ;;
    cursor) TARGETS+=("$HOME/.cursor/skills") ;;
    gemini) TARGETS+=("$HOME/.gemini/skills") ;;
    codex) TARGETS+=("$HOME/.codex/skills") ;;
    *) echo "unknown option: $a" >&2; exit 1 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=("$HOME/.claude/skills" "$HOME/.agents/skills")
for t in "${TARGETS[@]}"; do
  mkdir -p "$t"
  dest="$t/deep-humanizer"
  if [ -e "$dest" ] || [ -L "$dest" ]; then
    echo "skip: $dest already exists (remove it first to reinstall)"; continue
  fi
  if [ "$MODE" = link ]; then ln -s "$SRC" "$dest"; else cp -R "$SRC" "$dest"; fi
  echo "installed ($MODE): $dest"
done
echo "Next: pip install -r \"$SRC/scripts/requirements.txt\" && python -m spacy download en_core_web_sm"
