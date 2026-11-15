#!/usr/bin/env bash
# Hooks pip-autosave into venv/bin/activate. Run once after creating the venv:
#   python3 -m venv venv && bash scripts/setup-pip-autosave.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTIVATE="$ROOT/venv/bin/activate"
LINE='. "$(dirname "$VIRTUAL_ENV")/scripts/pip-autosave.sh"  # pip-autosave'

[ -f "$ACTIVATE" ] || { echo "No venv found. Run: python3 -m venv venv"; exit 1; }
if grep -q '# pip-autosave' "$ACTIVATE"; then
  echo "pip-autosave already installed."
else
  printf '\n%s\n' "$LINE" >> "$ACTIVATE"
  echo "✅ pip-autosave installed. Run: source venv/bin/activate"
fi
