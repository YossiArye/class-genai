#!/usr/bin/env bash
# Installs pip-autosave into the venv. Run once after creating the venv:
#   python3 -m venv venv && bash scripts/setup-pip-autosave.sh
# Re-run it if you recreate the venv.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"

[ -x "$PY" ] || { echo "No venv found. Run: python3 -m venv venv"; exit 1; }
SP="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

cp "$ROOT/scripts/pip_autosave.py" "$SP/pip_autosave.py"
echo "import pip_autosave" > "$SP/pip_autosave.pth"

# Remove the old shell-function hook from activate, if present
sed -i '' '/# pip-autosave$/d' "$ROOT/venv/bin/activate" 2>/dev/null || true

echo "✅ pip-autosave installed in $SP"
