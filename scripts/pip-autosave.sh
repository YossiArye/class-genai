# pip-autosave: keeps requirements.txt in sync with the venv.
# Sourced automatically by venv/bin/activate (see scripts/setup-pip-autosave.sh).
# After every successful `pip install` / `pip uninstall` (or pip3), it rewrites
# requirements.txt at the project root with `pip freeze`.

_pip_autosave_run() {
  local pipcmd="$1"; shift
  command "$pipcmd" "$@"
  local rc=$?
  # Only sync inside an active venv, after a successful install/uninstall
  if [ $rc -eq 0 ] && [ -n "$VIRTUAL_ENV" ]; then
    case "$1" in
      install|uninstall)
        local req="$(dirname "$VIRTUAL_ENV")/requirements.txt"
        command "$pipcmd" freeze > "$req" && echo "📦 requirements.txt updated ($(wc -l < "$req" | tr -d ' ') packages)"
        ;;
    esac
  fi
  return $rc
}

pip()  { _pip_autosave_run pip  "$@"; }
pip3() { _pip_autosave_run pip3 "$@"; }
