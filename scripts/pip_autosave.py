"""pip-autosave: keeps requirements.txt in sync with the venv.

Installed into the venv's site-packages by scripts/setup-pip-autosave.sh
(loaded at every Python start via pip_autosave.pth). After any
`pip install` / `pip uninstall` run with this venv's Python - `pip`, `pip3`,
`venv/bin/pip`, `python -m pip`, VS Code/PyCharm installs - it rewrites
<project>/requirements.txt with `pip freeze`.
"""
import atexit
import os
import sys


def _sync():
    argv = sys.argv
    if not argv or os.environ.get("PIP_AUTOSAVE_RUNNING"):
        return
    prog = os.path.basename(argv[0])
    is_pip = prog.startswith("pip") or "/pip/" in argv[0].replace("\\", "/")
    if not is_pip or not ({"install", "uninstall"} & set(argv[1:])):
        return

    import subprocess

    req = os.path.join(os.path.dirname(sys.prefix), "requirements.txt")
    env = dict(os.environ, PIP_AUTOSAVE_RUNNING="1")
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, env=env, check=True,
        ).stdout
    except Exception as e:  # never break pip because of the hook
        print(f"pip-autosave: could not update requirements.txt ({e})", file=sys.stderr)
        return
    with open(req, "w") as f:
        f.write(out)
    print(f"📦 requirements.txt updated ({len(out.splitlines())} packages)")


atexit.register(_sync)
