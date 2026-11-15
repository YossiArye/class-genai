# Python Basics — What Every Student Needs to Know

This guide covers the practical Python knowledge you need before touching any lesson file.
No prior Python experience assumed — we start from zero.

---

## 1. Installing Python

Before anything else, check if Python is already installed:

```bash
python3 --version
```

You should see something like `Python 3.11.4`. If you get `command not found`:

- **Mac:** Install via [python.org](https://python.org/downloads) or with Homebrew:
  ```bash
  brew install python
  ```
- **Windows:** Download the installer from [python.org](https://python.org/downloads).
  During installation, check **"Add Python to PATH"** — this is important.
- **Linux:** Python 3 is usually pre-installed. If not: `sudo apt install python3`

**Minimum version for this course: Python 3.10 or higher.**

---

## 2. `python` vs `python3` — What's the Difference?

Both commands start a Python interpreter. The difference is only which version each name points to,
and that depends on your machine:

| Command | What it means |
|---|---|
| `python3` | Always Python 3 (on Mac/Linux). Fallback if `python` isn't found |
| `python` | Python 3 on modern setups, and always Python 3 **inside an activated venv**. Might not exist on a bare system install; on very old Linux it could be Python 2 |
| `pip` / `pip3` | Package installer for the matching Python. Inside a venv, `pip` is the right one |

**Rule for this course:** use `python` and `pip` everywhere, as the lessons do (`python <file>.py`).
On most setups (Homebrew, python.org installer, conda, any activated venv) `python` is Python 3 and just works.

Only if `python` says `command not found` (some Macs and Linux machines ship only `python3`),
use `python3` and `pip3` instead. Everything else stays the same.

Not sure what you have? Check:

```bash
python --version        # should say Python 3.10 or higher
which python            # Mac/Linux: where it lives (should be inside venv/ once activated)
```

> **Using conda too?** If your prompt shows `(base)` next to `(venv)`, two environments are
> active at once and `python` may point to the wrong one. Run `conda deactivate` (or
> `conda config --set auto_activate_base false` once) so only the venv is active.

---

## 3. How to Run a Python File

Open your terminal, navigate to the project folder, and run:

```bash
python lesson.py
```

**What is the terminal?**
- Mac: open **Terminal** (or iTerm2)
- Windows: open **PowerShell** or **Command Prompt**
- VS Code: the built-in terminal at the bottom works great

**Navigating folders in the terminal:**

```bash
ls                          # list files in the current folder (Mac/Linux)
dir                         # same on Windows
cd projects/llm_tutorial    # move into a folder
cd ..                       # go up one level
pwd                         # show where you are right now
```

**Run from the right folder.** Always `cd` into the project root before running a lesson.
Python looks for imported files relative to where you ran the command:

```bash
cd ~/projects/llm_tutorial
python 00_theory/01_math_of_intelligence.py
```

**The interactive REPL** — a live Python shell, useful for quick tests:

```bash
python                # opens the shell
>>> 2 + 2
4
>>> name = "Yossi"
>>> print(f"Hello {name}")
Hello Yossi
>>> exit()            # or press Ctrl+D to quit
```

---

## 4. Reading Errors (Tracebacks)

When something goes wrong, Python prints a **traceback** — a trail showing where the error
happened. Most beginners read it top-to-bottom and get lost. Read it **bottom-up**:

```
Traceback (most recent call last):
  File "lesson.py", line 12, in <module>          ← step 2: your file, your line
    result = get_embeddings(text)
  File "lesson.py", line 5, in get_embeddings
    return model.encode(text)
ModuleNotFoundError: No module named 'sentence_transformers'   ← step 1: start here
```

1. **Last line first** — this is the actual error message
2. **Work upward** — find the first line that belongs to YOUR file (not library internals)
3. Fix the error at that line

**Common errors:**

| Error | What it means |
|---|---|
| `ModuleNotFoundError` | Package not installed — run `python -m pip install <name>` |
| `SyntaxError` | Typo in the code — check the line number |
| `NameError` | Used a variable before defining it |
| `KeyError` | Tried to access a dict key that doesn't exist |
| `IndentationError` | Python is strict about spaces/tabs — check alignment |

---

## 5. What Is a Virtual Environment?

Imagine you have two projects:
- Project A needs `openai` version 0.28
- Project B needs `openai` version 1.50

If you install both globally, they'd conflict. A **virtual environment (venv)** solves this
by giving each project its own isolated box of packages. Packages installed inside a venv
stay inside it — completely separate from other projects.

**Think of it as a clean Python installation just for this project.**

### Creating a venv from scratch

```bash
# 1. Go to your project folder
cd ~/projects/my_project

# 2. Create the venv (this creates a folder called "venv")
python3 -m venv venv

# 3. Activate it
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

Once active, your terminal prompt changes to show `(venv)` at the start:
```
(venv) yossi@mac my_project %
```

This means **all Python and pip commands now run inside the venv**, not globally.

### Installing packages inside the venv

```bash
pip install openai              # installs openai into the venv only
pip install langchain==0.3.0    # install a specific version
```

### Deactivating

```bash
deactivate    # returns to normal — venv is no longer active
```

### Confirming you're inside the right venv

```bash
which python        # Mac/Linux → should show .../venv/bin/python
where python        # Windows   → should show ...\venv\Scripts\python.exe
```

---

## 6. This Project's venv

For this course, the venv has already been created. You don't need to create it.

**What to do when you first clone the project:**

```bash
# Step 1 — activate the existing venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# Step 2 — install all required packages from the saved list
# (python -m pip guarantees the packages go into the Python you will run)
python -m pip install -r requirements.txt

# Step 3 — confirm it worked
python -c "import openai; print('OK')"
```

> **"No module named pip"?** Some venvs (for example ones created with `uv`) don't include `pip`.
> Either install packages with uv:
> ```bash
> uv pip install -r requirements.txt
> ```
> or add pip to the venv once, then use the normal command:
> ```bash
> python -m ensurepip --upgrade
> python -m pip install -r requirements.txt
> ```
> Venvs created with `python -m venv venv` (as shown in Section 5) always include pip.

If you ever break the venv or need to start fresh:

```bash
# Delete and recreate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 7. Tracking Libraries — requirements.txt and .gitignore

### The problem

When you install a package, it goes into `venv/`. But `venv/` is:
- **Hundreds of MB** — too large to put in git
- **Machine-specific** — paths inside it are hardcoded to your computer
- **Fully reproducible** — anyone can recreate it in one command

So we don't track `venv/` in git. Instead, we track the **list of packages** needed.

### requirements.txt — the package list

After installing something new, snapshot the current state:

```bash
pip freeze > requirements.txt
```

This writes every installed package and its exact version to `requirements.txt`.
Anyone who clones the project runs one command to get the exact same packages:

```bash
pip install -r requirements.txt
```

**Heads up:** `pip freeze` includes not just what you installed, but also all the hidden
dependencies those packages pulled in — often 50–100+ lines. The file works, but it's noisy.
For a cleaner `requirements.txt` with only the packages you actually use, see `pipreqs` below.

### .gitignore — what git ignores

The `.gitignore` file in this project tells git to skip:

```
venv/           ← the entire virtual environment folder
.env            ← your API keys (never commit these!)
db/             ← vector database files (auto-generated)
__pycache__/    ← compiled Python bytecode (auto-generated)
*.pyc           ← same, individual compiled files
```

**The rule: commit `requirements.txt`, never commit `venv/`.**

If you install a new package, always run `pip freeze > requirements.txt` and commit that file.

### No requirements.txt? Use automated tools

Reading imports manually works, but there are better ways.

#### Option 1 — pipreqs (recommended)

`pipreqs` scans your project's import statements, maps them to the correct PyPI package names
(it knows `PIL` → `Pillow`, `sklearn` → `scikit-learn`, etc.), and writes a clean
`requirements.txt` automatically.

```bash
# 1. Install pipreqs
pip install pipreqs

# 2. Run it in your project folder (. means current directory)
pipreqs .

# If requirements.txt already exists and you want to overwrite it:
pipreqs . --force
```

**Why pipreqs is better than `pip freeze`:** `pip freeze` dumps every package in your
environment — including dozens of hidden dependencies that your code never imports directly.
`pipreqs` only lists the packages you actually use, keeping `requirements.txt` clean and
readable.

#### Option 2 — IDE auto-detection (VS Code / PyCharm)

If you're using a modern editor, it detects missing packages from your imports automatically:

- **VS Code** (with the Python extension): missing imports get a yellow squiggly line.
  Hover over it → click **Quick Fix** (or `Ctrl+.`) → select **Install package**.
- **PyCharm**: red squiggly line on the import → press `Alt+Enter` → **Install package**.

#### Option 3 — uv (modern, ultra-fast alternative to pip)

`uv` is a next-generation Python package manager. It can run a script, automatically detect
what's missing, install it into a temporary environment, and execute — all in one command:

```bash
# Install uv (once, globally)
pip install uv

# Run any script — uv handles the environment and missing packages
uv run lesson.py
```

Useful when you want to quickly run a script without setting up a full venv manually.

---

## 8. Python in the Vibe Coding World

In AI-assisted development, your role shifts: **you describe what you want, the AI writes the
code**. But you still need to *read* and *verify* what gets generated.

You don't need to memorize syntax — you need to *recognize patterns* and understand what they do.

Here are the patterns that appear constantly in this course:

---

### f-strings — putting values inside text

```python
name = "Yossi"
score = 0.9234

print(f"Hello {name}")                    # → Hello Yossi
print(f"Score: {score:.2f}")              # → Score: 0.92  (2 decimal places)
print(f"Length: {len(name)} characters")  # → Length: 5 characters
```

The `f` before the quote means: treat `{...}` as Python expressions, not literal text.

---

### List comprehensions — compact loops that build lists

```python
# Long way (a for-loop that builds a list):
results = []
for doc in documents:
    results.append(doc.page_content)

# Short way (list comprehension — same result, one line):
results = [doc.page_content for doc in documents]

# With a filter — only keep long documents:
long_docs = [doc for doc in documents if len(doc.page_content) > 100]
```

Read it as: "give me `doc.page_content` for each `doc` in `documents`."

---

### Dictionary unpacking — merging settings

```python
defaults = {"temperature": 0.7, "model": "gpt-4"}
overrides = {"temperature": 0.0}

# Merge two dicts — overrides wins on conflict:
config = {**defaults, **overrides}
# → {"temperature": 0.0, "model": "gpt-4"}
```

The `**` "unpacks" a dict into key-value pairs. You'll also see it in function calls:
```python
client.chat.completions.create(**config)   # same as passing each key as an argument
```

---

### `with` — automatic cleanup

```python
with open("file.txt", "r") as f:
    content = f.read()
# file closes automatically here, even if an error occurred inside the block
```

The `with` statement guarantees cleanup runs no matter what. In this course you'll see it
with database connections, HTTP sessions, and LangChain callbacks.

---

### `async` / `await` — waiting without freezing

```python
import asyncio

async def get_response(prompt):
    # "await" means: pause HERE and wait for the result, but don't freeze everything else
    response = await client.chat.completions.create(model="gpt-4", messages=[...])
    return response

# Entry point — how you run an async function
asyncio.run(get_response("Hello"))
```

LLM API calls take 1–5 seconds. `async/await` lets the program do other work while waiting
instead of freezing. You don't need to master this now — just recognize the pattern when you
see it.

---

### Type hints — readable labels on functions

```python
def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:
    ...
```

`str` and `list[float]` are hints — they tell you what type each argument should be and what
the function returns. Python does **not** enforce them at runtime; they're for humans (and AI
tools) to understand the code faster.

---

### Decorators — adding behavior to a function

```python
@app.get("/search")           # marks this function as a FastAPI route
@retry(tries=3)               # automatically retries on failure
def search(query: str):
    ...
```

The `@name` above a function means: wrap this function with extra behavior defined by `name`.
You'll see decorators heavily in FastAPI (web routes) and LangChain (tool definitions).

---

### What is FastAPI? (the `@app.get` above)
**FastAPI** is a Python library for building web APIs: programs that wait for requests over HTTP
and send back answers (usually JSON). It's how you turn your Python code into a service that a
website, a mobile app, or another program can call.

Why it matters for GenAI: your RAG pipeline or agent is just Python functions. To let a
frontend or other users use them, you wrap them in an API. FastAPI is the most common way to do that.

**The smallest possible example:**

```python
from fastapi import FastAPI

app = FastAPI()                       # the server application

@app.get("/hello")                    # decorator: "answer GET requests to /hello"
def hello(name: str):                 # type hint: FastAPI reads ?name=... and checks it's a string
    return {"message": f"Hello {name}"}   # returned dict is sent back as JSON
```

Install and run it (the `uvicorn` server is what actually listens for requests):

```bash
python -m pip install fastapi uvicorn
uvicorn my_file:app --reload          # my_file.py contains the code above
```

Then open `http://127.0.0.1:8000/hello?name=Yossi` in a browser:

```json
{"message": "Hello Yossi"}
```

**Why people like it:**

| Feature | What you get |
|---|---|
| **Type hints = validation** | Wrong input types are rejected automatically with a clear error |
| **Automatic docs** | Open `http://127.0.0.1:8000/docs` for a ready-made page to try every endpoint |
| **`async` support** | Handles many users at once, great for slow LLM calls (see Section 10) |
| **Pydantic models** | Define request and response shapes as classes, which also fits structured LLM output |
| **Streaming** | Send an LLM answer token by token, like ChatGPT does |

It builds on three things from this section: **decorators** (`@app.get`), **type hints**, and
**`async`/`await`**.

---

### Imports — where code comes from

```python
import os                               # built-in module (no install needed)
from pathlib import Path                # one specific class from a built-in module
from langchain_openai import ChatOpenAI # from an installed package
import numpy as np                      # installed package, aliased as "np"
```

**If you see `ModuleNotFoundError`:** the package isn't installed.
Run `pip install <package-name>` with the venv active.

**If you see `ImportError`:** the package is installed but the specific name you're importing
doesn't exist inside it — check the spelling or the package version.

---

## 9. Asking AI About Errors (the Right Way)

More context = better answer. Always paste the **full traceback**, not just the error message:

```
❌ Bad:  "I'm getting a ModuleNotFoundError"

✅ Good: "I'm running python 00_theory/01_math_of_intelligence.py
         with the venv active. I get this error:

         Traceback (most recent call last):
           File "2-rag-vector-databases/1.chroma_intro_in_memory.py", line 3, in <module>
             import chromadb
         ModuleNotFoundError: No module named 'chromadb'

         I already ran pip install -r requirements.txt."
```

The more specific you are about what you ran, what you expected, and what you got — the faster
you'll get a precise fix.

---

## 10. Python Concurrency in 30 Seconds

Python has a **GIL (Global Interpreter Lock)** — only one thread executes Python code at a time.
The GIL is released during I/O (network calls, file reads), so threads *can* overlap there.
But that still doesn't make threading the right tool for every situation:

| Problem | Right tool | Why |
|---|---|---|
| Script making one API call | Sync (default) | No concurrency needed — keep it simple |
| A few tasks at once | `threading` | GIL released during I/O; fine for small numbers of threads |
| Server handling many users | `asyncio` | Single thread, no OS overhead — scales to thousands of connections |
| CPU-heavy work (data crunching) | `multiprocessing` | Spawns separate processes, each with its own GIL — true parallelism |

**In this course:** sync code is almost always the right choice. `async/await` appears when
we build FastAPI servers. You won't need `multiprocessing` unless you're batch-processing
large datasets.
