"""
17. Custom Ollama Models (Modelfile)
=====================================
What this lesson teaches:
- How to create a custom Ollama model programmatically using the Python library
- What a Modelfile is and how it maps to the Python API
- How to bake a system prompt AND inference parameters (temperature, top_p, …) into a model
- Safe cleanup patterns: deleting a model before creating it (idempotency)
- Good hygiene: removing ephemeral models when you're done to free disk space

Prerequisites: Lesson 14 (Ollama basics) and Lesson 12 (temperature & top_p)

How to run:
    python 17_ollama_custom_model.py

Expected output:
    A numbered list of your local Ollama models, then a one-sentence cynical
    response from the custom "cynical1" model, then a cleanup confirmation.

Note:
    The companion file `ollama_modelfile` shows how to do the SAME thing as
    Section 3 using a Modelfile and the CLI command:
        ollama create cynical1 -f ollama_modelfile
    Both approaches produce an identical result.
"""

import ollama  # Official Ollama Python client — pip install ollama

# ── Constants ─────────────────────────────────────────────────────────────────
# Keeping magic values as constants makes it easy to swap the base model or
# rename your custom model without hunting through the code.

MODEL_BASE = "llama3.2:latest"  # The model we build ON TOP of (needs to be pulled already)
MODEL_NAME = "cynical"         # The name we give to our new custom model


# ── 1. Setup ──────────────────────────────────────────────────────────────────
# List every model currently stored locally so students can see what they have.

print("=" * 55)
print("SECTION 1 — Local models currently available")
print("=" * 55)

# ollama.list() returns an object with a 'models' attribute containing a list
# of model objects. Each object has at minimum a 'model' field (the full
# name:tag string, e.g. "llama3.2:latest").
models_response = ollama.list()

local_models = models_response.models  # List of model info objects

if not local_models:
    print("  (no models found — have you run `ollama pull llama3.2`?)")
else:
    for i, m in enumerate(local_models, start=1):
        # m.model is the full identifier, e.g. "llama3.2:latest"
        print(f"  {i}. {m.model}")

print()


# ── 2. Safe Cleanup (pre-flight) ──────────────────────────────────────────────
# Before we create the model we want to make sure we're starting from a clean
# slate. If "cynical1" was left over from a previous run, ollama.create() would
# just overwrite it — but being explicit here teaches the delete pattern AND
# avoids any stale-state surprises.

print("=" * 55)
print("SECTION 2 — Pre-flight cleanup")
print("=" * 55)

try:
    # Attempt to delete the model. This WILL raise an exception if the model
    # doesn't exist yet, which is totally fine — we catch it below.
    ollama.delete(MODEL_NAME)
    print(f"  Deleted existing '{MODEL_NAME}' model (cleaning up a previous run).")
except Exception:
    # Model didn't exist — nothing to delete. That's the expected path on a
    # fresh machine. We swallow the error and move on.
    print(f"  '{MODEL_NAME}' not found — nothing to delete. Continuing.")

print()


# ── 3. Create the Custom Model ────────────────────────────────────────────────
# ollama.create() is the Python equivalent of running:
#
#     ollama create cynical -f ollama_modelfile
#
# See the companion file `ollama_modelfile` for the exact Modelfile syntax.
# Both approaches produce an identical model on disk.

print("=" * 55)
print("SECTION 3 — Creating custom model")
print("=" * 55)

print(f"  Building '{MODEL_NAME}' on top of '{MODEL_BASE}'...")

# TIP: `from_` (with the trailing underscore) is used because `from` is a
# reserved keyword in Python, so the Ollama library had to rename the parameter.
#
# `parameters` bakes inference settings into the model — the Python equivalent
# of the PARAMETER directives in `ollama_modelfile`. Once baked in, every call
# to this model uses them automatically; you don't pass them per request.
# These are the same dials taught in lesson 12 (temperature & top_p).
ollama.create(
    model=MODEL_NAME,       # Name of the new model we're registering locally
    from_=MODEL_BASE,       # Base model to inherit weights from
    system=(                # System prompt baked into the model at creation time —
                            # every conversation starts with this context, invisibly
        "You are a cynical and sarcastic assistant. "
        "You answer every question with dry wit and barely concealed disdain, "
        "but you always give a technically correct answer."
    ),
    parameters={
        "temperature": 0.8,      # creativity — enough spark for the sarcasm
        "top_p": 0.9,            # nucleus sampling — varied but blocks the weird tail
        "top_k": 40,             # only sample from the 40 most likely tokens
    },
)

print(f"  '{MODEL_NAME}' created successfully.")
print()


# ── 4. Test the Custom Model ──────────────────────────────────────────────────
# Now we fire a prompt at our newly minted cynical assistant to verify the
# system prompt and parameters are actually baked in.

print("=" * 55)
print("SECTION 4 — Testing the model")
print("=" * 55)

PROMPT = (
    "Tell me what you think about me as an LLM engineer. "
    "Respond in exactly one sentence."
)

print(f"  Prompt : {PROMPT}")
print()

# TIP: ollama.generate() (single-turn, no message history) is simpler than
# ollama.chat() when you just need a one-shot response without passing prior
# conversation turns. Use ollama.chat() when you're building a multi-turn bot.
response = ollama.generate(
    model=MODEL_NAME,
    prompt=PROMPT,
)

# The response object has a 'response' attribute containing the raw text string.
print(f"  {MODEL_NAME} says:")
print(f"  >> {response.response}")
print()


# ── 5. Cleanup ────────────────────────────────────────────────────────────────
# Deleting the model when the lesson is done is good practice:
#   1. Custom models derived from a base model can be recreated in seconds
#      by running the same create() call above — nothing is truly lost.
#   2. Each model copy occupies disk space in the same order of magnitude as
#      the base model (several GB for llama3.2).
#   3. In a teaching context, cleanup keeps the student's `ollama list` tidy.

print("=" * 55)
print("SECTION 5 — Cleanup")
print("=" * 55)

ollama.delete(MODEL_NAME)
print(f"  '{MODEL_NAME}' deleted. Disk space reclaimed.")
print()

print("=" * 55)
print("Lesson 17 complete.")
print("=" * 55)
