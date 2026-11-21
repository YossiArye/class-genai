"""
07. Few-Shot Prompting (In-Context Learning)
============================================

"Show, Don't Tell."

Zero-shot = describe what you want in words.
Few-shot  = show examples of input → output, then ask.

The key insight: examples don't just teach the MODEL WHAT TO DO —
they also teach it WHAT FORMAT TO USE. That second part is what
zero-shot consistently gets wrong.

Real contrast in this lesson:
  Zero-shot: "Login button does nothing on Safari 17."
  → A 10-point numbered essay with markdown headers.

  Few-shot: same question, after 3 examples.
  → "ACTION: Fix Now | PRIORITY: P1 | TEAM: Frontend"

When the output must be parseable (stored in a DB, piped to another
system, fed into a dashboard), few-shot is the only reliable option.

Prerequisites: Lesson 03 (OpenAI basics)

How to run:
    python 07_few_shot.py

Expected output:
    Zero-shot section: a long multi-paragraph response
    Few-shot section: a single compact pipe-delimited line
    Consistency demo: zero-shot format varies; few-shot format is identical
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"

# ── 1. Zero-Shot — the problem ────────────────────────────────────────────────
# No examples, just a question. The model answers in whatever format it likes.
# Because the question sounds like a support conversation, GPT gives a
# conversational essay — completely unusable if you need a structured output.
print("=" * 60)
print("ZERO-SHOT — no examples, open-ended question")
print("=" * 60)

r = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "user", "content": "The app crashes when I try to upload a profile picture."}
    ],
)
print(r.choices[0].message.content)

# ── 2. Few-Shot — the solution ────────────────────────────────────────────────
# Three user/assistant pairs teach two things at once:
#   1. HOW TO CLASSIFY: crash → Fix Now, feature → Backlog, UX issue → Improve
#   2. WHAT FORMAT TO USE: ACTION: X | PRIORITY: Y | TEAM: Z
#
# Neither of these is written in words anywhere. The model infers both from examples.
# This is why it's called "in-context learning" — the examples ARE the training data.
print("\n" + "=" * 60)
print("FEW-SHOT — 3 examples teach format + classification logic")
print("=" * 60)

r = client.chat.completions.create(
    model=MODEL,
    messages=[
        # Example 1 — bug, high priority, frontend
        {"role": "user",      "content": "Login button does nothing on Safari 17."},
        {"role": "assistant", "content": "ACTION: Fix Now | PRIORITY: P1 | TEAM: Frontend"},

        # Example 2 — feature request, low priority
        {"role": "user",      "content": "Can you add a dark mode?"},
        {"role": "assistant", "content": "ACTION: Backlog | PRIORITY: P3 | TEAM: Design"},

        # Example 3 — UX friction, medium priority
        {"role": "user",      "content": "Error toast disappears too fast to read."},
        {"role": "assistant", "content": "ACTION: Improve | PRIORITY: P2 | TEAM: UX"},

        # Actual query — same format as examples
        {"role": "user",      "content": "The app crashes when I try to upload a profile picture."},
    ],
)
print(r.choices[0].message.content)

# ── 3. Consistency demo — run the same 3 inputs through both approaches ───────
# Zero-shot: each input gets a different format (paragraph, list, short sentence…)
# Few-shot: every input gets the identical pipe-delimited structure
print("\n" + "=" * 60)
print("CONSISTENCY DEMO — same 3 inputs, zero-shot vs few-shot")
print("=" * 60)

inputs = [
    "Login button does nothing on Safari 17.",
    "Can you add a dark mode?",
    "Error toast disappears too fast to read.",
]

# Zero-shot — notice the format changes with every response
print("\n--- Zero-Shot (format drifts) ---")
for msg in inputs:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": msg}],
    )
    # Truncate to first line so the output is readable side-by-side
    first_line = r.choices[0].message.content.strip().splitlines()[0]
    print(f"Q: {msg}")
    print(f"A: {first_line}\n")

# Few-shot — format is locked; every response has the same three fields
print("--- Few-Shot (format locked) ---")
# TIP: building a base message list then appending per query avoids repeating
#      the examples. This pattern is standard in production few-shot pipelines.
few_shot_base = [
    {"role": "user",      "content": "Database stops responding under heavy load."},
    {"role": "assistant", "content": "ACTION: Fix Now | PRIORITY: P1 | TEAM: Backend"},
    {"role": "user",      "content": "Add CSV export to the reports page."},
    {"role": "assistant", "content": "ACTION: Backlog | PRIORITY: P3 | TEAM: Backend"},
    {"role": "user",      "content": "Font size feels a bit small on mobile."},
    {"role": "assistant", "content": "ACTION: Improve | PRIORITY: P2 | TEAM: Mobile"},
]

for msg in inputs:
    r = client.chat.completions.create(
        model=MODEL,
        messages=few_shot_base + [{"role": "user", "content": msg}]
    )
    output = r.choices[0].message.content.strip()
    print(f"Q: {msg}")
    print(f"A: {output}")

    # ── 4. Bonus: the output is now machine-parseable ─────────────────────────
    # Because few-shot locked the format, we can reliably split on " | "
    # and extract fields programmatically — something impossible with zero-shot.
    parts = dict(item.split(": ", 1) for item in output.split(" | "))
    print(f"   → parsed: action={parts.get('ACTION')!r}  "
          f"priority={parts.get('PRIORITY')!r}  "
          f"team={parts.get('TEAM')!r}\n")

print("=" * 60)
print("KEY TAKEAWAY:")
print("  Zero-shot → model picks its own format → unparseable in production")
print("  Few-shot  → examples lock the format → consistent, machine-readable")
print("  Rule of thumb: 2-3 examples are usually enough.")
print("=" * 60)
