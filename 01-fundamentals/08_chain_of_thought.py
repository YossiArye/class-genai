"""
08. Chain of Thought (Reasoning)
================================

LLMs generate text one token at a time. Each token is predicted based
on all previous tokens. This means: more tokens = more "thinking time."

Chain of Thought (CoT) works by forcing the model to write out reasoning
steps BEFORE it writes the answer. Each step becomes context that
constrains the next step — and ultimately the final answer.

Without CoT, the model jumps straight to the most statistically likely
answer. That answer is often wrong on problems that require multiple
dependent steps.

The trigger is as simple as appending: "Let's think step by step."

Real contrast shown in this lesson (tested, not theoretical):
  Without CoT: "Lily is 4, her brother is 3× her age. When Lily is 12?"
  → Model answers 36. (It multiplied 12 × 3 — plausible but wrong.)

  With CoT: same question.
  → Model first finds the age gap (8 years), then adds it → 20. Correct.

Prerequisites: Lesson 03 (OpenAI basics)

How to run:
    python 08_chain_of_thought.py

Expected output:
    Without CoT: a single wrong number (36)
    With CoT: a step-by-step breakdown ending with the correct answer (20)
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"

# ── The problem ───────────────────────────────────────────────────────────────
# This question has a trap: the intuitive answer (12 × 3 = 36) is wrong.
# The correct answer requires two steps:
#   1. Find the brother's current age: 4 × 3 = 12
#   2. The age GAP stays constant: 12 - 4 = 8 years
#   3. When Lily is 12: brother = 12 + 8 = 20
#
# Without CoT, the model shortcuts to "3 times Lily's future age" = 36.
PROBLEM = (
    "Lily is 4 years old. Her brother is 3 times her age. "
    "When Lily is 12, how old will her brother be?"
)

# ── 1. Without CoT — answer only ─────────────────────────────────────────────
# Asking for a number only forces the model to commit immediately —
# no room to reason, so it pattern-matches to the most plausible token: 36.
print("=" * 60)
print("WITHOUT CoT — answer only")
print("=" * 60)
print(f"Q: {PROBLEM}\n")

r_direct = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": PROBLEM + "\n\nAnswer with a number only. No explanation."
    }],
)
print(f"A: {r_direct.choices[0].message.content.strip()}")
print("   (correct answer is 20 — the model multiplied 12 × 3 = 36 by instinct)")

# ── 2. With CoT — step by step ────────────────────────────────────────────────
# "Let's think step by step." is the canonical CoT trigger from the 2022
# Wei et al. paper. It prompts the model to write intermediate steps,
# and each step anchors the next one correctly:
#
#   Token N: "the age difference is 8"
#   Token N+1: "12 + 8 =" → model predicts 20, not 36
#
# Without that anchor in the context, token N+1 after "Lily is 12" is
# statistically predicted as "× 3 = 36".
print("\n" + "=" * 60)
print("WITH CoT — 'Let\\'s think step by step.'")
print("=" * 60)
print(f"Q: {PROBLEM}\n")

r_cot = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": PROBLEM + "\n\nLet's think step by step."
    }],
)
print(r_cot.choices[0].message.content.strip())

# ── 3. Bonus: a harder multi-step problem ────────────────────────────────────
# CoT helps even more when there are multiple independent sub-calculations
# that must be combined. Without CoT, the model often drops one sub-step.
#
# This problem requires 5 steps:
#   cups/day dogs → cups/day cats → total cups/day → total cups/week → bags
BONUS = (
    "Sarah has 3 dogs and 2 cats. She feeds each dog 2 cups per meal "
    "and each cat 1 cup per meal. She feeds them twice a day. "
    "Food comes in 10-cup bags. How many bags does she need for a full week?"
)

print("\n" + "=" * 60)
print("BONUS — harder multi-step (without vs with CoT)")
print("=" * 60)

r_b_direct = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": BONUS + "\n\nAnswer with a number only. No explanation."
    }],
)
r_b_cot = client.chat.completions.create(
    model=MODEL,
    messages=[{
        "role": "user",
        "content": BONUS + "\n\nLet's think step by step."
    }],
)

print(f"Without CoT: {r_b_direct.choices[0].message.content.strip()!r}  (correct = 12)")
print(f"\nWith CoT:\n{r_b_cot.choices[0].message.content.strip()}")

print("\n" + "=" * 60)
print("KEY TAKEAWAY:")
print("  More tokens before the answer = more 'compute' spent on reasoning.")
print("  'Let\\'s think step by step.' is free — add it whenever accuracy matters.")
print("  CoT is most powerful on: age/date problems, multi-step word problems,")
print("  logic puzzles, any task where the wrong answer is intuitively plausible.")
print("=" * 60)
