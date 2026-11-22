"""
12. Controlling Creativity (Temperature, Top P & Top K)
=======================================================

LLMs don't pick the single "best" next word. They work like a lottery:
compute a probability for every word in the vocabulary, then DRAW one.
A word with 60% probability has a 60% chance of being drawn — but the
model might still land on a rarer word. This is what creates variety.

Temperature and Top P both act on this lottery — but at different stages.
They work SEQUENTIALLY, not as alternatives — which is why this lesson
explains and demonstrates them together (see the ⚠️ OpenAI note below for
what is recommended in practice).

HOW THE LOTTERY ACTUALLY WORKS (the "roulette wheel")
  The computer can only do one random thing: pick a number between 0 and 1,
  where every number is equally likely. The model turns that into a
  WEIGHTED draw by splitting the 0→1 line into slices, one per word, each
  slice as wide as that word's probability:

    0.0 ----------- 0.6 ------- 0.8 ----- 0.9 --- 0.95 --- 1.0
    |     house      |   work   | store  | park |  other  |
    |      60%       |   20%    |  10%   |  5%  |   5%    |

  Draw 0.31 → lands in "house". Draw 0.73 → "work". Draw 0.84 → "store".
  The random number itself is NOT biased — every point on the line is equally
  likely. The bias comes from the slice WIDTHS: a wide slice is a bigger
  target. Think of a dartboard that is 60% red: throw blindly and you will hit
  red most of the time, but not always.

  Everything in this lesson is about changing the slice widths:
    • temperature   → makes the slices more unequal (low) or more equal (high)
    • top_p / top_k → deletes the thin slices entirely, then the survivors
                      are scaled back up so they sum to 100% again

WHY DRAW AT ALL? WHY NOT ALWAYS PICK THE TOP WORD?
  1. There often IS no single "right" word. "Yesterday I went to the ..."
     can continue with house / work / store / park — all valid. The
     probabilities just reflect how often each one appears in real text.
  2. When there IS a right answer ("The capital of France is ..."), the top
     word has ~97%, so the lottery is nearly rigged anyway — you get the same
     answer almost every time without forcing it.
  3. Always picking the top word ("greedy decoding", i.e. temperature=0) has
     real downsides: bland, cliché text; repetition loops ("I think that I
     think that I think..."); and the most likely word at each step does not
     guarantee the best sentence overall.
  4. People want variety: ask for 5 business names and you want 5 different ones.
  5. A 10% word is not a mistake when it wins — it wins about 1 draw in 10.
     And a sentence is MANY draws in a row: if every word is 90% certain, the
     chance that ALL 20 words are the top choice is 0.9^20 ≈ 12%. So most
     sentences contain at least one "less likely" word — that is what
     creativity feels like.

──────────────────────────────────────────────────────────────────
STEP 1 — TEMPERATURE: reshapes the probability scores
──────────────────────────────────────────────────────────────────
Before the lottery, the model has raw scores (logits) for every word.
Temperature divides every score by the temperature value, then those
divided scores are converted to percentages via softmax.

  Softmax in two steps: (1) raise e (≈ 2.718) to the power of each score,
  (2) divide each result by the sum of all of them → percentages that add up to 100%.

  Worked example — four words with raw scores (logits):
      house = 2.0   work = 0.9   store = 0.2   park = -0.5

    temp=1.0 (no change):  scores / 1.0 → 2.0, 0.9, 0.2, -0.5
      e^score → 7.39, 2.46, 1.22, 0.61        (sum = 11.68)
      house 63%  | work 21%   | store 10.5% | park 5%

    temp=0.5 (low):        scores / 0.5 → 4.0, 1.8, 0.4, -1.0
      e^score → 54.6, 6.05, 1.49, 0.37        (sum = 62.5)
      house 87%  | work 9.7%  | store 2.4%  | park 0.6%

    temp=0.2 (very low):   scores / 0.2 → 10, 4.5, 1.0, -2.5
      e^score → 22026, 90, 2.7, 0.08          (sum = 22119)
      house 99.6% | work 0.4% | store 0.01% | park ~0%

    temp=1.5 (high):       scores / 1.5 → 1.33, 0.6, 0.13, -0.33
      e^score → 3.79, 1.82, 1.14, 0.72        (sum = 7.48)
      house 51%  | work 24%   | store 15%   | park 10%

  WHY does dividing by a small number widen the gaps? The gap between house
  and work is 2.0 - 0.9 = 1.1. Dividing by 0.5 turns it into 2.2; dividing by
  0.2 turns it into 5.5. Then e^gap turns the gap into a RATIO: at temp=1.0
  house is e^1.1 ≈ 3x as likely as work; at temp=0.2 it is e^5.5 ≈ 245x as
  likely. Dividing by a number above 1 does the opposite: it squeezes the
  gaps, so the ratios shrink toward 1 and the wheel gets flatter.

  ⚠️ temperature=0 would mean dividing by zero, so APIs special-case it:
  they simply pick the highest-scoring word (greedy).

  Low temperature (e.g. 0.2):
    Widens the gaps — the leading words get almost 100%, rare words
    approach 0%. The lottery is nearly rigged toward the top word.
    → Predictable, precise, conservative text.

  High temperature (e.g. 1.5):
    Shrinks the gaps — takes probability away from the top words and
    redistributes it to rarer, weirder words. The lottery is flatter.
    → Creative, surprising, sometimes incoherent text.

  temperature=0.0    → always picks the peak (deterministic)
  temperature=0.5-0.7 → balanced (default for chat)
  temperature=1.0    → creative, varies each run
  temperature>1.0    → unpredictable, risks hallucination

──────────────────────────────────────────────────────────────────
STEP 2 — TOP P: filters the pool AFTER temperature reshapes it
──────────────────────────────────────────────────────────────────
Top P acts like a bouncer. After temperature has set the probabilities,
Top P sorts all words from most likely to least likely and keeps adding
them to the draw pool until their cumulative probability reaches P.
The remaining words — no matter how many — are discarded entirely.
The survivors' slices are then rescaled to sum to 100% again, e.g. a pool of
{A: 60%, B: 20%, C: 10%} becomes A 67% | B 22% | C 11% before the draw.

Example (after temperature has set the percentages):
  Word A: 60%  |  Word B: 20%  |  Word C: 10%  |  Word D: 5%  |  ...

  top_p=0.5 → keep adding until cumulative ≥ 50%:
    Word A alone = 60% → already ≥ 50% → pool = {A} only.
    Word D is locked out even at high temperature.

  top_p=0.9 → keep adding until cumulative ≥ 90%:
    A=60%, A+B=80%, A+B+C=90% → pool = {A, B, C}.
    Word D (5%) is cut — it would push cumulative over 90%.

  top_p=1.0 → keep everything → no filtering (full vocabulary).

  Why this is powerful: top_p ADAPTS to the model's confidence. When the
  model is very sure (one word ~90%), the pool stays tiny; when it's unsure
  (many words at 5-10%), the pool grows automatically.

WHY USE BOTH TOGETHER?
  High temperature + low top_p = "controlled creativity":
    Temperature flattens the distribution (more variety), but
    Top P blocks the truly absurd bottom of the list.
    Result: creative but still coherent.

  Low temperature + high top_p = standard precise mode:
    Temperature sharpens the distribution, Top P barely matters
    because the top words already dominate.

  ⚠️ OpenAI note — what the API docs actually recommend:
    For both parameters the OpenAI API reference says: "We generally recommend
    altering this or `temperature` but not both." In practice that is because
    the two interact, so changing both at once makes it hard to tell which
    dial caused a change in behavior. Practical workflow: tune ONE dial first
    (usually temperature) and leave top_p at 1.0, unless you have a specific
    reason to block the tail at high temperature. This lesson sets both on
    purpose, to show how they interact — in your own projects, change one,
    test, and only then decide whether you need the other.

──────────────────────────────────────────────────────────────────
STEP 3 (BONUS) — TOP K: a simpler, count-based filter
──────────────────────────────────────────────────────────────────
Top K is the older, simpler cousin of Top P. Instead of adding words
until a cumulative PROBABILITY is reached, Top K just keeps a fixed
NUMBER of the most likely words and throws away the rest.

Example (same distribution):
  Word A: 60%  |  Word B: 20%  |  Word C: 10%  |  Word D: 5%  |  ...

  top_k=1  → keep only the 1 most likely word  → {A}        (always picks A)
  top_k=3  → keep the 3 most likely words      → {A, B, C}  (D and beyond cut)
  top_k=40 → keep the top 40 words             → a wide, varied pool

Top P vs Top K — the key difference:
  • Top P is ADAPTIVE: when the model is confident the nucleus shrinks;
    when it's unsure the nucleus grows. The size flexes with certainty.
  • Top K is FIXED: it always keeps exactly K words, confident or not.
    Simpler, but it can't adapt — top_k=40 keeps 40 words even when the
    model is 99% sure of one answer.
  In practice top_p is usually preferred; top_k is still common, and the
  two are often combined (apply top_k first, then top_p on what remains).

  Two scenarios that make the difference concrete:

  Confident model — "The capital of France is ...":
    Paris 97% | Lyon 1% | Nice 0.5% | ...thousands of tiny ones
    top_p=0.9 → Paris alone already covers 97% → pool = {Paris}. Nothing wasted.
    top_k=40  → keeps 40 words no matter what; 39 are junk that each still
                have a (tiny) chance of being drawn by mistake.

  Unsure model — "Yesterday I went to the ...":
    house 12% | work 11% | store 10% | park 9% | gym 8% | cafe 7% | ...
    top_p=0.9 → needs roughly 15-20 words to reach 90% → the pool grows by itself.
    top_k=3   → keeps only house/work/store and throws away gym and cafe,
                which are almost as likely — just because they ranked 5th and 6th.

  Why combine them: top_k first makes a cheap, rough cut (say 40 candidates out
  of ~100,000 words — which can also be faster, since you don't sort the whole
  vocabulary), then top_p does the smart, confidence-aware trimming on those 40.

⚠️ OpenAI's API does NOT expose top_k.
  The OpenAI chat API gives you only `temperature` and `top_p` — there is
  no `top_k` parameter. That's why the runnable demos below use just those
  two. (Passing top_k to the OpenAI client would raise an error.)

  Top K IS available in other runtimes — notably **Ollama**, which exposes
  `top_k`, `top_p`, AND `temperature`. You'll see top_k used for real in
  the Ollama lessons (14-17) — especially lesson 17, where the line
  `PARAMETER top_k 40` bakes it straight into a custom model.

──────────────────────────────────────────────────────────────────
RULE OF THUMB: temperature controls the shape; top_p controls the size.
If you do use both, do it deliberately, not randomly (and keep the ⚠️ OpenAI
note above in mind). Common combos:
  Creative + safe:    temperature=1.2, top_p=0.9  (flatten, but block the tail)
  Precise:            temperature=0.2, top_p=1.0  (sharp curve, no filtering needed)
  Chatbot default:    temperature=0.7, top_p=0.9  (slight creativity, some filtering)

By task type:
  Code, math, data extraction (one right answer)  → low temperature
  Chatbot / general Q&A                            → medium temperature, some filtering
  Slogans, stories, brainstorming                  → high temperature + strong top_p filtering
──────────────────────────────────────────────────────────────────

Prerequisites: Lesson 03 (OpenAI basics)

How to run:
    python 12_temperature_and_top_p.py

Expected output:
    Temperature section: low temp = predictable story opening;
                         high temp = surprising/unusual story opening.
    Top P section: low top_p = safe vocabulary; high top_p = richer word choice.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODEL = "gpt-4o-mini"


def completions(prompt, temperature, top_p, n=3, max_tokens=15):
    """Request n completions in one call — lets us compare variance side by side."""
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,  # STEP 1: reshapes the odds (divide scores, then softmax)
        top_p=top_p,              # STEP 2: filters the pool AFTER temperature has reshaped it
        # ⚠️ There is no top_k=... argument here — OpenAI doesn't expose it (see section 4).
        # TIP: n=3 asks the API to generate 3 independent completions in one call.
        #      Perfect for showing consistency (temp=0) vs. variance (temp=1.5)
        #      without running the script multiple times.
        n=n,
        max_tokens=max_tokens,
    )
    return [c.message.content.strip() for c in r.choices]


# ── 1. Temperature: determinism vs. variance ─────────────────────────────────
# Prompt: one-word completion. Short output = differences are impossible to miss.
# The "safe" answer (books / dust) has ~80% probability at the top.
#
# temp=0.0: the lottery is rigged → same word every single draw.
#   All 3 completions are identical — that IS the point.
#
# temp=0.7: the curve is slightly flattened → occasionally a less obvious
#   word wins, but the output stays natural and readable.
#
# temp=1.5: the curve is very flat → the top word no longer dominates.
#   All 3 completions land on DIFFERENT words — creative, surprising,
#   sometimes poetic.
#
# TIP: each of the 3 completions is its own independent lottery draw. At temp=0
#   there is nothing to draw — the top word always wins. At temp>0 the wheel is
#   spun again for every completion (and for every word inside each one).
#
# ⚠️ Common mistake: expecting temp=0.7 to ALWAYS give different answers.
#   It is a lottery, not a rule — with a strong favourite you will often see
#   repeats even at 0.7. Run the script a few times to see the real spread.

WORD_PROMPT = "Complete with FIVE WORDS only, nothing else: 'The old library smelled like'"

print("=" * 60)
print("1. TEMPERATURE — same prompt asked 3 times, watch consistency")
print(f"   Prompt: '{WORD_PROMPT}'")
print("=" * 60)

for temp, label in [
    (0.0, "deterministic — 3 draws, all identical (lottery is rigged)"),
    (0.7, "balanced — small variation, still safe words"),
    (1.5, "creative — 3 draws, all different (lottery is fair)"),
]:
    words = completions(WORD_PROMPT, temperature=temp, top_p=1.0)
    print(f"\n  temp={temp}  [{label}]")
    for i, w in enumerate(words, 1):
        print(f"    draw {i}: {w!r}")

# ── 2. Top P: vocabulary filter at high temperature ──────────────────────────
# temperature=1.5 throughout — the distribution is flat, many words compete.
# Now we add the top_p bouncer and watch it restrict the vocabulary.
#
# top_p=0.1 → tiny nucleus (cumulative ≥ 10% = usually just the top 1 word).
#   Only the safest/most-common words survive.
#   ⚠️ Expect LITTLE variation here: at every position where the top word has
#   ≥ 10%, the pool is that ONE word — so top_p=0.1 largely cancels the variety
#   that temperature=1.5 created. Variation only appears at the rare positions
#   where the model is so unsure that no single word reaches 10%.
#
# top_p=0.5 → medium nucleus — a handful of reasonable words allowed.
#
# top_p=1.0 → no filter — the full vocabulary is eligible.
#   Rare, unusual, or poetic words can now appear.
#
# KEY INSIGHT: top_p doesn't make output more creative — that's temperature's job.
# top_p CONTROLS HOW WILD the creativity can get.
#
# TIP: top_p is applied to EVERY word of the output separately, each with its
#   own pool. Where the model is confident the pool is 1 word; where it is
#   unsure the pool is big. That is what "adaptive" means.
#
# In roulette-wheel terms: temperature changes how wide each slice is;
#   top_p deletes the thin slices and rescales what is left to 100%.
#   e.g. pool {A: 60%, B: 20%, C: 10%} → rescaled to A 67%, B 22%, C 11%.

SENTENCE_PROMPT = "Describe thunder in exactly one sentence."

print("\n" + "=" * 60)
print("2. TOP P — vocabulary filter (temperature=1.5 fixed for all)")
print(f"   Prompt: '{SENTENCE_PROMPT}'")
print("=" * 60)

for top_p, label in [
    (0.1, "tiny nucleus  — only the most common words survive"),
    (0.5, "medium nucleus — broader word choice, still safe"),
    (1.0, "no filter     — full vocabulary, rare/poetic words can appear"),
]:
    sentences = completions(SENTENCE_PROMPT, temperature=1.5, top_p=top_p, n=2, max_tokens=40)
    print(f"\n  top_p={top_p}  [{label}]")
    for i, s in enumerate(sentences, 1):
        print(f"    → {s}")

# ── 3. The four combinations — the full 2×2 picture ─────────────────────────
# Prompt: a marketing tagline. Needs creativity, has a clear "cliché" default.
# All four use the same prompt — compare the tone and vocabulary directly.
#
#  temp=0.0, top_p=1.0 → PRECISE:
#    Sharp curve picks the most probable word every time.
#    Result: safe, predictable, corporate tagline.
#
#  temp=1.5, top_p=1.0 → WILD:
#    Flat distribution + full vocabulary → anything can happen.
#    Result: might be brilliant, might be bizarre.
#
#  temp=0.7, top_p=0.9 → CONTROLLED (the sweet spot):
#    Temperature flattens the curve a little (some creative choices), but
#    top_p cuts the truly absurd tail. Fresh-ish language, still readable.
#    ⚠️ Why not temp=1.5 + top_p=0.1? With a 10% cutoff the pool is usually ONE
#    word, so it behaves almost like temp=0 — the high temperature is wasted.
#    For more creativity, raise temperature AND keep top_p high (e.g. 1.2 / 0.9).
#
#  temp=0.0, top_p=0.1 → ULTRA-SAFE:
#    Sharp curve + tiny nucleus. Barely differs from PRECISE because
#    at temp=0 the top word already dominates — top_p adds nothing.
#
# HOW TO READ THIS GRID: temperature sets the SHAPE of the odds, top_p sets the
#   SIZE of the pool. top_p only visibly matters when temperature is high enough
#   that many words compete — at temp=0 the pool is effectively one word anyway.
#
# TIP: with only n=2 samples per row, differences can be subtle. Run the script
#   a few times (or raise n) — one run is a few spins of the wheel, not the full picture.

TAGLINE_PROMPT = "Write a one-sentence with 20 WORDS only about army"

print("\n" + "=" * 60)
print("3. ALL FOUR COMBINATIONS — same prompt, same model")
print(f"   Prompt: '{TAGLINE_PROMPT}'")
print("=" * 60)

for temp, top_p, label in [
    (0.0, 1.0, "PRECISE           temp=0.0, top_p=1.0"),
    (1.5, 1.0, "WILD              temp=1.5, top_p=1.0"),
    (0.7, 0.9, "CONTROLLED        temp=0.7, top_p=0.9  ← sweet spot"),
    (0.0, 0.1, "ULTRA-SAFE        temp=0.0, top_p=0.1  ← same as PRECISE"),
]:
    taglines = completions(TAGLINE_PROMPT, temperature=temp, top_p=top_p, n=2, max_tokens=30)
    print(f"\n  [{label}]")
    for t in taglines:
        print(f"    → {t}")

# ── 4. A third dial: TOP K (not demoable here — OpenAI has no top_k) ──────────
# We can't run a top_k demo because the OpenAI API doesn't support it.
# It's printed here so the concept is on your radar before the Ollama lessons.
#
# Mental model: top_k is a fixed-size doorway ("only the first K in line get in");
#   top_p is a doorway that resizes itself to the crowd.
#   Confident model ("capital of France is ..."): top_p → pool of 1; top_k=40 → 40 words (39 junk).
#   Unsure model ("yesterday I went to the ..."): top_p → big pool; top_k=3 → drops good words ranked 4th+.
print("\n" + "=" * 60)
print("4. TOP K — the dial OpenAI does NOT have")
print("=" * 60)
print("  top_k keeps a FIXED NUMBER of the most likely words, then samples.")
print("    top_k=1  → only the single most likely word (fully deterministic)")
print("    top_k=40 → the 40 most likely words (a wide, varied pool)")
print()
print("  Top P vs Top K:")
print("    top_p adapts to the model's confidence (nucleus grows/shrinks)")
print("    top_k is fixed (always exactly K words, confident or not)")
print()
print("  ⚠️ OpenAI's API exposes only temperature + top_p — no top_k.")
print("     You'll use top_k for real in the Ollama lessons (14-17),")
print("     e.g. lesson 17's Modelfile line: PARAMETER top_k 40")
