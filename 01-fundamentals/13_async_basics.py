"""
13. Async Basics
================
What this lesson teaches:
- Why async/await exists and when you need it
- How to use asyncio.gather() to run multiple API calls at once
- How parallel API calls collapse total wait time from SUM to MAX

The demo sends the SAME 3 questions two ways and times each:
  1. Sequential — wait for each answer before sending the next  → times add up
  2. Parallel   — fire all 3 at once with asyncio.gather()       → pay for the slowest

Why a "warmup" call first?
    The very first HTTPS request in a program is slow (~2-4s) because Python
    has to OPEN the network connection to OpenAI (DNS lookup + TLS handshake).
    Every call after that reuses the same pooled connection and is much faster.
    If we didn't warm up, that one-time setup cost would land on the first
    timed call and make the comparison misleading.
    (This connection cost — NOT prompt caching — is the real reason the first
    call looks slow. OpenAI's prompt cache only applies to prompts over ~1024
    tokens; ours are tiny, so caching never plays a role here. That also means
    reusing the same questions in both sections is a perfectly fair test.)

Prerequisites: Python basics, OpenAI API key in a .env file, lessons 03-12

How to run:
    python 13_async_basics.py

Expected output:
    Sequential: 3 answers one after another, ~5-7s total
    Parallel:   the same 3 answers arriving together, ~2-3s total
    Comparison: parallel is roughly 3x faster (paying for MAX, not SUM)
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────

import asyncio                          # Python's built-in async event loop
import time                             # For wall-clock timing measurements
from openai import AsyncOpenAI          # Async version of the OpenAI client
from dotenv import load_dotenv          # Loads API key from .env into os.environ

# ── 2. Configuration ──────────────────────────────────────────────────────────

load_dotenv()                           # Must be called before AsyncOpenAI() reads env vars

MODEL = "gpt-4o-mini"
MAX_TOKENS = 60

# The same 3 questions are used for BOTH the sequential and parallel runs,
# so each mode does identical work — the only thing that changes is HOW we
# wait for the answers. (Safe to reuse: these prompts are far too short to
# trigger OpenAI's prompt cache, so neither run gets an unfair head start.)
QUESTIONS = [
    ("neural-net", "What is a neural network? One sentence only."),
    ("backprop",   "What is backpropagation? One sentence only."),
    ("gradient",   "What is gradient descent? One sentence only."),
]

# ── 3. Async helper — a single API call ──────────────────────────────────────

# TIP: "async def" marks a coroutine — a function that can pause and resume.
#      Calling ask(...) does NOT run it immediately; it returns a coroutine object.
#      You must either "await" it or schedule it with asyncio.gather / asyncio.run.
async def ask(client: AsyncOpenAI, label: str, question: str) -> None:
    """Send one question, then print the answer with how long that call took."""
    start = time.perf_counter()

    # TIP: "await" pauses THIS coroutine until the API responds, but it does NOT
    #      block other coroutines. The event loop can run other tasks while we wait.
    #      That is exactly what makes parallel mode possible.
    response = await client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": question}],
    )

    elapsed = time.perf_counter() - start
    answer = response.choices[0].message.content.strip()
    print(f"  [{label}] finished in {elapsed:.2f}s — {answer}")


# ── 4. Warmup — open the connection before we measure anything ────────────────

async def warmup(client: AsyncOpenAI) -> None:
    """
    Make one tiny throwaway call so the HTTPS connection is already open
    before timing starts. This removes the ~2-4s first-request setup cost
    from the comparison so both modes are measured fairly.
    """
    print("Warming up the connection (one throwaway call, not timed)…")
    await client.chat.completions.create(
        model=MODEL,
        max_tokens=1,
        messages=[{"role": "user", "content": "hi"}],
    )
    print("Connection ready.\n")


# ── 5. Sequential mode — one call, then the next ─────────────────────────────

async def sequential(client: AsyncOpenAI) -> float:
    """Await each call in turn. Total time = sum of every call."""
    print("── Sequential: send a question, wait for the answer, repeat ──")
    start = time.perf_counter()

    # Each "await" blocks here until that call returns before starting the next.
    for label, question in QUESTIONS:
        await ask(client, label, question)

    total = time.perf_counter() - start
    print(f"  → Sequential total: {total:.2f}s  (the call times ADD UP)\n")
    return total


# ── 6. Parallel mode — all calls at once ─────────────────────────────────────

async def parallel(client: AsyncOpenAI) -> float:
    """Fire all calls together with gather(). Total time ≈ the slowest one."""
    print("── Parallel: fire all questions at once, wait for all answers ──")
    start = time.perf_counter()

    # TIP: asyncio.gather() schedules every coroutine on the event loop at once,
    #      then waits for all of them to finish. While one call waits on the
    #      network, the others are already in flight — so the total wall-clock
    #      time is about the SLOWEST single call, not the sum.
    # TIP: the * unpacks the list into separate arguments, which gather() expects.
    await asyncio.gather(*[
        ask(client, label, question)
        for label, question in QUESTIONS
    ])

    total = time.perf_counter() - start
    print(f"  → Parallel total: {total:.2f}s  (you pay for the SLOWEST call)\n")
    return total


# ── 7. Main orchestrator ──────────────────────────────────────────────────────

async def main() -> None:
    # One client, reused everywhere — it keeps a pool of open HTTP connections.
    client = AsyncOpenAI()

    await warmup(client)                 # open the connection BEFORE we measure
    seq_time = await sequential(client)
    par_time = await parallel(client)

    # ── Comparison ────────────────────────────────────────────────────────────
    speedup = seq_time / par_time if par_time > 0 else float("inf")

    print("── Result ───────────────────────────────────────────────")
    print(f"  Sequential : {seq_time:.2f}s   (waited for each call in turn)")
    print(f"  Parallel   : {par_time:.2f}s   (all calls overlapped)")
    print(f"  Parallel was {speedup:.1f}x faster — same work, a fraction of the wait.")
    print()
    print("  Why: the calls spend almost all their time WAITING on the network.")
    print("  Sequential wastes that wait doing nothing. Parallel overlaps the")
    print("  waits, so 3 calls take about as long as 1. The more calls you")
    print("  fan out, the bigger the win — batch embeddings, multi-tool agents, etc.")


# ── 8. Entry point ────────────────────────────────────────────────────────────

# TIP: asyncio.run() creates a fresh event loop, runs the coroutine, then closes
#      it. It's the standard way to start async code from a normal script. Don't
#      call it inside an already-running loop (e.g. a Jupyter cell — use
#      "await main()" there instead).
if __name__ == "__main__":
    asyncio.run(main())
