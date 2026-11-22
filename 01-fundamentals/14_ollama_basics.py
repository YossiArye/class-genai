"""
14. Running Local Models (Ollama)
=================================

Cloud APIs (like OpenAI) are powerful, but sometimes you want privacy, no costs,
or offline capability. **Ollama** is a popular tool to run models (like Llama 3, Mistral)
on your own computer.

Prerequisites:
1.  Download Ollama from https://ollama.com/
2.  Start the Ollama service — three ways, pick one:

    Option A — desktop app (macOS / Windows, recommended):
        Install Ollama → open the app → a cat icon appears in the menu bar (Mac)
        or system tray (Windows). It auto-starts at every login, no terminal needed.
        To stop: click the icon → Quit.
        To restart after quitting: search "Ollama" in Spotlight (Mac) or Start (Windows).

    Option B — terminal daemon (Linux / Docker / no GUI):
        $ ollama serve
        Keep this terminal open while running models.

    Option C — already running?
        If the cat icon is visible in the menu bar / system tray, Ollama is running.
        No extra step needed — just run the script.

3.  Pull a model: Run `ollama pull llama3.2` in your terminal.
4.  `pip install ollama`

Concepts Covered:
1.  **Local Inference**: The model runs on YOUR CPU/GPU, not a server.

    CPU (Central Processing Unit) — the main chip in every computer. A few powerful
    cores (4–16) designed to handle one task at a time very fast. Good at sequential logic.

    GPU (Graphics Processing Unit) — originally built for video games. Thousands of small
    weaker cores that all run simultaneously. Bad at sequential tasks, extremely fast at
    doing the same math operation on thousands of numbers at once.

    LLMs are giant matrix multiplications — the same math repeated billions of times per
    token. That's exactly what GPUs are built for:

        CPU:  8 cores   → 8 multiplications at a time   → slow for LLMs
        GPU:  4000 cores → 4000 multiplications at once → fast for LLMs

    In practice with Ollama:
    - No GPU (CPU only):      works, but slow — ~3–8 tokens/second on a modern laptop
    - Apple Silicon (M1/M2/M3): built-in GPU shares memory with CPU, Ollama uses it
                                automatically — much faster (~30–60 tokens/second)
    - NVIDIA GPU:             fastest option, used in data centers and gaming PCs

    When OpenAI responds fast it's because their servers run racks of NVIDIA A100/H100
    GPUs. When Ollama feels slower, it's the same math running on your laptop CPU.

2.  **Sync API**: Waiting for the full response at once (similar to the OpenAI example).
3.  **Streaming**: Printing tokens as they arrive instead of waiting for the full reply.
4.  **Model Selection**: Using open-weights models (e.g., Llama 3.2).
5.  **Model Management**: Listing the models installed locally via Python.
"""

import ollama

# Usually 'llama3.2' is a good balance of speed and smarts for laptops.
MODEL_NAME = "llama3.2"

print(f"Querying local model: {MODEL_NAME}...")

try:
    # ── 1. Sync Chat ───────────────────────────────────────────────────────────
    # This connects to the local Ollama service (usually running on port 11434).
    # Since it's local, we don't need an API Key!
    # ollama.chat() waits for the FULL reply before returning — simple but slower
    # to first output than streaming.
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": "Explain the concept of 'recursion' in programming using a joke."
            },
        ],
    )

    # The structure is very similar to OpenAI's response object.
    content = response['message']['content']

    print("\n--- Response ---")
    print(content)

    # ── 2. Streaming Chat ──────────────────────────────────────────────────────
    # Setting stream=True returns a Generator — an iterable that yields one chunk
    # per token as the model produces it.  This is why chatbots feel "live":
    # the user sees words appearing instead of staring at a blank screen.
    print("\n\n--- Streaming Answer ---")

    stream = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Count to 5 explicitly."}
        ],
        stream=True,
    )

    for chunk in stream:
        # TIP: Each chunk contains only the NEW token(s) since the last chunk —
        # not the full text so far. Print without newline so it flows naturally.
        part = chunk['message']['content']
        print(part, end="", flush=True)

    print("\n--- Done ---")

    # ── 3. List Local Models ───────────────────────────────────────────────────
    # ollama.list() queries the Ollama daemon for every model you have pulled.
    # Useful for letting users pick a model or verifying a model is available
    # before trying to use it.
    print("\n\nLocal Models Available:")
    models = ollama.list()
    for m in models['models']:
        print(f"  - {m['model']}")

except Exception as e:
    print(f"\nError: {e}")
    # On macOS: Ollama runs as a menu bar app — it auto-starts at login.
    # You don't need `ollama serve`; just open the Ollama app once.
    # On Linux/Docker (no GUI): run `ollama serve` manually to start the daemon.
    print("Did you make sure Ollama is running?")
    print("  macOS: Open the Ollama app (it runs in the menu bar automatically)")
    print("  Linux: Run `ollama serve` in a terminal")
    print(f"Did you pull the model? (Run `ollama pull {MODEL_NAME}`)")
