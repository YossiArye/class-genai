"""
16. Ollama Raw API (Understanding the Protocol)
================================================

What this teaches:
    How to talk to Ollama with plain HTTP — no ollama library, no higher-level libraries.
    This reveals what every higher-level library does under the hood.

How to run:
    python 16_ollama_raw_api.py

Prerequisites:
    - Ollama running (macOS/Windows: open the app; Linux: `ollama serve`)
    - Model pulled: `ollama pull llama3.2`

Difference vs file 15 (15_ollama_cli.sh):
    15_ollama_cli.sh — sends HTTP requests using `curl` in a shell script.
        Who uses it: anyone testing an API quickly from the terminal.
        Language: Bash (shell commands).
        Output: raw JSON printed to the terminal, hard to process further.
        stream: false — waits for the complete response before printing.

    16_ollama_raw_api.py — sends HTTP requests using `requests` in Python.
        Who uses it: a Python developer who wants to understand the protocol,
                     or build something without installing the ollama library.
        Language: Python.
        Output: parsed and processed — you can extract fields, store results, etc.
        stream: True (default) — reads the response token-by-token as it arrives.

    Both talk to the SAME Ollama server on the SAME port (11434).
    The only difference is the tool sending the request: curl vs Python requests.

    Above both sits the `ollama` Python library (file 14) — it wraps exactly
    this kind of HTTP call so you don't have to write it yourself.

    Layer diagram (highest to lowest level):
        ollama library  (file 14) ← you just call ollama.chat()
        requests / curl (file 16/15) ← you write the HTTP call manually
        HTTP / TCP      ← the actual network protocol
        Ollama server   ← listens on localhost:11434
        The model       ← llama3.2 running on your hardware

Key Concept — Streaming JSON:
    Ollama (by default) streams the response token-by-token.
    Each token arrives as a separate JSON line:
        {"model":"llama3.2","response":"Quantum","done":false}
        {"model":"llama3.2","response":" entanglement","done":false}
        ...
        {"model":"llama3.2","response":".","done":true,"total_duration":1234567890}

    We read line-by-line, parse each JSON object, and print each token immediately.
    This is what creates the "typing" effect you see in ChatGPT.
"""

import requests
import json

# ── Configuration ────────────────────────────────────────────────────────────
URL   = "http://localhost:11434/api/generate"
MODEL = "llama3.2"

payload = {
    "model": MODEL,
    "prompt": "Explain Quantum Entanglement in one sentence.",
    # stream defaults to True in Ollama — omitting it means we GET a stream.
    # Set "stream": false to get one big JSON response (like file 15 does).
}

print(f"Connecting to {URL} with model {MODEL}...\n")

# ── 1. Make the HTTP Request ─────────────────────────────────────────────────
# stream=True tells the `requests` library: keep the connection open,
# don't wait for the full response — let us read it piece by piece.
response = requests.post(URL, json=payload, stream=True)

# ── 2. Process the Stream ────────────────────────────────────────────────────
if response.status_code == 200:
    print("--- Stream Started ---")

    for line in response.iter_lines():
        if not line:
            continue

        # Each line is a complete JSON object — parse it
        data = json.loads(line)

        # "response" holds the next token (one word or part of a word)
        chunk = data.get("response", "")

        # end="" prevents newline after each token; flush=True prints immediately
        # — without flush, Python would buffer and print in chunks, killing the streaming effect
        print(chunk, end="", flush=True)

        if data.get("done"):
            # "done": true arrives on the last line, which also contains timing stats
            # total_duration is in nanoseconds — divide by 1e9 to get seconds
            print("\n--- Stream Finished ---")
            duration = data.get("total_duration", 0) / 1e9
            print(f"\nStats: Total Duration: {duration:.2f}s")
            break

else:
    print(f"Error: {response.status_code} - {response.text}")
