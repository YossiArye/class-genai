#!/bin/bash

# 15. Ollama via Command Line (CLI / cURL)
# =========================================
#
# Before writing Python code, it helps to understand that Ollama acts as a web server.
# You can talk to it using ANY language that speaks HTTP (JavaScript, Go, Rust, etc.).
# The universal way to test APIs is `curl`.
#
# How to run this file:
#   Option A — execute directly:
#       bash 15_ollama_cli.sh
#
#   Option B — make it executable first, then run as a program:
#       chmod +x 15_ollama_cli.sh   ← only needed once
#       ./15_ollama_cli.sh
#
#   Both do the same thing. Option B is the Unix convention for scripts you run often.
#
# Prerequisites:
# 1. Install Ollama: https://ollama.com/
# 2. Start Ollama:
#      macOS/Windows — open the Ollama app (cat icon in menu bar / system tray)
#      Linux         — run `ollama serve` in a separate terminal
# 3. Pull a model: `ollama pull llama3.2`

echo "--- 1. Simple Generation ---"
echo "Sending 'Why is the sky blue?' to Llama 3..."
# We send a POST request with a JSON body containing 'model' and 'prompt'.
# 'stream': false tells it to wait for the whole answer (easier to read in terminal).
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Why is the sky blue?",
  "stream": false
}'

echo -e "\n\n--- 2. Chat Mode (With Roles) ---"
# The /api/chat endpoint allows setting System and User roles.
curl http://localhost:11434/api/chat -d '{
  "model": "llama3.2",
  "messages": [
    { "role": "system", "content": "You are a pirate." },
    { "role": "user", "content": "Hello!" }
  ],
  "stream": false
}'

echo -e "\n\n--- 3. JSON Output Mode ---"
# Useful for scripting. We force the output to be JSON.
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Give me a random user with name and age. Output JSON only.",
  "format": "json",
  "stream": false
}'

echo -e "\n\n--- 4. Streaming Mode ---"
# With "stream": true (the default), Ollama sends the response token-by-token.
# Each token arrives as a separate JSON line:
#   {"model":"llama3.2","response":"Quantum","done":false}
#   {"model":"llama3.2","response":" entanglement","done":false}
#   ...
#   {"model":"llama3.2","response":".","done":true}
#
# In the terminal this looks like the model is "typing" in real time.
# This is exactly the same streaming effect you see in ChatGPT.
#
# Note: omitting "stream" altogether also gives you streaming — true is the default.
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.2",
  "prompt": "Count from 1 to 5, one number per line.",
  "stream": true
}'

echo -e "\n\nDone!"
