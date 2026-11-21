"""
03. OpenAI Basic Usage
======================

This script demonstrates how to interact with the industry-standard OpenAI API (GPT models).
Most modern LLM applications use this same pattern (Client -> Chat Completion -> Response).

Prerequisites:
1.  An OpenAI API Key (OPENAI_API_KEY) in your .env file.
2.  `pip install openai python-dotenv`

Concepts Covered:
1.  **Environment Variables**: Securely loading API keys (never hardcode them in scripts!).
2.  **The Client**: Setting up the connection to OpenAI.
3.  **Chat Completions**: The main endpoint for sending messages and getting a response.
4.  **Roles**: 
    - `system`: Instructions for how the AI should behave (the "personality").
    - `user`: The actual input/question from the human.
"""

from openai import OpenAI
from dotenv import load_dotenv
import os

# 1. Load Secrets
# ---------------
# This looks for a .env file in the current directory and loads variables like OPENAI_API_KEY.
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Stop the script if no key is found to prevent confusing errors later
    raise ValueError("Missing OPENAI_API_KEY in .env file!")

# 2. Initialize the Client
# ------------------------
# The client handles the HTTP requests and authentication for us.
client = OpenAI(api_key=api_key)

print("Sending request to OpenAI...")

# 3. Create a Completion
# ----------------------
# We send a list of messages. The model reads them in order and predicts the next response.
response = client.chat.completions.create(
    model="gpt-4o-mini",  # A fast, cheaper model good for testing
    messages=[
        # SYSTEM: Sets the behavior. The user doesn't usually see this.
        {"role": "system", "content": "You are a poetic assistant. Answer everything in rhymes."},
        
        # USER: The actual prompt.
        {"role": "user", "content": "Why is the sky blue?"}
    ]
)

# 4. Extract the Answer
# ---------------------
# The response object contains technical metadata (tokens used, finish reason, etc.)
# The actual text is deep inside `choices[0].message.content`.
answer = response.choices[0].message.content

print("\n--- Model Response ---")
print(answer)
print("-" * 30)

# 5. Usage Metadata
# -----------------
# It's important to track how many tokens you used, as you pay per token.
usage = response.usage
print(f"\nUsage Stats:")
print(f"Prompt Tokens: {usage.prompt_tokens}")
print(f"Completion Tokens: {usage.completion_tokens}")
print(f"Total Tokens: {usage.total_tokens}")
