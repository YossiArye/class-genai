"""
04. Streaming Responses
=======================

In the previous examples, we waited for the ENTIRE answer to be generated before printing it.
For long answers, this feels slow (latency).
**Streaming** allows us to print each chunk (token) as soon as it is generated, 
giving the user immediate feedback.

Concepts Covered:
1.  **Iterators**: Looping over a connection that stays open.
2.  **Chunks**: Receiving partial text pieces (deltas) instead of a full message.
3.  **UX Improvement**: Why streaming makes apps feel faster.

We will use OpenAI for this example, but Ollama supports streaming too!
Lesson 14 shows it with the ollama library — the pattern is identical.
"""

from openai import OpenAI
from dotenv import load_dotenv
import sys

load_dotenv()
client = OpenAI()

print("Streaming a story about a space cat...\n")

# 1. Enable Streaming
# -------------------
# set stream=True in the API call.
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Write a short 3-sentence story about a cat who becomes a starship captain."}
    ],
    stream=True,  # <--- MAGIC SWITCH
)

print("--- START STREAM ---")

# 2. Iterate Over the Stream
# --------------------------
# The 'stream' object is an iterable. We loop through it.
# Each 'chunk' is a tiny piece of the response.
for chunk in stream:
    # Extract the partial content text
    # It might be None for the very first or last chunk (metadata)
    content = chunk.choices[0].delta.content
    
    if content:
        # Print immediately to the console without a newline, so it flows like typing.
        # flush=True ensures it appears instantly even in buffered terminals.
        print(content, end="", flush=True)

print("\n--- END STREAM ---")
