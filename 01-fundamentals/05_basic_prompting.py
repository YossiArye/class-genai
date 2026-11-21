"""
05. Basic Prompting Principles
==============================

"Garbage in, Garbage out."
The quality of the AI's output depends heavily on how you ask the question.
This script demonstrates 4 fundamental principles of good prompting.

Principles:
1.  **Clarity**: Be unambiguous.
2.  **Specificity**: Define the desired format and length.
3.  **Context**: Give background info (The "Why" and "Who").
4.  **Direct Instruction**: Tell it exactly what to do.

We use the helper function `get_completion` to keep the code clean.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_completion(prompt):
    """Helper to get a simple response"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# 1. The "Bad" Prompt
# -------------------
# Vague, no constraints.
bad_prompt = "Tell me about code."
print(f"--- BAD PROMPT ---\nInput: {bad_prompt}")
print(f"Output:\n{get_completion(bad_prompt)[:200]}...\n(Response truncated because it's too generic)\n")

# 2. The "Good" Prompt (Applied Principles)
# -----------------------------------------
# Specific, Contextual, Clear.
good_prompt = """
I am a beginner Python student.
Explain what a 'Variable' is.
Use a real-world analogy (like a box).
Keep the explanation under 3 sentences.
"""

print(f"--- GOOD PROMPT ---\nInput: {good_prompt}")
print(f"Output:\n{get_completion(good_prompt)}\n")

# 3. Instruction Formatting
# -------------------------
# Using delimiters (like quotes or dashes) helps the model separate your instructions
# from the text you want it to process.
text_to_summarize = """
The mitochondria is a double-membrane-bound organelle found in most eukaryotic organisms. 
Some cells in some multicellular organisms may lack mitochondria...
"""

delimiter_prompt = f"""
Summarize the text delimited by triple quotes into a single word.

'''{text_to_summarize}'''
"""

print(f"--- DELIMITERS ---\nOutput: {get_completion(delimiter_prompt)}")
