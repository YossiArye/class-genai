"""
06. Roles: System vs User
=========================

The `system` message is one of the most powerful tools in an engineer's kit.
It sets the **global behavior**, tone, and constraints of the AI.
In many ways, the System Prompt *is* the application code.

Roles:
- **System**: The invisible boss. Sets the rules.
- **User**: The end-user interacting with the app.

Example: Building a "Sarcastic Tech Support Bot".
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def chat_with_persona(system_instruction, user_input):
    print(f"\n--- Persona: {system_instruction[:30]}... ---")
    print(f"User: {user_input}")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            # The System Prompt is usually invisible to the user
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_input}
        ]
    )
    print(f"AI: {response.choices[0].message.content}")


# Scenario 1: The Helpful Tutor
# -----------------------------
chat_with_persona(
    system_instruction="You are a patient elementary school math teacher. Explain things simply and encourage the student.",
    user_input="I don't understand fractions."
)

# Scenario 2: The Sarcastic IT Guy
# --------------------------------
chat_with_persona(
    system_instruction="You are a cynical IT support specialist who hates answering tickets. Be sarcastic and unhelpful.",
    user_input="My mouse isn't working."
)

# Scenario 3: The Security Guard (Output Control)
# ----------------------------------------------
# System prompts are also used for security constraints (Jailbreak protection basics).
chat_with_persona(
    system_instruction="You are a dataset assistant. You ONLY output JSON. You NEVER output regular text. If asked for opinion, output null.",
    user_input="What is the meaning of life?"
)
