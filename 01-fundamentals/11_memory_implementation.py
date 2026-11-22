"""
11. Memory Implementation
=========================

LLMs are **Stateless**. They don't remember you.
Every time you send a request, the model sees it as a brand new conversation.

"Memory" is just an illusion created by the developer.
We simply send the *entire conversation history* back to the model with every new request.

This script builds a simple Chatbot Loop that remembers what you said.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# 1. Initialize History
# ---------------------
# The "memory" is just a Python list variable.
history = [
    {"role": "system", "content": "You are a helpful assistant named Jarvis."}
]

print("Chat with Jarvis (type 'quit' to exit)\n")

while True:
    # 2. Get User Input
    user_input = input("You: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    # 3. Add to History
    # -----------------
    history.append({"role": "user", "content": user_input})

    # 4. Send the ENTIRE History
    # --------------------------
    # The model reads the whole list, sees the context, and generates the next logical response.
    # Note: If this list gets too long, you run out of tokens (Context Window Limit).
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=history
    )
    
    bot_reply = response.choices[0].message.content
    print(f"Jarvis: {bot_reply}")

    # 5. Add Reply to History
    # -----------------------
    # We must also save what the bot said, so it remembers its own answers next time.
    history.append({"role": "assistant", "content": bot_reply})

print("\nHistory Dump:")
for msg in history:
    print(msg)


# Exmple:
# You: i love Timber
# Jarvis: That's great! Timber has a lot of wonderful qualities, whether you’re referring to timber as a material used in construction and woodworking or perhaps you’re talking about Timberland, the footwear and apparel brand. Could you clarify what you love about timber?
# You: Who i love?
# Jarvis: I can't know who you love unless you share more information with me. If you'd like to talk about your feelings or the person you love, I'm here to listen and help!
# You: Timber is my Dog
# Jarvis: That's wonderful! Dogs are such great companions. What kind of dog is Timber, and what do you love most about him?
# You: Who I love?
# Jarvis: It sounds like you love your dog, Timber! Dogs often hold a special place in our hearts. If you’d like to share more about your relationship with Timber or anything else, feel free to do so!