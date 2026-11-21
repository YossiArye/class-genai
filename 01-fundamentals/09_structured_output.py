"""
09. Structured Output (JSON)
============================

In software engineering, we rarely want a poem or an essay. We want **Data**.
Getting an LLM to output valid JSON is critical for connecting it to other code (APIs, Databases).

Methods:
1.  **Prompt Engineering**: Asking nicely "Please return JSON". (Prone to errors).
2.  **Response Format**: OpenAI's native `json_object` mode (Guaranteed validity).
3.  **Structured Output (Pydantic)**: Strict schema enforcement (Advanced).

We will demonstrate the native `response_format={"type": "json_object"}`.
"""

from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# Task: Extract movie info from a review.
text = """
I watched 'Inception' last night. It was directed by Christopher Nolan and released in 2010.
The visuals were mind-bending, though the plot was a bit confusing. I give it 9/10.
"""

print(f"Input Text:\n{text}\n")

# 1. JSON Mode
# ------------
# Two rules for JSON mode:
# 1. You MUST set response_format={ "type": "json_object" }
# 2. You MUST explicitly mention the word "JSON" in the system or user prompt.
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system", 
            "content": "You are a data extractor. Output JSON with keys: title, director, year, rating, sentiment."
        },
        {"role": "user", "content": text}
    ],
    response_format={"type": "json_object"},
)

json_string = response.choices[0].message.content
print(f"Raw JSON String:\n{json_string}\n")

# 2. Parsing
# ----------
# Because we are guaranteed valid JSON, we can parse it safely.
try:
    data = json.loads(json_string)
    print("Parsed Object:")
    print(f"Movie: {data.get('title')}")
    print(f"Rating: {data.get('rating')}")
except json.JSONDecodeError:
    print("Error decoding JSON")

# Note: For production, libraries like Pydantic or Instructor are even better/stricter.
