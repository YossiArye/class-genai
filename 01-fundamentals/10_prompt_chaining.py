"""
10. Prompt Chaining
===================

Don't try to do everything in one prompt.
**Prompt Chaining** is breaking a complex task into a sequence of smaller, simpler prompts,
where the output of one becomes the input of the next.

Benefit:
- Higher accuracy (each step focuses on one thing).
- Easier debugging (you can see where it failed).
- Lower context usage (each step only sees what it needs).

Task: Write a targeted marketing email for a product based on a user review.
Process:
1.  Analyze Sentiment (Positive/Negative).
2.  Identify Key Features mentioned.
3.  Write Email based on features.
"""

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

# Input Data
user_review = "I've been using the SuperVacuum 3000. It's incredibly quiet, which I love because of my sleeping baby. But the battery life is kinda short."

print(f"Review: {user_review}\n")

# Step 1: Extract Features
# ------------------------
print("Step 1: Extracting positive features...")
response_1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract the POSITIVE features mentioned in the review. Return a comma-separated list."},
        {"role": "user", "content": user_review}
    ]
)
features = response_1.choices[0].message.content
print(f"Features: {features}\n")

# Step 2: Write Email
# -------------------
print("Step 2: Writing email...")
# We feed 'features' (output of step 1) into step 2.
response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a marketing expert. Write a short 3-line email thanking the user and highlighting the features they liked."},
        {"role": "user", "content": f"Verified Customer Review Features: {features}"}
    ]
)
email = response_2.choices[0].message.content

print("--- Final Result ---")
print(email)
