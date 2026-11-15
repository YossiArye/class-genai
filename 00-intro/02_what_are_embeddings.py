"""
02. What are Embeddings?
========================

Computers can't understand the string "Dog". 
To a computer, "Dog" is just bytes.

To understand **Meaning**, we convert words into **Vectors** (Lists of numbers).
This is called an **Embedding**.

Key Property:
Similar words have similar coordinate numbers. 
"Dog" should be closer to "Cat" than to "Car".

This script demonstrates "Word Algebra" with simplified fake embeddings.
"""

import numpy as np

# 1. Define Fake 2D Embeddings
# ----------------------------
# Imagine a 2D space where:
# X-axis = "Royalty" (Right is more royal)
# Y-axis = "Gender"  (Up is Female, Down is Male)

#                   [Royalty, Gender]
embeddings = {
    "King":  np.array([0.9,    -0.8]),
    "Queen": np.array([0.95,    0.9]),
    "Man":   np.array([0.1,    -0.8]),
    "Woman": np.array([0.1,     0.9]),
    "Apple": np.array([-0.5,    0.0]) # Not royal, gender neutral
}

print("--- Word Algebra Demo ---\n")

for word, vec in embeddings.items():
    print(f"Word: {word:6} | Vector: {vec}")

print("\n")

# 2. Similarity Search
# --------------------
# To find how similar "King" is to "Queen", we measure the physical distance between them.
# We use Euclidean Distance (Pythagoras Theorem): sqrt(x^2 + y^2).
#
# `np.linalg.norm(v)` calculates the length (magnitude) of a vector.
# `vec1 - vec2` gives us the difference vector (the path from one to the other).
# So, `norm(vec1 - vec2)` is the length of that path.
def get_distance(w1, w2):
    difference = embeddings[w1] - embeddings[w2]
    return np.linalg.norm(difference)

print(f"Distance between King and Queen: {get_distance('King', 'Queen'):.4f}")
print(f"Distance between King and Apple: {get_distance('King', 'Apple'):.4f}")
print("(Lower distance means more similar meanings!)\n")


# 3. The Magic of Analogies
# -------------------------
# The famous example: King - Man + Woman = ?
# Logic: Take "King", remove the "Maleness" (subtract Man), add "Femaleness" (add Woman).

print("Analogy: King - Man + Woman = ?")
result_vector = embeddings["King"] - embeddings["Man"] + embeddings["Woman"]

print(f"Result Vector: {result_vector}")

# Find which word is closest to this result
best_match = None
min_dist = float('inf')

for word, vec in embeddings.items():
    # Don't match the input words themselves to keep it fair
    if word in ["King", "Man", "Woman"]:
        continue
        
    dist = np.linalg.norm(vec - result_vector)
    if dist < min_dist:
        min_dist = dist
        best_match = word

print(f"Closest Match: {best_match}")
print(f"(Distance to Queen was {min_dist:.4f})")
