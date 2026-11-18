"""
01. The Math of Intelligence (Simplified)
=========================================

At its core, a Neural Network is just a **Weighted Sum**.
It multiplies inputs by their importance (weights) and adds them up.
This operation is called a **Dot Product**.

Analogy: Estimating a House Price
---------------------------------
How do you subconsciously estimate the price of a house? 
You look at features (Inputs) and multiply them by their value (Weights).

Features (The Inputs):
1.  **Number of Bedrooms** (Positive value)
2.  **Age of House in Years** (Negative value - older is cheaper)
3.  **Distance to City in km** (Negative value - closer is better)

The Neural Network "learns" the correct Weights to make accurate predictions.
"""

import numpy as np

print("--- Neural Network / Regression Demo ---\n")

# 1. Define Inputs (The House Features)
# -------------------------------------
# Let's say we are looking at a specific house:
# - 3 Bedrooms
# - 50 Years Old
# - 10 km from the city
inputs = np.array([3, 50, 10]) 

print(f"Inputs (House Features): {inputs}")
print("(Interpretation: 3 Rooms, 50 Years Old, 10km Distance)\n")


# 2. Define Weights (The Valuation Rules)
# ---------------------------------------
# These represent how much each feature impacts the result.
# - Bedroom Value: +100 (Each room adds $100k)
# - Age Penalty:   -2   (Lose $2k for every year old)
# - Distance Cost: -5   (Lose $5k for every km away)
weights = np.array([100, -2, -5])

print(f"Weights (Valuation Rules): {weights}")
print("(Interpretation: +100 per Room, -2 per Year, -5 per km)\n")


# 3. The Calculation (Dot Product)
# --------------------------------
# The model combines them:
# Score = (Rooms * RoomValue) + (Age * AgePenalty) + (Distance * DistanceCost)
# Score = (3 * 100)           + (50 * -2)          + (10 * -5)
# Score = 300                 - 100                - 50
# Score = 150

score = np.dot(inputs, weights)

print("Calculation:")
print(f"  ({inputs[0]} rooms * {weights[0]} value)")
print(f"+ ({inputs[1]} years * {weights[1]} penalty)")  # Fixed print logic in next lines
print(f"+ ({inputs[2]} km    * {weights[2]} cost)")
print("-" * 30)
print(f"Final Score (Estimated Value): {score}")


# 4. The Activation (The Decision)
# --------------------------------
# A neuron often converts this number into a binary decision (Yes/No).
# Let's say we only buy houses if the score is > 200.

threshold = 200
decision = "BUY" if score > threshold else "PASS"

comparison = ">" if score > threshold else "<="
print(f"\nResult: Score {score} is {comparison} {threshold}, so Decision is: {decision}")
