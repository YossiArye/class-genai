"""
02. Tokenization Explained
==========================

Computers cannot understand raw text like "Hello". They only understand numbers.
**Tokenization** is the process of breaking text into smaller chunks called "tokens" 
and converting them into numerical IDs.

Concepts Covered:
1.  **Tokenizer**: A tool trained alongside the model to split text into vocabulary items.
2.  **Encoding**: Text -> Numbers (Token IDs).
3.  **Decoding**: Numbers -> Text.
4.  **Sub-word Tokens**: Common words are single tokens; rare words are split into pieces.

Key Insight:
An LLM does not read 'words'. It reads these token IDs.
"""

from transformers import AutoTokenizer

# 1. Load the Tokenizer
# ---------------------
# We must use the EXACT same tokenizer that the model was trained with.
# If we used a BERT tokenizer for GPT-2, the numbers would mean completely different things.
print("Loading tokenizer for 'gpt2'...")
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')

# 2. Basic Encoding and Decoding
# ------------------------------
text = "The quick brown fox jumps."
print(f"\nOriginal Text: '{text}'")

# Encode: Convert string to list of integers
token_ids = tokenizer.encode(text)
print(f"Token IDs:     {token_ids}")

# Decode: Convert list of integers back to string
decoded_text = tokenizer.decode(token_ids)
print(f"Decoded Text:  '{decoded_text}'")

print("-" * 60)

# 3. Visualizing Tokens (Sub-word Splitting)
# ------------------------------------------
# Let's see how the tokenizer splits complex or made-up words.
# GPT-2 uses "Byte-Pair Encoding" (BPE). It assumes space is part of the next word.

examples = [
    "antigravity",       # A standard word
    "unconstitutionally", # A long, complex word
    "ingeniarius",       # A Latin/Foreign word (Engineer)
    "12345"              # Numbers
]

print("\nDetailed Token Breakdown:")
for ex in examples:
    # Encodes to IDs
    ids = tokenizer.encode(ex)
    
    # helper function "convert_ids_to_tokens" shows the actual string representation of each ID
    # In GPT-2, a special character Ġ usually represents a leading space.
    token_strings = tokenizer.convert_ids_to_tokens(ids)
    
    print(f"\nInput: '{ex}'")
    print(f"IDs:    {ids}")
    print(f"Chunks: {token_strings}")

print("-" * 60)
print("\nTakeaway: Notice how rare words are broken into smaller chunks.")
print("The model sees [394, 12034] instead of one word 'antigravity'.")
print("This allows the model to handle ANY text, even words it has never seen before!")
