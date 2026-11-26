"""
01. ChromaDB — In-Memory Intro
==============================
What this lesson teaches:
- What ChromaDB is and why vector databases matter for AI
- How ChromaDB automatically converts text to vectors (embeddings) under the hood
- How to query by similarity: asking "what is most like this?" instead of exact match
- Reading results: understanding `documents`, `ids`, and `distances`
- Batch querying: multiple queries in ONE call, and reading the list-of-lists reply

Prerequisites: none — this is the entry point for the RAG module

How to run:
    python 01_chroma_intro_in_memory.py

Expected output:
    Single-query results with IDs and distance scores (lower = more similar),
    then two queries answered side by side from one batched call.
"""

import chromadb  # vector database for storing and searching embeddings

# ── 1. Create client and collection ───────────────────────────────────────────
# metadata={"hnsw:space": "cosine"} sets the distance metric used for similarity search.
# We use cosine here because this collection stores text — cosine measures meaning, not magnitude.
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="my_collection",
    metadata={"hnsw:space": "cosine"}
)

# ── 2. Add documents (batch operation) ────────────────────────────────────────
# MECHANISM EXPLAINED:
# 1. ChromaDB receives the raw text ("This is a sample...").
# 2. It automatically uses a default embedding model (all-MiniLM-L6-v2) to convert text -> Vector (list of numbers).
# 3. It stores:
#    - ID: "doc1"
#    - Document: The raw text (so we can read it later)
#    - Vector: The mathematical representation (hidden, for search)
collection.upsert(
    documents=[
        "This is a sample document about JS.",
        "This document discusses machine learning.",
        "Natural language processing is a fascinating field."
    ],
    ids=["doc1", "doc2", "doc3"]
)

# ── 3. Query by similarity ────────────────────────────────────────────────────
# MECHANISM EXPLAINED:
# 1. Chroma converts your query ("Tell me about AI...") into a Vector using the SAME model.
# 2. It compares this query vector against all stored vectors using a distance metric.
#    In all cases: lower score = more similar, 0 = identical.
#
#    "l2"  (Euclidean) — Ruler distance between two points in space.
#                        Score 0 = same point. Grows as vectors move apart.
#                        Like GPS: Tel Aviv and Haifa are far, so big distance.
#                        Use when: raw size matters (pixels, sensor values).
#
#    "cosine"          — Compass direction. Measures the angle, ignores length.
#                        Score 0 = same direction (identical meaning). Score 2 = opposite.
#                        "I love pizza" and "Pizza is amazing" point the same way
#                        even if one is longer. Length is irrelevant — only direction matters.
#                        Use when: meaning matters → ALL text and NLP tasks. ✅
#
#    "ip" (Inner Product) — Popularity vote. Angle × length of BOTH vectors.
#                        Longer (higher-magnitude) vectors score better even if less aligned.
#                        Like search results that mix relevance + how many times something
#                        was clicked. Popular items naturally float up.
#                        Use when: relevance + popularity blended (recommendation systems).
#                        ⚠️ Not for text search — use cosine.
#
# 3. It returns the 'k' nearest neighbors (n_results=2).
results = collection.query(
    query_texts=["Tell me about AI and machine learning."],
    n_results=2
)

# Results structure:
# {
#     'ids': [['doc2', 'doc3']],
#     'documents': [['This document discusses machine learning.', 'Natural language processing...']],
#     'distances': [[0.123, 0.456]],
#     'metadatas': [[None, None]]
# }

# Understanding ChromaDB cosine distance scores:
#
# Range: 0 to 2  (NOT 0 to 1 — a common misconception)
#   0.0       = identical meaning (same sentence)
#   2.0       = completely opposite meaning (extremely rare in practice)
#
# What a score tells you:
#   0.0 – 0.3  → very similar, near-duplicate
#   0.3 – 0.7  → related topics, meaningful match
#   0.7 – 1.0  → loosely related
#   1.0+       → likely unrelated
#
# ⚠️ Don't judge by the absolute number alone — always look at the relative ranking.
#    A result with distance 0.4 is good if the next result is 0.9.
#    The gap between results matters more than the raw score.

print("── SINGLE QUERY ────────────────────────────────────────────")
for i, doc in enumerate(results['documents'][0]):
    print(f"ID: {results['ids'][0][i]}")
    print(f"Content: {doc}")
    print(f"Distance: {results['distances'][0][i]}\n")

# ── 4. Multiple queries in ONE call ───────────────────────────────────────────
#
# query_texts accepts a LIST — Chroma embeds and searches all of them in one call.
# Why batch instead of a loop of single queries?
#   - one network/API round-trip instead of N
#   - the query texts are embedded together (one model invocation, not N)
# Same idea as batch embedding in lesson 07 — batching is the default habit to build.
queries = [
    "Tell me about AI and machine learning.",
    "What is JavaScript?"
]

multi_results = collection.query(
    query_texts=queries,   # the whole list goes in at once
    n_results=2
)

# The reply is a LIST OF LISTS — one inner list per query, in the same order:
# {
#     'ids':       [['doc2', 'doc3'],   # ← results for queries[0]
#                   ['doc1', 'doc2']],  # ← results for queries[1]
#     'documents': [[...], [...]],
#     'distances': [[...], [...]]
# }
# So results['documents'][0] belongs to queries[0], [1] to queries[1], etc.

print("── BATCHED QUERIES ─────────────────────────────────────────")
for query_idx, query in enumerate(queries):
    print(f"{'='*60}")
    print(f"QUERY {query_idx + 1}: {query}")
    print(f"{'='*60}")

    for i, doc in enumerate(multi_results['documents'][query_idx]):
        print(f"  Result {i + 1}:")
        print(f"  ID: {multi_results['ids'][query_idx][i]}")
        print(f"  Content: {doc}")
        print(f"  Distance: {multi_results['distances'][query_idx][i]}")
