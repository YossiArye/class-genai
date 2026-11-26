"""
02. ChromaDB — Metadata Filtering (the `where` clause)
======================================================
What this lesson teaches:
- Attaching metadata to documents at upsert time (`metadatas=[...]`)
- WHY similarity alone is not enough: embeddings capture MEANING, but they are
  blind to facts like year, author, or category — "recent AI advances" will happily
  return a 2018 document, because "recent" is not in the vector
- The `where` clause: the DEVELOPER writes a metadata filter by hand, and Chroma
  restricts the similarity search to documents that pass it
- Filter operators: equality, $gte/$lte, $in, and combining with $and
- where_document: filtering on the TEXT itself ($contains) — no similarity involved

Prerequisites: 01_chroma_intro_in_memory.py

How to run:
    python 02_chroma_metadata_filtering.py

Expected output:
    The same query answered with and without filters — showing how each `where`
    clause shrinks the candidate set before similarity ranking happens.
"""

import chromadb  # vector database for storing and searching embeddings

# ── 1. Create client and collection ───────────────────────────────────────────
client = chromadb.Client()
collection = client.get_or_create_collection(
    name="articles",
    metadata={"hnsw:space": "cosine"}  # cosine = best for text: measures meaning, not magnitude
)

# ── 2. Add documents WITH metadata ────────────────────────────────────────────
#
# metadatas must align with documents by position: metadata #2 belongs to document #2.
# Values must be scalars (str / int / float / bool) — no lists or nested dicts.
collection.upsert(
    documents=[
        "Machine learning models improve with more training data.",
        "Deep learning uses neural networks with many stacked layers.",
        "JavaScript frameworks like React dominate front-end development.",
        "Transformer models changed natural language processing forever.",
        "CSS grid makes complex web page layouts simple.",
    ],
    metadatas=[
        {"topic": "ai",  "year": 2018},
        {"topic": "ai",  "year": 2022},
        {"topic": "web", "year": 2021},
        {"topic": "ai",  "year": 2023},
        {"topic": "web", "year": 2019},
    ],
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
)

# Small display helper so each demo below stays one screen tall.
def show(results):
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]   # query() returns metadata alongside each document
        # TIP: :.3f formats the distance to 3 decimals — enough to compare, easy to read.
        print(f"  {results['ids'][0][i]}  (topic={meta['topic']}, year={meta['year']})  distance={results['distances'][0][i]:.3f}")
        print(f"    {doc}")
    print()

question = "recent advances in artificial intelligence"

# ── 3. The problem: similarity search is blind to metadata ────────────────────
#
# "recent" is a FACT about the year — but the embedding only encodes the MEANING
# of the words. Watch the 2018 document rank right next to the 2023 one:
print("🔍 NO FILTER — similarity only")
print("=" * 60)
results = collection.query(query_texts=[question], n_results=3)
show(results)

# ── 4. The fix: the developer writes a `where` filter ─────────────────────────
#
# `where` runs BEFORE similarity ranking: Chroma first keeps only the documents
# whose metadata passes the filter, then ranks those by distance to the query.
print("🔍 where={'year': {'$gte': 2022}} — only recent documents")
print("=" * 60)
results = collection.query(
    query_texts=[question],
    n_results=3,
    where={"year": {"$gte": 2022}},   # $gte = greater-or-equal; also $gt, $lt, $lte, $ne
)
show(results)

print("🔍 where={'topic': 'ai'} — only AI documents")
print("=" * 60)
results = collection.query(
    query_texts=[question],
    n_results=3,
    where={"topic": "ai"},            # plain value = equality (shorthand for {"$eq": "ai"})
)
show(results)

# Combining conditions: $and / $or take a LIST of filters.
print("🔍 where=$and — AI documents from 2022 onwards")
print("=" * 60)
results = collection.query(
    query_texts=[question],
    n_results=3,
    where={"$and": [
        {"topic": "ai"},
        {"year": {"$gte": 2022}},
    ]},
)
show(results)

# ── 5. Operator cheat sheet ───────────────────────────────────────────────────
#
#   {"topic": "ai"}                       equality (shorthand for {"$eq": "ai"})
#   {"year": {"$ne": 2018}}               not equal
#   {"year": {"$gt": 2020}}               greater than          (also $gte, $lt, $lte)
#   {"topic": {"$in": ["ai", "web"]}}     value is in the list  (also $nin = not in)
#   {"$and": [filter1, filter2]}          both must hold        (also $or)

# ── 6. Bonus: filter on the TEXT itself with where_document ───────────────────
#
# get() fetches by filter ALONE — no query text, no similarity, no ranking.
# $contains does a substring match on the document text.
print("🔍 get(where_document={'$contains': 'neural'}) — plain text match, no similarity")
print("=" * 60)
hits = collection.get(where_document={"$contains": "neural"})
for doc_id, doc in zip(hits["ids"], hits["documents"]):   # get() returns flat lists, not lists-of-lists
    print(f"  {doc_id}: {doc}")
print()

print("✅ `where` filters run BEFORE ranking — they shrink the candidate set")
print("✅ Embeddings capture meaning; metadata carries the facts (year, topic, author)")
print("💡 Here YOU wrote every filter by hand. In module 04 (self-query) an LLM")
print("   reads the user's question and writes these `where` filters automatically.")
