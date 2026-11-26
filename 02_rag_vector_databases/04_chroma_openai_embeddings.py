"""
04. ChromaDB — OpenAI Embeddings
================================
What this lesson teaches:
- How to plug a different embedding model into ChromaDB (OpenAI text-embedding-3-small)
- That the collection embeds both your documents AND your queries with the same model
- What an OpenAI embedding looks like on disk: open Chroma's SQLite file and decode the vectors
- How big OpenAI vectors are compared to the default local model from lesson 03

Prerequisites: 03_chroma_persistent_default_embeddings.py, OPENAI_API_KEY in .env

How to run:
    python 02_rag_vector_databases/04_chroma_openai_embeddings.py

Expected output:
    The 2 documents closest to the query with their distances, then the decoded vectors
    (1536 numbers each) and a size comparison with lesson 03's collection if it exists.
"""

import os
import sqlite3                                        # to open Chroma's file directly (section 4)

import chromadb                                       # vector database
import numpy as np                                    # to turn raw bytes back into numbers
from chromadb.utils import embedding_functions        # built-in embedding functions
from dotenv import load_dotenv                        # reads OPENAI_API_KEY from .env

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"            # OpenAI's cheap, good default embedder
DB_PATH         = "./02_rag_vector_databases/db"      # same folder as lesson 03

# ── 1. Setup ─────────────────────────────────────────────────────────────────
#
# The only change from lesson 03: a different embedding_function. Chroma calls the OpenAI API
# for every document you add and every query you run — so this costs money (very little) and
# needs internet, but gives better vectors than the small local model.

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=DB_PATH)

# Start fresh each run so the demo (and the log in section 4) is reproducible
try:
    client.delete_collection("openai_embeddings")
except Exception:
    pass  # collection didn't exist yet — that's fine
collection = client.get_or_create_collection(
    name="openai_embeddings",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # cosine = best for text: measures meaning, not magnitude
)
# ⚠️ A collection is locked to its embedding model. Vectors from two models live in different
#    "spaces" — comparing them is meaningless. Change the model → re-embed everything.

# ── 2. Add documents (one batch call) ────────────────────────────────────────

print("=" * 55)
print("SECTION 2 — upsert() with OpenAI embeddings")
print("=" * 55)

# TIP: one upsert() with a list = one API call for all documents, not one call each
collection.upsert(
    documents=[
        "Python is a programming language.",
        "ChromaDB stores vectors.",
        "JavaScript runs in the browser.",
    ],
    ids=["doc1", "doc2", "doc3"],
)
print(f"  {collection.count()} documents embedded with {EMBEDDING_MODEL}")

# ── 3. Similarity search ─────────────────────────────────────────────────────
#
# The query text is embedded with the SAME model, then compared to the stored vectors.

print("\n" + "=" * 55)
print("SECTION 3 — Similarity search")
print("=" * 55)

query   = "Tell me about programming languages"
results = collection.query(query_texts=[query], n_results=2)

print(f"Query: '{query}'\n")
for doc_id, doc, distance in zip(results["ids"][0], results["documents"][0], results["distances"][0]):
    print(f"  [{doc_id}]  distance: {distance:.3f}")
    print(f"       {doc}")

# ── 4. Look inside the database — what an OpenAI embedding really is ────────
#
# Same idea as lesson 03, section 6: the vectors sit in the embeddings_queue table as raw bytes.
# Both lessons share one chroma.sqlite3 file, so we keep only THIS collection's rows: each row's
# "topic" column ends with the collection's id.
#
# ⚠️ These are Chroma's INTERNAL tables, not a public API — they can change between versions.
#    Fine for learning; in real code always read through collection.get().

print("\n" + "=" * 55)
print("SECTION 4 — Inside chroma.sqlite3")
print("=" * 55)

OPERATIONS = {0: "add", 1: "update", 2: "upsert", 3: "delete"}   # Chroma's operation codes

# TIP: "file:...?mode=ro" + uri=True opens the file READ-ONLY — we can't damage Chroma's data
conn = sqlite3.connect(f"file:{DB_PATH}/chroma.sqlite3?mode=ro", uri=True)
rows = conn.execute(
    "SELECT seq_id, id, operation, vector FROM embeddings_queue WHERE topic LIKE ? ORDER BY seq_id",
    (f"%{collection.id}",),                      # TIP: a ? parameter, never an f-string in SQL
)

# TIP: seq_id is global to the whole file and grows every run — we number our rows from 1
for row_num, (seq_id, doc_id, operation, blob) in enumerate(rows, start=1):
    vector     = np.frombuffer(blob, dtype=np.float32)   # 4 bytes per float32 number
    length     = np.linalg.norm(vector)                  # the vector's length
    first_five = np.round(vector[:5], 3)
    print(f"  #{row_num} {OPERATIONS[operation]:<6} {doc_id}  {len(blob)} bytes → "
          f"{len(vector)} numbers  {first_five} ...  length {length:.2f}")

# Size comparison across every collection in the file (lesson 03's "demo" too, if you ran it)
print(f"\n  {'Collection':<20} {'Numbers':>13}  {'Bytes per vector':>16}")
size_rows = conn.execute("""
    SELECT c.name, MAX(LENGTH(q.vector))
    FROM collections c JOIN embeddings_queue q ON q.topic LIKE '%' || c.id
    GROUP BY c.name
""")
for name, size in size_rows:
    numbers = size // 4                                   # float32 = 4 bytes
    print(f"  {name:<20} {numbers:>13}  {size:>16}")
conn.close()

print("""
  What to notice:
    • 6144 bytes = 1536 numbers × 4 bytes — 4× bigger than all-MiniLM-L6-v2 (384 numbers).
      1 million documents ≈ 6 GB of vectors alone. Model choice is also a storage decision.
    • Every length is 1.00 — OpenAI normalizes its vectors too, so cosine works well.
    • Same 3 sentences, different numbers: each model has its own "space". Never mix them.
""")

# Clean up (uncomment to remove the collection from disk)
# client.delete_collection("openai_embeddings")
