"""
03. ChromaDB — Persistent Storage & add() vs upsert()
======================================================
What this lesson teaches:
- How to save a ChromaDB collection to disk (PersistentClient)
- The real difference between add() and upsert() — demonstrated live
- How to peek inside the collection to verify what actually changed
- Why upsert() is the safe default for most production use cases
- What an embedding really is on disk: open Chroma's SQLite file and decode the vectors

Prerequisites: 01_chroma_intro_in_memory.py

How to run:
    python 03_chroma_persistent_default_embeddings.py

Expected output:
    Live before/after snapshots showing what each operation does to the DB,
    including a warning that add() silently ignores an existing ID (ChromaDB 1.x).
"""

import sqlite3                                        # to open Chroma's file directly (section 6)

import chromadb                                       # vector database
import numpy as np                                    # to turn raw bytes back into numbers
from chromadb.utils import embedding_functions        # built-in embedding models

DB_PATH = "./02_rag_vector_databases/db"              # Chroma keeps everything in this folder

# ── 1. Setup ─────────────────────────────────────────────────────────────────
#
# PersistentClient saves the collection to disk in the given path.
# Every run reads from / writes to the same SQLite file there.
# Use chromadb.Client() (no args) for a temporary in-memory collection instead.

default_ef = embedding_functions.DefaultEmbeddingFunction()  # uses all-MiniLM-L6-v2 locally

client = chromadb.PersistentClient(path=DB_PATH)

# Start fresh each run so the demo is reproducible
try:
    client.delete_collection("demo")  # delete if it exists from a previous run
except Exception:
    pass  # collection didn't exist yet — that's fine
collection = client.get_or_create_collection(
    name="demo",
    embedding_function=default_ef,
    metadata={"hnsw:space": "cosine"}  # cosine = best for text
)


def show_collection(label):
    """Print every document currently in the collection."""
    data = collection.get(include=["documents"])
    print(f"\n── {label} ──")
    if not data["ids"]:
        print("  (empty)")
    for doc_id, doc in zip(data["ids"], data["documents"]):
        print(f"  [{doc_id}]  {doc}")


# ── 2. add() — insert only, skips duplicates ─────────────────────────────────
#
# add() never overwrites: if the ID already exists, the new document is dropped.
# In ChromaDB 1.x this happens WITHOUT an error — easy to miss.

print("=" * 55)
print("SECTION 2 — add() behaviour")
print("=" * 55)

collection.add(documents=["Python is a programming language."], ids=["doc1"])
collection.add(documents=["ChromaDB stores vectors."],          ids=["doc2"])
show_collection("After first add() — 2 documents inserted")

# Now try to add doc1 again with different content
print("\nTrying to add doc1 again with add()...")
# ⚠️ ChromaDB 1.x does NOT raise here: add() skips an existing ID silently and keeps the old
#    document. (Old 0.3.x versions raised DuplicateIDError.) This is the dangerous part —
#    your "new" data is thrown away and nothing tells you. So we check it ourselves.
before_doc = collection.get(ids=["doc1"])["documents"][0]
try:
    collection.add(documents=["This is a NEW version of doc1."], ids=["doc1"])
    after_doc = collection.get(ids=["doc1"])["documents"][0]
    if after_doc == before_doc:
        print("  ⚠️  No error — but add() IGNORED the new text. doc1 still holds the old version.")
except Exception as e:
    # TIP: older ChromaDB versions raise here instead — catch the base Exception so the
    #      lesson works on any version
    print(f"  💥 Error: {type(e).__name__}: {e}")

show_collection("After add() with existing ID — collection unchanged")

# ── 3. upsert() — insert OR update, never errors ─────────────────────────────
#
# upsert() = "update if exists, insert if not".
# It silently replaces the document when the ID already exists.
# This is the safe default for pipelines where data may change over time.

print("\n" + "=" * 55)
print("SECTION 3 — upsert() behaviour")
print("=" * 55)

# Add a new document (same as add — inserts when ID is new)
collection.upsert(documents=["JavaScript runs in the browser."], ids=["doc3"])
show_collection("After upsert() with new ID — doc3 inserted")

# Now upsert doc1 — it exists, so it gets updated silently
collection.upsert(documents=["This is a NEW version of doc1 via upsert."], ids=["doc1"])
show_collection("After upsert() with existing ID — doc1 updated, no error")

# ── 4. Similarity Search ─────────────────────────────────────────────────────
#
# This is the core of ChromaDB: given a query in plain text, find the most
# semantically similar documents — even if they share no exact words.
#
# Under the hood:
#   1. Your query text is embedded into a vector using the same model.
#   2. ChromaDB computes cosine distance between the query vector and every stored vector.
#   3. It returns the n_results closest ones (lower distance = more similar).

print("\n" + "=" * 55)
print("SECTION 4 — Similarity Search")
print("=" * 55)

query = "Tell me about programming languages"
results = collection.query(
    query_texts=[query],  # ChromaDB embeds this automatically using the collection's model
    n_results=3           # return the 3 most similar documents
)

print(f"Query: '{query}'\n")
for i, (doc_id, doc, distance) in enumerate(zip(
    results["ids"][0],
    results["documents"][0],
    results["distances"][0]
), start=1):
    # TIP: results["ids"], results["documents"], results["distances"] are all
    #      lists-of-lists — one inner list per query. [0] gets the first (and only) query's results.
    print(f"  #{i}  [{doc_id}]  distance: {distance:.3f}")
    print(f"       {doc}")

print()
print("Notice: 'Python is a programming language' scores lowest distance (most similar).")
print("'ChromaDB stores vectors' is less related — higher distance score.")

# ── 5. Side-by-side summary ───────────────────────────────────────────────────

print("\n" + "=" * 55)
print("SECTION 5 — Summary")
print("=" * 55)
print("""
  Operation  │ ID is new       │ ID already exists
  ───────────┼─────────────────┼──────────────────────────
  add()      │ ✅ inserts      │ ⚠️ skips silently, keeps the OLD document
  upsert()   │ ✅ inserts      │ ✅ replaces with the NEW document

  Rule of thumb:
    Use add()    → only for data that never changes (and check for existing IDs yourself)
    Use upsert() → when data changes over time and re-runs should refresh it (most cases)
""")

# ── 6. Look inside the database — what an embedding really is ────────────────
#
# "Persistent" just means: one SQLite file on disk (chroma.sqlite3) + one folder per collection
# holding the HNSW search index. We open the SQLite file ourselves, read-only, and decode the
# vectors Chroma saved. An embedding turns out to be nothing magic: a list of float numbers
# stored as raw bytes.
#
# Where things live:
#   embedding_metadata  → the text (key 'chroma:document') and your metadata
#   embeddings_queue    → the vectors, as a LOG of every operation (add, upsert...)
#   <uuid>/data_level0.bin → the HNSW index that query() actually searches (fast, not SQL)
#
# ⚠️ These are Chroma's INTERNAL tables, not a public API — they can change between versions.
#    Fine for learning; in real code always read through collection.get().

print("\n" + "=" * 55)
print("SECTION 6 — Inside chroma.sqlite3")
print("=" * 55)

OPERATIONS = {0: "add", 1: "update", 2: "upsert", 3: "delete"}   # Chroma's operation codes

# TIP: "file:...?mode=ro" + uri=True opens the file READ-ONLY — we can't damage Chroma's data
conn = sqlite3.connect(f"file:{DB_PATH}/chroma.sqlite3?mode=ro", uri=True)
# Lesson 04 writes to the same file, so keep only THIS collection's rows:
# each row's "topic" column ends with the collection's id.
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
conn.close()

print("""
  What to notice:
    • 1536 bytes = 384 numbers × 4 bytes. 384 is the size of all-MiniLM-L6-v2.
      (OpenAI's text-embedding-3-small would give 1536 numbers = 6 KB per document.)
    • Every length is 1.00 — the model normalizes its vectors, which is why cosine works well.
    • 5 rows for 3 documents: it is a log. Row #3 is the ignored add() of doc1 —
      Chroma even computed its embedding, and still kept the old document.
""")
