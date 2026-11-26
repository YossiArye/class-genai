"""
06. ChromaDB — Manual OpenAI Embeddings
=======================================
What this lesson teaches:
- How to create embeddings yourself with the OpenAI API (client.embeddings.create)
- How to embed many chunks in ONE API call (a batch) and match each vector to its chunk
- How to give Chroma ready-made vectors with upsert(embeddings=...) — Chroma then skips embedding
- How to search with your own query vector: query(query_embeddings=...)

Prerequisites: 05_chroma_load_pdf_files.py, OPENAI_API_KEY in .env,
               PDF files in 02_rag_vector_databases/pdfs/

How to run:
    python 02_rag_vector_databases/06_manual_openai_embeddings.py

Expected output:
    How many chunks were loaded per PDF, how many API calls embedded them, then the 2 chunks
    closest to "The major ideologies" with source file, page and distance (best ≈ 0.45).
"""

import os

import chromadb                                       # vector database
from dotenv import load_dotenv                        # reads OPENAI_API_KEY from .env
from openai import OpenAI                             # we call the embeddings API ourselves
from pypdf import PdfReader                           # extracts text from PDF pages

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
DB_PATH         = "./02_rag_vector_databases/db"
PDF_DIR         = "02_rag_vector_databases/pdfs"
BATCH_SIZE      = 500    # chunks per API call — well under OpenAI's limit of 2,048 inputs per call

client = OpenAI()

# ── 1. Why embed manually? ───────────────────────────────────────────────────
#
# In lessons 04–05 Chroma called OpenAI for us (embedding_function=...). Here WE create the
# vectors and hand them to Chroma with embeddings=..., so Chroma skips the embedding step.
#
# What you gain:
#   1. Caching: save the vectors to disk and never pay for the same text twice  → lesson 07.
#   2. Retry & resume: if a big job fails at batch 7 of 20, retry batch 7 only.
#   3. Cost control: you see exactly what text (and how many tokens) you send.
#   4. Reuse: the same vectors can go into Chroma, FAISS (lessons 09–10) or anywhere else.

# No embedding_function here: this collection only receives ready-made vectors.
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Start fresh each run — upsert() never deletes, so chunks from old PDFs would stay forever
try:
    chroma_client.delete_collection("pdf_manual_embeddings")
except Exception:
    pass  # collection didn't exist yet — that's fine
collection = chroma_client.get_or_create_collection(
    name="pdf_manual_embeddings",
    metadata={"hnsw:space": "cosine"},  # cosine = best for text: measures meaning, not magnitude
)


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks (same as lesson 05)."""
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts in ONE API call. Returns one vector per text, in the same order."""
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    vectors  = [None] * len(texts)
    for item in response.data:
        # item.index = the position of this text in the list we sent → vector i belongs to texts[i]
        vectors[item.index] = item.embedding
    return vectors


# ── 2. Load the PDFs into chunks ─────────────────────────────────────────────

print("=" * 55)
print("SECTION 2 — Loading PDFs")
print("=" * 55)

all_docs, all_ids, all_metadata = [], [], []

# TIP: sorted() — os.listdir() returns files in random order, which would change the IDs per run
for filename in sorted(os.listdir(PDF_DIR)):
    if not filename.endswith(".pdf"):
        continue
    reader      = PdfReader(os.path.join(PDF_DIR, filename))
    file_chunks = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""   # ⚠️ "" (or None) for scanned / image-only pages
        for chunk_num, chunk in enumerate(split_text(text)):
            if not chunk.strip():
                continue  # an empty chunk would waste tokens
            all_docs.append(chunk)
            all_ids.append(f"{filename}_p{page_num}_c{chunk_num}")
            all_metadata.append({"source": filename, "page": page_num})
            file_chunks += 1
    print(f"  {filename}: {file_chunks} chunks")

# ── 3. Embed all chunks ourselves — in batches ───────────────────────────────
#
# The embeddings API takes a LIST of texts: one network round-trip for many chunks.
# Same tokens → same price as one-by-one, but much faster.
# We cut the list into batches of BATCH_SIZE so a big corpus never exceeds the request limit.

print("\n" + "=" * 55)
print("SECTION 3 — Embedding the chunks (batched)")
print("=" * 55)

all_embeddings = []
api_calls      = 0
for start in range(0, len(all_docs), BATCH_SIZE):
    batch = all_docs[start:start + BATCH_SIZE]
    all_embeddings.extend(embed_batch(batch))
    api_calls += 1

vector_size = len(all_embeddings[0])
print(f"  {len(all_embeddings)} chunks → {api_calls} API call(s) → "
      f"{len(all_embeddings)} vectors × {vector_size} numbers")

# ── 4. Store the ready-made vectors in Chroma ────────────────────────────────
#
# embeddings= is the key: Chroma stores our vectors as they are and does not call any model.
# TIP: the four lists line up by position — ids[i], documents[i], metadatas[i] and
#      embeddings[i] all describe the same chunk.

collection.upsert(
    documents=all_docs,
    ids=all_ids,
    metadatas=all_metadata,
    embeddings=all_embeddings,
)
print(f"  {collection.count()} chunks stored in Chroma")

# ── 5. Search with our own query vector ──────────────────────────────────────
#
# ⚠️ The query MUST be embedded with the same model as the documents. This collection has no
#    embedding_function, so query_texts= would use Chroma's default 384-number model and crash
#    against our 1536-number vectors. So we embed the query ourselves → query_embeddings=.

print("\n" + "=" * 55)
print("SECTION 5 — Search with query_embeddings=")
print("=" * 55)

query           = "The major ideologies"
query_embedding = embed_batch([query])[0]          # a batch of one
results         = collection.query(query_embeddings=[query_embedding], n_results=2)

print(f"Query: '{query}'\n")
for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    preview = doc[:100].replace("\n", " ")
    print(f"  {meta['source']}  page {meta['page']}  distance: {distance:.3f}")
    print(f"       {preview}...\n")
