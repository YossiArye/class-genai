"""
05. ChromaDB — Loading PDF Files (chunking + metadata)
======================================================
What this lesson teaches:
- How to read the text out of PDF files with pypdf
- Why we split long text into overlapping chunks before embedding it
- How to store metadata (source file, page) next to each chunk — so answers can cite their source
- How to filter a search by metadata with where=

Prerequisites: 04_chroma_openai_embeddings.py, OPENAI_API_KEY in .env,
               PDF files in 02_rag_vector_databases/pdfs/

How to run:
    python 02_rag_vector_databases/05_chroma_load_pdf_files.py

Expected output:
    How many pages and chunks were loaded per PDF, the 2 chunks closest to the query with
    their source file and page, the same query limited to one PDF, and a peek at 3 stored vectors.
"""

import os

import chromadb                                       # vector database
from chromadb.utils import embedding_functions        # built-in embedding functions
from dotenv import load_dotenv                        # reads OPENAI_API_KEY from .env
from pypdf import PdfReader                           # extracts text from PDF pages

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
DB_PATH         = "./02_rag_vector_databases/db"
PDF_DIR         = "02_rag_vector_databases/pdfs"
CHUNK_SIZE      = 500    # characters per chunk (not tokens — simple and good enough here)
OVERLAP         = 50     # characters shared by two neighbouring chunks

# ── 1. Setup ─────────────────────────────────────────────────────────────────

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name=EMBEDDING_MODEL,
)

client = chromadb.PersistentClient(path=DB_PATH)

# Start fresh each run. With get_or_create + upsert alone, chunks from an OLD version of a PDF
# would stay in the collection forever (upsert updates and adds, it never deletes).
try:
    client.delete_collection("pdf_collection")
except Exception:
    pass  # collection didn't exist yet — that's fine
collection = client.get_or_create_collection(
    name="pdf_collection",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},  # cosine = best for text: measures meaning, not magnitude
)


# ── 2. Chunking ──────────────────────────────────────────────────────────────
#
# Why chunk? One embedding for a whole 20-page PDF is a blurry average of everything in it —
# it matches nothing well. Small chunks = precise matches, and only the relevant part is
# sent to the LLM later.
#
# Why overlap? A sentence cut in half at a chunk border would be lost from both chunks.
# With 50 shared characters, the end of one chunk is repeated at the start of the next.

def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks of chunk_size characters."""
    step = chunk_size - overlap
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


# ── 3. Load the PDFs ─────────────────────────────────────────────────────────
#
# We chunk page by page, so every chunk knows which page it came from. That metadata is what
# lets a RAG answer say "source: file X, page 4" later.

print("=" * 55)
print("SECTION 3 — Loading PDFs")
print("=" * 55)

all_docs, all_ids, all_metadata = [], [], []

# TIP: sorted() — os.listdir() returns files in random order, which would change the IDs per run
for filename in sorted(os.listdir(PDF_DIR)):
    if not filename.endswith(".pdf"):
        continue
    reader      = PdfReader(os.path.join(PDF_DIR, filename))
    file_chunks = 0
    for page_num, page in enumerate(reader.pages, start=1):
        # ⚠️ extract_text() returns "" (or None in old pypdf) for scanned / image-only pages
        text = page.extract_text() or ""
        for chunk_num, chunk in enumerate(split_text(text)):
            if not chunk.strip():
                continue  # skip empty chunks — they would waste an embedding call
            all_docs.append(chunk)
            all_ids.append(f"{filename}_p{page_num}_c{chunk_num}")   # IDs must be unique
            all_metadata.append({"source": filename, "page": page_num})
            file_chunks += 1
    print(f"  {filename}: {len(reader.pages)} pages → {file_chunks} chunks")

# TIP: one upsert() for all chunks = Chroma sends them to OpenAI in batches, not one call each
collection.upsert(documents=all_docs, ids=all_ids, metadatas=all_metadata)
print(f"\n  Total: {collection.count()} chunks stored")

# ── 4. Search — with the source of every result ──────────────────────────────

print("\n" + "=" * 55)
print("SECTION 4 — Similarity search")
print("=" * 55)

query   = "The major ideologies"
results = collection.query(query_texts=[query], n_results=2)

print(f"Query: '{query}'\n")
for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    preview = doc.replace("\n", " ")
    print(f"  {meta['source']}  page {meta['page']}  distance: {distance:.3f}")
    print(f"       {preview}.................\n")

# ── 5. Filter by metadata ────────────────────────────────────────────────────
#
# where= limits the search to chunks whose metadata matches — e.g. "only this document".
# This is how real apps search one customer's files, or one product's manuals.

print("=" * 55)
print("SECTION 5 — Search inside one PDF only (where=)")
print("=" * 55)

one_file = all_metadata[0]["source"]
results  = collection.query(query_texts=[query], n_results=2, where={"source": one_file})

print(f"Query: '{query}'   where source = '{one_file}'\n")
for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
    preview = doc[:100].replace("\n", " ")
    print(f"  {meta['source']}  page {meta['page']}  distance: {distance:.3f}")
    print(f"       {preview}...\n")

# ── 6. Peek at the stored vectors ────────────────────────────────────────────
#
# collection.get() returns what Chroma stored. Embeddings are NOT included by default
# (they are big), so we ask for them with include=.

print("=" * 55)
print("SECTION 6 — What is stored per chunk")
print("=" * 55)

data = collection.get(limit=3, include=["embeddings", "metadatas"])
for chunk_id, meta, vector in zip(data["ids"], data["metadatas"], data["embeddings"]):
    first_three = [round(float(x), 3) for x in vector[:3]]
    print(f"  {chunk_id:<45} page {meta['page']:>2}  {first_three}... ({len(vector)} numbers)")
