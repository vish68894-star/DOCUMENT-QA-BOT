from sentence_transformers import SentenceTransformer
import chromadb

# ── Settings ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "documentqa"
CHROMA_PATH = "./vectorstore"
TOP_K = 3   # number of most relevant chunks to retrieve
# ──────────────────────────────────────────────────────────────────────────────

# Load embedding model and ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print("="*60)
print("  DocumentQA — Retrieval Test")
print(f"  Chunks in database: {collection.count()}")
print("="*60)

# ── Ask a question ────────────────────────────────────────────────────────────
query = input("\nAsk a question about your document:\n> ")

# Convert question to embedding
query_embedding = model.encode([query]).tolist()

# Search ChromaDB for top-k most relevant chunks
results = collection.query(
    query_embeddings=query_embedding,
    n_results=TOP_K
)

# ── Print results ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  Top {TOP_K} relevant chunks found:")
print("="*60)

for i, doc in enumerate(results["documents"][0]):
    print(f"\n--- Result {i+1} ---")
    print(doc)

print("\n" + "="*60)
print("Retrieval complete.")
print("="*60)
