from sentence_transformers import SentenceTransformer
import chromadb
import requests
import json

# ── Settings ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "documentqa"
CHROMA_PATH = "./vectorstore"
TOP_K = 3
OLLAMA_MODEL = "tinyllama"

# Expletives guardrail list
EXPLETIVES = ["damn", "shit", "fuck", "ass", "crap", "bastard", "bitch", "hell"]
# ──────────────────────────────────────────────────────────────────────────────

def check_expletives(text):
    text_lower = text.lower()
    for word in EXPLETIVES:
        if word in text_lower:
            return True
    return False

def retrieve_chunks(query, collection, model):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )
    return results["documents"][0]

def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    )
    return response.json()["response"]

def build_prompt(query, chunks):
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    return f"""You are a helpful assistant. Answer the question based only on the context below.
Also mention which page number the answer comes from if it appears in the context.

Context:
{context}

Question: {query}

Answer:"""

# ── Load model and DB ─────────────────────────────────────────────────────────
print("="*60)
print("  DocumentQA — Q&A System")
print("="*60)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(COLLECTION_NAME)

print(f"  Chunks in database: {collection.count()}")
print("  Type 'exit' to quit.")
print("="*60)

# ── Chat loop ─────────────────────────────────────────────────────────────────
while True:
    query = input("\nYou: ").strip()

    if query.lower() == "exit":
        print("Goodbye!")
        break

    if not query:
        continue

    # Guardrail check
    if check_expletives(query):
        print("\nBot: I'm sorry, but expletives are not allowed. Please rephrase your question politely.")
        continue

    # Retrieve relevant chunks
    print("\n[Retrieving relevant sections...]")
    chunks = retrieve_chunks(query, collection, embedding_model)

    # Generate answer
    print("[Generating answer...]\n")
    prompt = build_prompt(query, chunks)
    answer = ask_ollama(prompt)

    print(f"Bot: {answer}")
    print("-"*60)
