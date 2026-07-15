from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import os

# ── Settings ──────────────────────────────────────────────────────────────────
FILE = "sample.pdf"          # Change to your file name
COLLECTION_NAME = "documentqa"
CHROMA_PATH = "./vectorstore"
# ──────────────────────────────────────────────────────────────────────────────

print("="*60)
print("STEP 1: Extracting text from document...")
print("="*60)

reader = PdfReader(FILE)
full_text = ""
for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        full_text += f"\n[Page {page_num}]\n{text}"

print(f"Total characters extracted: {len(full_text)}")

print("\n" + "="*60)
print("STEP 2: Splitting into chunks...")
print("="*60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_text(full_text)
print(f"Total chunks created: {len(chunks)}")

print("\n" + "="*60)
print("STEP 3: Loading embedding model...")
print("="*60)

model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")

print("\n" + "="*60)
print("STEP 4: Generating embeddings for each chunk...")
print("="*60)

embeddings = model.encode(chunks, show_progress_bar=True)
print(f"Embeddings generated: {len(embeddings)} vectors")
print(f"Each vector size: {len(embeddings[0])} dimensions")

print("\n" + "="*60)
print("STEP 5: Storing in ChromaDB...")
print("="*60)

# Create ChromaDB client and collection
os.makedirs(CHROMA_PATH, exist_ok=True)
client = chromadb.PersistentClient(path=CHROMA_PATH)

# Delete existing collection if it exists (fresh start)
try:
    client.delete_collection(COLLECTION_NAME)
except:
    pass

collection = client.create_collection(COLLECTION_NAME)

# Add chunks with embeddings and metadata
collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"chunk_index": i} for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in ChromaDB.")
print(f"Database saved at: {CHROMA_PATH}")

print("\n" + "="*60)
print("DONE. Embeddings stored successfully.")
print("="*60)