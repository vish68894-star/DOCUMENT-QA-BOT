from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ── Change this to your file name if different ────────────────────────────────
FILE = "sample.pdf"
# ─────────────────────────────────────────────────────────────────────────────

# Step 1: Extract all text from the PDF
reader = PdfReader(FILE)
full_text = ""

for page_num, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    full_text += f"\n[Page {page_num}]\n{text}"

print(f"Total characters extracted: {len(full_text)}")
print("=" * 50)

# Step 2: Split the text into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # each chunk is max 500 characters
    chunk_overlap=50      # chunks share 50 characters with the next one
)

chunks = splitter.split_text(full_text)

print(f"Total chunks created: {len(chunks)}")
print("=" * 50)

# Step 3: Preview first 3 chunks
for i, chunk in enumerate(chunks[:3]):
    print(f"\n--- Chunk {i+1} ---")
    print(chunk)

print("=" * 50)
print("Chunking complete.")