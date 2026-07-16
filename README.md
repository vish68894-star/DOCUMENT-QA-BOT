# DOCUMENT-QA-BOT


# DocumentQA — AI-Powered Document Question Answering System

A domain-agnostic, standalone chatbot that allows users to upload documents and ask questions about their contents. Built on a Retrieval-Augmented Generation (RAG) architecture with hybrid retrieval, page and section references, and an expletive guardrail.

---

## Features

- **Multi-format document upload** — PDF, Word (.docx), Excel (.xlsx), TXT, Markdown
- **Q&A chatbot** — ask any question about the uploaded document
- **Page and section references** — every answer cites the exact page and section it came from
- **Document summarization** — one-click summary of the entire document
- **Hybrid RAG** — combines semantic vector search (ChromaDB) and keyword search (BM25) for better retrieval
- **Expletive guardrail** — politely blocks inappropriate language
- **Off-topic guardrail** — detects and blocks questions unrelated to the uploaded document
- **AirLLM support** — layer-wise model sharding for running 7B LLMs on limited hardware

---

## System Architecture

```
User
 ↓
Upload Document (PDF / Word / Excel / TXT / Markdown)
 ↓
Text Extraction (pdfplumber / python-docx / openpyxl)
 ↓
Chunking (LangChain RecursiveCharacterTextSplitter)
 ↓
Embeddings (Sentence Transformers — all-MiniLM-L6-v2)
 ↓
Vector Store (ChromaDB) + BM25 Index (rank_bm25)
 ↓
Hybrid Retrieval (Vector + Keyword)
 ↓
Guardrails (Expletive + Off-topic check)
 ↓
LLM Inference (Ollama/llama3.2 or AirLLM/Platypus2-7B)
 ↓
Answer with Page + Section References
 ↓
Streamlit Chat Interface
```

---

## Tech Stack

| Category | Tool |
|---|---|
| UI | Streamlit |
| PDF Parsing | pdfplumber, pypdf |
| Word Parsing | python-docx |
| Excel Parsing | openpyxl |
| RAG Framework | LangChain |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB |
| Keyword Search | rank_bm25 |
| LLM (Demo) | llama3.2 via Ollama |
| LLM (Production) | AirLLM with Platypus2-7B |

---

## Project Structure

```
DocumentQA/
├── app.py                  Streamlit chatbot UI (main application)
├── airllm_server.py        AirLLM Flask API server (production LLM)
├── 01_ingestion.py         Document text and table extraction
├── 02_chunking.py          Text chunking with LangChain
├── 03_embeddings.py        Embedding generation and ChromaDB storage
├── 04_retrieval.py         Semantic retrieval from ChromaDB
├── 05_qa.py                Terminal Q&A pipeline with guardrails
├── 06_airllm_test.py       AirLLM standalone test script
├── .streamlit/
│   └── config.toml         Streamlit light theme configuration
├── vectorstore/            ChromaDB persistent database
├── venv/                   Main virtual environment
└── venv_airllm/            AirLLM isolated virtual environment
```

---

## Setup and Installation

### Prerequisites
- Python 3.11
- [Ollama](https://ollama.com/download) installed

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/DocumentQA.git
cd DocumentQA
```

### Step 2 — Create virtual environment
```bash
python -m venv venv
```

**Windows:**
```bash
& venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install streamlit pypdf pdfplumber python-docx openpyxl langchain langchain-text-splitters sentence-transformers chromadb requests rank_bm25
```

### Step 4 — Pull the LLM
```bash
ollama pull llama3.2
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## How to Use

1. Upload a document from the sidebar (PDF, Word, Excel, TXT, or Markdown)
2. Click **Process Document** and wait for indexing to complete
3. Select a retrieval mode — Hybrid, Vector only, or Keyword only
4. Type a question in the chat box and press Enter
5. The answer will include page and section references
6. Click **Summarize Document** for a one-click summary

---

## AirLLM Setup (Production)

AirLLM enables running large 7B parameter models on limited hardware using layer-wise sharding.

### Step 1 — Create AirLLM environment
```bash
python -m venv venv_airllm
& venv_airllm\Scripts\Activate.ps1   # Windows
pip install airllm flask torch transformers==4.38.2 optimum==1.16.2 accelerate==0.27.2 huggingface_hub==0.21.0
```

### Step 2 — Run AirLLM server
```bash
python airllm_server.py
```

The server downloads Platypus2-7B (~13GB) on first run and starts on `http://localhost:5001`.

### Step 3 — Switch the app to AirLLM
In `app.py`, update the endpoint in the `ask_ollama` function:
```python
# Change this:
"http://localhost:11434/api/generate"
# To this:
"http://localhost:5001/generate"
```

> **Note:** AirLLM inference on CPU takes ~5 minutes per query. A CUDA-enabled GPU reduces this to under 10 seconds.

---

## Retrieval Modes

| Mode | How it works | Best for |
|---|---|---|
| **Hybrid** | Combines vector search + BM25 keyword search | General use — most complete answers |
| **Vector only** | Semantic similarity via embeddings | Conceptual or paraphrased queries |
| **Keyword only** | BM25 term frequency ranking | Exact terminology or section lookups |

---

## Guardrails

| Guardrail | Trigger | Response |
|---|---|---|
| Expletive filter | Query contains profanity | "Expletives are not allowed. Please rephrase your question politely." |
| Off-topic filter | Query is unrelated to the document | "I can only answer questions related to the uploaded document." |

---

## Requirements

```
streamlit
pypdf
pdfplumber
python-docx
openpyxl
langchain
langchain-text-splitters
sentence-transformers
chromadb
requests
rank_bm25
```

---

## Developed by

K Vishwajit — Mechatronics Engineering, VIT Chennai
Internship Project — June/July 2026
