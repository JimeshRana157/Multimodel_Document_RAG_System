# 📑 Multimodal Document RAG System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Qdrant](https://img.shields.io/badge/Qdrant-MultiVector%20MaxSim-red.svg)](https://qdrant.tech/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, enterprise multimodal retrieval-augmented generation (RAG) system for complex documents (PDFs, financial reports, technical documentation, research papers, scanned documents).

Combines **ColPali visual multi-vector embeddings** and **BM25 lexical signals** with **Reciprocal Rank Fusion (RRF)**, backed by **Qdrant Vector Database (MaxSim)**, providing grounded answers via **Vision-Language Models (VLM)**, real-time streaming, decoupled clean citations, dynamic patch-relevance heatmaps, and a full IR benchmark suite.

---

## 🏛️ System Architecture

```
                                  ┌─────────────────────────────┐
                                  │      Streamlit Frontend     │
                                  │  (Chat, Explorer, Benchmark)│
                                  └──────────────┬──────────────┘
                                                 │ HTTP / SSE
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                     FastAPI Backend Server                                  │
│                                                                                             │
│  ┌───────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐  │
│  │   Document Ingestion  │   │     Indexing Pipelines    │   │     Retrieval & Fusion    │  │
│  │                       │   │                           │   │                           │  │
│  │  • PyMuPDF 300 DPI    │   │  • ColPali Visual Embeds  │   │  • BM25 Lexical Search    │  │
│  │  • Text & Table OCR   │──▶│  • BM25 Inverted Index    │──▶│  • Qdrant MultiVec Search │  │
│  │  • Hash Deduplication │   │  • Qdrant MaxSim Storage  │   │  • Reciprocal Rank Fusion │  │
│  └───────────────────────┘   └───────────────────────────┘   └─────────────┬─────────────┘  │
│                                                                            │                │
│                                                                            ▼                │
│  ┌───────────────────────────────────────────────────────┐   ┌───────────────────────────┐  │
│  │                Generation & Explainability            │   │    Evaluation Engine      │  │
│  │                                                       │   │                           │  │
│  │  • Vision-Language Model (OpenAI / Local Ollama LLaVA)│   │  • Recall@1, 3, 5         │  │
│  │  • Decoupled Structured Citations [Doc, Page]         │   │  • MRR & NDCG@5           │  │
│  │  • Dynamic Patch Relevance Heatmap Overlays           │   │  • P50/P95 Latency Profil │  │
│  └───────────────────────────────────────────────────────┘   └───────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

1. **Dual Visual & Lexical Hybrid Retrieval**:
   - **ColPali Multi-Vector Embeddings**: Encodes page images into spatial patch tokens ($16 \times 16$ grid) preserving layout, charts, formulas, and visual structure.
   - **BM25Okapi Lexical Index**: Tokenizes page text & table contents with smart English stopwords.
   - **Reciprocal Rank Fusion (RRF)**: $RRF(d) = \sum_{r \in R} \frac{w_r}{k + \text{rank}_r(d)}$ with configurable visual vs. lexical balance.

2. **Qdrant Vector DB with Late-Interaction (MaxSim)**:
   - Native multivector comparator implementing ColBERT-style late interaction:
     $$S(q, d) = \sum_{i} \max_{j} (\mathbf{q}_i \cdot \mathbf{d}_j^\top)$$
   - Automatic remote Docker connection with built-in embedded SQLite/local fallback.

3. **Dynamic Patch Relevance Heatmaps**:
   - Computes query-token $\times$ image-patch similarity matrix: $r_j = \max_i (\mathbf{q}_i \cdot \mathbf{d}_j^\top)$.
   - Maps normalized relevance back to image pixel coordinates, rendering visual heatmap overlays for explainable AI.

4. **Decoupled Answer & Citation Pipeline**:
   - The Vision LLM produces clean, natural-language answers without text pollution.
   - Separate, traceable citations (`[Doc: filename, Page: X]`) linked directly to visual page evidence.

5. **Search Scope Control**:
   - Filter queries across `All Indexed Documents` or isolate to a `Specific Document`.

6. **Information Retrieval Benchmark Suite**:
   - Rigorous evaluation metrics: **Recall@1, Recall@3, Recall@5, MRR, NDCG@5**.
   - Independent latency profiling with **Mean, P50 (median), and P95** measurements.

7. **CPU-Friendly & Production Ready**:
   - Automatic hardware detection (`CUDA`, `MPS`, `CPU`). Zero-freeze CPU execution.

---

## 📁 Repository Structure

```
multimodal-financial-rag/
├── api/
│   ├── __init__.py
│   ├── main.py                     # FastAPI application factory & lifespan
│   └── routes.py                   # REST API routes (upload, index, search, ask, health)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central Settings dataclass & device auto-detect
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py        # PyMuPDF rendering (300 DPI) & metadata
│   │   └── text_extractor.py       # Page-level text & table extraction
│   ├── indexing/
│   │   ├── __init__.py
│   │   ├── bm25_indexer.py         # BM25Okapi inverted index with persistence
│   │   ├── colpali_engine.py       # Multi-vector vision & query embedder
│   │   └── qdrant_store.py         # Qdrant multivector collection & search
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── late_interaction.py     # MaxSim late-interaction & patch heatmaps
│   │   └── hybrid_fusion.py        # Weighted Reciprocal Rank Fusion (RRF)
│   ├── generation/
│   │   ├── __init__.py
│   │   └── vision_llm.py           # Vision LLM (OpenAI/Ollama) & extractive reasoning
│   └── evaluation/
│       ├── __init__.py
│       └── benchmark_ragas.py      # Recall@K, MRR, NDCG@K, Latency percentiles
├── ui/
│   └── app.py                      # 3-Tab Streamlit UI (Chat, Explorer, Benchmark)
├── scripts/
│   └── download_sample_data.py     # Public PDF download utility
├── tests/
│   ├── conftest.py                 # Pytest fixtures & synthetic PDF generator
│   ├── test_api.py                 # API endpoint tests
│   ├── test_bm25.py                # BM25 indexing & persistence tests
│   ├── test_colpali.py             # Embedder dimension tests
│   ├── test_hybrid_fusion.py       # RRF fusion & deduplication tests
│   ├── test_maxsim.py              # MaxSim & heatmap overlay tests
│   ├── test_pdf_processor.py       # PDF rendering & metadata tests
│   ├── test_qdrant.py              # Qdrant collection & search tests
│   └── test_text_extractor.py      # Text & table extraction tests
├── data/                           # Ignored in git (renders, index, raw uploads)
├── .env.example                    # Environment template
├── .gitignore                      # Git ignore rules
├── docker-compose.yaml             # Qdrant container service
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## 🚀 Step-by-Step Installation & Recreation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/ssj27072002/MultiModal_RAG_System.git
cd MultiModal_RAG_System
```

### 2. Set Up a Python Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Copy the example environment file:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Edit `.env` to configure your preferred settings:

```env
# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=document_pages
QDRANT_LOCAL_PATH=./data/qdrant_storage

# Vision-Language Model (Choose Option A or B)
# Option A: Local Ollama (Free, runs locally)
VLM_ENDPOINT=http://localhost:11434/v1
VLM_MODEL=llava
VLM_API_KEY=not-needed

# Option B: OpenAI GPT-4o
# VLM_ENDPOINT=https://api.openai.com/v1
# VLM_MODEL=gpt-4o
# VLM_API_KEY=sk-your-openai-key-here

# Hardware & Retrieval Settings
DEVICE=auto
PRECISION=auto
TOP_K=5
RRF_K=60
VISUAL_WEIGHT=0.7
LEXICAL_WEIGHT=0.3
DPI=300
```

---

## 🏃 Running the Application

### Step 1: (Optional) Start Qdrant Docker & Local VLM

- **Qdrant** (Optional - if Docker is not installed, the system automatically uses embedded local storage):
  ```bash
  docker compose up qdrant -d
  ```

- **Local Ollama VLM** (Optional - if you want local vision generation):
  ```bash
  ollama run llava
  ```

### Step 2: Start the FastAPI Backend Server

In your first terminal:
```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
*API Swagger Documentation will be live at: [http://localhost:8000/docs](http://localhost:8000/docs)*

### Step 3: Start the Streamlit Frontend UI

In your second terminal:
```bash
python -m streamlit run ui/app.py
```
*Streamlit UI will be live at: [http://localhost:8501](http://localhost:8501)*

---

## 💻 Using the Application

1. **Upload Documents (Sidebar)**:
   - Upload any single-page or multi-page PDFs.
   - Click **"🚀 Upload & Auto-Index"** to automatically render high-res pages (300 DPI), extract texts/tables, and index visual & lexical representations.
2. **Ask Document AI (Tab 1)**:
   - Ask natural language questions with configurable Search Scope (`All Documents` or `Current Document Filter`).
   - Receive streaming answers, decoupled verified citations, expandable visual page evidence, and detailed **Retrieval Diagnostics** (BM25 vs Visual latency, component scores).
3. **Document Explorer & Visual Search (Tab 2)**:
   - Browse high-resolution rendered pages in the interactive gallery.
   - Perform targeted visual similarity searches inside specific documents.
4. **RAG Evaluation & Benchmark (Tab 3)**:
   - Run empirical comparisons between **BM25 Lexical**, **ColPali Visual**, and **Hybrid RRF**.
   - Review genuine Recall@K, MRR, NDCG@5, and latency percentile matrices.

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/upload` | Upload PDF files, render pages, extract text, and auto-index |
| `POST` | `/documents/index` | Trigger background indexing for specific or all documents |
| `GET`  | `/documents` | List all processed documents and their indexing status |
| `DELETE` | `/documents/{id}` | Delete a document, its index points, and rendered pages |
| `POST` | `/search` | Run hybrid, lexical, or visual retrieval with diagnostics |
| `POST` | `/ask` | Streaming SSE endpoint for grounded VLM answer generation |
| `GET`  | `/pages/{id}/{page}` | Serve high-resolution rendered PNG of a document page |
| `GET`  | `/health` | System health status (Qdrant, VLM, ColPali, document count) |

---

## 🧪 Running Automated Tests

Run the complete test suite with `pytest`:

```bash
python -m pytest tests/ -v --tb=short
```

---

## 🛠️ Technology Stack

- **Frameworks**: FastAPI, Streamlit, PyTorch, Hugging Face Transformers
- **Retrieval & NLP**: ColPali, SentenceTransformers, Rank-BM25, PyMuPDF (fitz)
- **Vector Database**: Qdrant (MultiVector MaxSim)
- **Visualization**: Matplotlib, PIL (Pillow)
- **Testing**: Pytest, Pytest-Asyncio, HTTPX

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
