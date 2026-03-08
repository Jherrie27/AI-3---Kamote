---
title: SeamanBot Llama
emoji: ⚓
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---

# SeamanBot — Llama-3.1-8B-Instant

A permanent Gradio deployment of **SeamanBot**, a RAG-powered chatbot for the POEA Standard Employment Contract for Filipino Seafarers.

**Model:** Llama-3.1-8B-Instant via Groq API  
**Retrieval:** Hybrid BM25 + FAISS → RRF fusion → BGE Reranker v2-m3 → MMR diversity  
**Embeddings:** `BAAI/bge-large-en-v1.5` (1024-dim)

---

## Setup

### 1. Set the GROQ API key

The app reads the Groq API key from the `GROQ_API_KEY` environment variable. **Never hardcode your key.**

```bash
export GROQ_API_KEY="your_key_from_console.groq.com"
```

On Hugging Face Spaces, add it as a **Secret** (Settings → Repository secrets → `GROQ_API_KEY`).

### 2. Place `seaman.pdf` in the directory

The app expects the POEA Standard Employment Contract PDF to be in the same directory as `app.py`:

```
deploy_llama/
├── app.py
├── requirements.txt
├── README.md
└── seaman.pdf   ← place it here
```

On Hugging Face Spaces, upload `seaman.pdf` as a repository file.

---

## Running Locally

```bash
cd deploy_llama
pip install -r requirements.txt
export GROQ_API_KEY="your_key_here"
python app.py
```

The app will be available at `http://localhost:7860`.

---

## Deploying on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Set **SDK** to `Gradio`.
3. Upload all files from `deploy_llama/` (including `seaman.pdf`).
4. Add `GROQ_API_KEY` as a repository secret in Space Settings.
5. The Space will build and launch automatically.

No GPU is required — all heavy computation uses the Groq cloud API.

---

## Architecture

```
User Question
     │
     ▼
Multi-Query Expansion (Groq LLM → 3 reformulations)
     │
     ▼
Hybrid Retrieval: BM25 + FAISS (k=80 each)
     │
     ▼
RRF Fusion + Stem Keyword Boost
     │
     ▼
BGE Reranker v2-m3 (top 120 → top 15)
     │
     ▼
MMR Diversity (λ=0.70, final top 5)
     │
     ▼
Metadata-Aware Prompt → Llama-3.1-8B-Instant (Groq)
     │
     ▼
Anti-Hallucination Retry Gate (faithfulness ≥ 0.35)
     │
     ▼
Answer + Retrieved Context
```
