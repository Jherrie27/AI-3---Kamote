---
title: SeamanBot Qwen
emoji: ⚓
colorFrom: green
colorTo: teal
sdk: gradio
sdk_version: "4.0"
app_file: app.py
pinned: false
---

# SeamanBot — Qwen2.5-1.5B-Instruct

A permanent Gradio deployment of **SeamanBot**, a RAG-powered chatbot for the POEA Standard Employment Contract for Filipino Seafarers.

**Model:** Qwen2.5-1.5B-Instruct (loaded locally with 4-bit NF4 quantization)  
**Retrieval:** Hybrid BM25 + FAISS → RRF fusion → BGE Reranker v2-m3 → MMR diversity  
**Embeddings:** `BAAI/bge-large-en-v1.5` (1024-dim)

> ⚠️ **GPU Required:** This app loads the Qwen model locally with 4-bit quantization via BitsAndBytes. A CUDA-capable GPU is required.

---

## Setup

### 1. GPU requirement

This app requires a CUDA-capable GPU (NVIDIA) for 4-bit NF4 quantization. At least **4 GB VRAM** is recommended for the 1.5B model.

On Hugging Face Spaces, select a **GPU runtime** (e.g., T4 or A10G) when creating the Space.

### 2. Place `seaman.pdf` in the directory

The app expects the POEA Standard Employment Contract PDF to be in the same directory as `app.py`:

```
deploy_qwen/
├── app.py
├── requirements.txt
├── README.md
└── seaman.pdf   ← place it here
```

On Hugging Face Spaces, upload `seaman.pdf` as a repository file.

---

## Running Locally

```bash
cd deploy_qwen
pip install -r requirements.txt
python app.py
```

The app will be available at `http://localhost:7860`.

> **Note:** The first run will download the Qwen2.5-1.5B-Instruct model (~3 GB) and the BGE embedding/reranker models (~1.5 GB total) from Hugging Face Hub.

---

## Deploying on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Set **SDK** to `Gradio`.
3. Select a **GPU hardware** tier (T4 small or above) in Space Settings.
4. Upload all files from `deploy_qwen/` (including `seaman.pdf`).
5. The Space will build and launch automatically.

No API keys are required — the model runs entirely locally on the GPU.

---

## Architecture

```
User Question
     │
     ▼
Multi-Query Expansion (3 template-based reformulations)
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
Metadata-Aware Prompt → Qwen2.5-1.5B-Instruct (4-bit NF4, local GPU)
     │
     ▼
Anti-Hallucination Retry Gate (faithfulness ≥ 0.35)
     │
     ▼
Answer + Retrieved Context
```
