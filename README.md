# SeamanBot — AI3 Machine Project

**SeamanBot** is a RAG-powered chatbot for the **POEA Standard Employment Contract (SEC) for Filipino Seafarers**.  
Built by Group Kamote | CSS181 Section 03 | Jherrie Nathaniel C. Millena · Xabina Rein D.J. Teodoro · Pratik Nanwani

---

## Overview

SeamanBot answers questions about Philippine maritime labor law using a multi-layer Retrieval-Augmented Generation (RAG) pipeline:

- **Hybrid retrieval:** BM25 + FAISS with Reciprocal Rank Fusion (RRF)
- **Multi-query expansion:** LLM-generated (Llama) or template-based (Qwen) query reformulations
- **Reranking:** BGE Reranker v2-m3 cross-encoder
- **Diversity:** MMR (Maximal Marginal Relevance, λ=0.70)
- **Embeddings:** `BAAI/bge-large-en-v1.5` (1024-dim)
- **Anti-hallucination:** Faithfulness retry gate (threshold ≥ 0.35)

Two model variants are available:

| Variant | Model | Hardware |
|---|---|---|
| **Llama** | Llama-3.1-8B-Instant via Groq API | CPU-only (cloud API) |
| **Qwen** | Qwen2.5-1.5B-Instruct (4-bit NF4) | GPU required |

---

## Permanent Deployments

### Llama variant — `deploy_llama/`

Standalone Gradio app using **Llama-3.1-8B-Instant** via the Groq cloud API.

```bash
cd deploy_llama
pip install -r requirements.txt
export GROQ_API_KEY="your_key_from_console.groq.com"
# Place seaman.pdf in deploy_llama/
python app.py
```

See [`deploy_llama/README.md`](deploy_llama/README.md) for full instructions and Hugging Face Spaces deployment guide.

### Qwen variant — `deploy_qwen/`

Standalone Gradio app using **Qwen2.5-1.5B-Instruct** loaded locally with 4-bit NF4 quantization. Requires a CUDA GPU.

```bash
cd deploy_qwen
pip install -r requirements.txt
# Place seaman.pdf in deploy_qwen/
python app.py
```

See [`deploy_qwen/README.md`](deploy_qwen/README.md) for full instructions and Hugging Face Spaces deployment guide.

---

## Directory Structure

```
AI-3---Kamote/
├── deploy_llama/                  ← Permanent deployment: Llama-3.1-8B-Instant (Groq)
│   ├── app.py                     ← Standalone Gradio app
│   ├── requirements.txt
│   └── README.md
├── deploy_qwen/                   ← Permanent deployment: Qwen2.5-1.5B-Instruct (local GPU)
│   ├── app.py                     ← Standalone Gradio app
│   ├── requirements.txt
│   └── README.md
├── css181_03_kamote_llama_v6.py   ← Original Colab script (Llama, with evaluation)
├── css181_03_kamote_qwen_v6.py    ← Original Colab script (Qwen, with evaluation)
├── CSS181_03_Kamote_Llama_v6.ipynb
├── CSS181_03_Kamote_Qwen_v6.ipynb
└── README.md
```

---

## Notes

- `seaman.pdf` (the POEA Standard Employment Contract) is **not included** in this repository. Place it in the appropriate `deploy_llama/` or `deploy_qwen/` directory before running.
- The Groq API key is **never hardcoded**. Set the `GROQ_API_KEY` environment variable or Space secret.
- The original Colab notebooks (`*.ipynb`) and scripts (`*.py`) include evaluation metrics (Recall@K, Precision@K, ROUGE-L, BERTScore) that are not part of the deployed apps.