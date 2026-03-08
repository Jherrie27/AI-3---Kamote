# -*- coding: utf-8 -*-
"""SeamanBot — Qwen2.5-1.5B-Instruct (Local 4-bit)
Standalone Gradio deployment app.

Requires a CUDA-capable GPU for 4-bit NF4 quantization.
Place seaman.pdf in the same directory as this file before running.
"""

import os
import re
import math
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import nltk
import pdfplumber
from rank_bm25 import BM25Okapi

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

print("All libraries imported.")
print(f"CUDA available: {torch.cuda.is_available()}")

# ── Model configuration ──────────────────────────────────────────────────────
MODEL_LABEL        = "Qwen2.5-1.5B-Instruct (Local 4-bit)"
MODEL_DISPLAY_NAME = "Qwen2.5-1.5B-Instruct"
MODEL_NAME         = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

print("Loading model with 4-bit NF4 quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)
model.eval()
print(f"Model loaded: {MODEL_NAME}")

# v6: BGE-Large embedding + BGE Reranker v2 m3
EMBED_MODEL  = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

print(f"Loading bi-encoder   : {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
print(f"Loading cross-encoder: {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)
print("Retrieval models ready.")
print(f"  Embedding dim : {embedder.get_sentence_embedding_dimension()}")

# ── Section/topic metadata detection ────────────────────────────────────────
_META_PATTERNS = [
    (r"(?i)\bsection\s+(\d+[A-Za-z]?)\b",             "Section {}"),
    (r"(?i)\bpart\s+(I{{1,3}}V?|VI{{0,3}}|\d+)\b",    "Part {}"),
    (r"(?i)\b(medical benefit|sickness|treatment|injur)", "Medical Benefits"),
    (r"(?i)\b(death benefit|burial|deceased)",            "Death & Burial Benefits"),
    (r"(?i)\b(disabilit)",                                "Disability Benefits"),
    (r"(?i)\b(repatri)",                                   "Repatriation"),
    (r"(?i)\b(overtime|working hours)",                   "Working Hours & Overtime"),
    (r"(?i)\b(basic salary|basic wage|monthly salary)",   "Wages & Salary"),
    (r"(?i)\b(allotment|remittance)",                     "Allotment & Remittance"),
    (r"(?i)\b(terminat|dismissal|disciplin)",             "Termination & Discipline"),
    (r"(?i)\b(insurance|coverage|personal accident)",     "Insurance"),
    (r"(?i)\b(placement fee|recruitment fee)",            "Placement & Recruitment"),
    (r"(?i)\b(due process|right to be heard)",            "Due Process"),
    (r"(?i)\b(beneficiar|qualified dependent)",           "Beneficiaries"),
    (r"(?i)\b(obligation|employer shall|ship owner)",     "Employer Obligations"),
]

def _detect_section(text: str) -> str:
    probe = text[:200]
    for pattern, label in _META_PATTERNS:
        m = re.search(pattern, probe)
        if m:
            groups = m.groups()
            return label.format(groups[0].strip().title()) if "{}" in label and groups else label
    return "General"

# ── Layout-aware PDF extractor (column-aware) ────────────────────────────────
def load_and_clean_pdf(pdf_path: str) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            w, h = page.width, page.height
            for bbox in [(0, 0, w / 2, h), (w / 2, 0, w, h)]:
                col = page.crop(bbox)
                raw = col.extract_text()
                if raw:
                    raw = raw.replace("-\n", "").replace("\n", " ")
                    raw = re.sub(r"Page \d+ of \d+", "", raw)
                    raw = re.sub(r"[ \t]+", " ", raw).strip()
                    if raw:
                        full_text.append(raw)
    return "\n\n".join(full_text)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

PDF_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seaman.pdf")
pdf_text   = load_and_clean_pdf(PDF_PATH)
raw_chunks = [c.strip() for c in splitter.split_text(pdf_text) if len(c.strip()) > 80]

# ── Attach metadata to every chunk ──────────────────────────────────────────
chunk_texts = []
chunk_meta  = []
for c in raw_chunks:
    chunk_texts.append(c)
    chunk_meta.append({"section": _detect_section(c)})

print(f"Total characters : {len(pdf_text):,}")
print(f"Total chunks     : {len(chunk_texts)}")
print(f"Avg chunk length : {sum(len(c) for c in chunk_texts)//len(chunk_texts)} chars")
sections_found = set(m["section"] for m in chunk_meta)
print(f"Unique sections  : {len(sections_found)} — {sorted(sections_found)}")

# ── Dense index (FAISS) ──────────────────────────────────────────────────────
print("Generating BGE-Large dense embeddings...")
chunk_embeddings = embedder.encode(
    chunk_texts, show_progress_bar=True, batch_size=32, normalize_embeddings=True
)
dim   = chunk_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(chunk_embeddings.astype(np.float32))
print(f"FAISS index  : {index.ntotal} vectors  ({dim}-dim)")

# ── Sparse index (BM25) ──────────────────────────────────────────────────────
print("Building BM25 sparse index...")
tokenized_corpus = [c.lower().split() for c in chunk_texts]
bm25_index       = BM25Okapi(tokenized_corpus)
print(f"BM25 index   : {len(tokenized_corpus)} documents")
print("All indexes ready — v6.")

# ═══════════════════════════════════════════════════════════════════════════════
# LAYERED ANTI-HALLUCINATION DEFENSE — SeamanBot v6
# ───────────────────────────────────────────────────────────────────────────────
#  Layer 1   BGE-Large bi-encoder   BAAI/bge-large-en-v1.5 (1024-dim)
#  Layer 1b  Hybrid retrieval       BM25 + FAISS → RRF fusion, k_init=80
#  Layer 1c  Multi-query expansion  3 template-based query reformulations (Qwen)
#  Layer 1d  Stem keyword boost     6-char prefix matching
#  Layer 1e  BGE Reranker v2-m3     reranks merged pool, k_final=5
#  Layer 1f  MMR diversity          lambda=0.70
#  Layer 2   Metadata-aware prompt  chunk section labels injected into context
#  Layer 3   Prompt constraints     forbidden refusal phrases, STRICT fallback
#  Layer 4   Decoding params        rep_penalty=1.15, no_repeat_ngram=4 [Qwen]
#  Layer 5   Retry gate             retry only if faith improves >= 0.05
# ═══════════════════════════════════════════════════════════════════════════════

_STOPWORDS = {
    "what","is","the","a","an","of","for","to","in","on","are","be","by","at",
    "as","if","or","and","how","who","does","do","can","was","were","will","with",
    "from","that","this","it","its","they","their","when","under","before","after",
    "must","shall","should","would","any","all","been","have","has","not","no",
    "which","about","into","also","upon","may","per","given","such","during",
    "give","get","set","let","put","take","make","use","than","then","each",
    "some","more","very","just","only","here","there","where","case","cases",
}

def _cos(a, b):
    a, b = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

MARITIME_SYNONYMS = {
    r"\bcontract\b"                           : "contract employment term duration period agreement",
    r"\bsalary\b|\bwage\b"                 : "salary wage pay basic compensation monthly rate",
    r"\bmedical\b"                            : "medical sickness injury treatment illness hospitalization medicine",
    r"\brepatri\w+"                           : "repatriation repatriated return passage homeward cost expense transport",
    r"\bdeath\b|\bdi(ed|es)\b"             : "death deceased die died burial compensation beneficiary",
    r"\bdisciplin\w+"                         : "disciplinary discipline offense misconduct penalty dismiss sanction violation",
    r"\bovertime\b"                           : "overtime hours work additional pay rate beyond",
    r"\binsurance\b"                          : "insurance coverage protection life accident personal",
    r"\bterminat\w+"                          : "termination pre-termination dismissal grounds cause end",
    r"\bdisability\b|\bdisabled\b"         : "disability disabled permanent total partial grade schedule injury",
    r"\ballotment\b"                          : "allotment remittance family beneficiary percent percentage monthly mandatory",
    r"\bleave\b"                              : "leave vacation rest days paid entitlement annual",
    r"\bplacement\b"                          : "placement fee recruitment agency prohibited illegal banned",
    r"\bbenefit\w*"                           : "benefit benefits compensation payment entitlement",
    r"\bseafarer\b"                           : "seafarer crew member mariner employee worker",
    r"\bemployer\b"                           : "employer owner company agency principal obligation shall",
    r"\bdocument\w*"                          : "documents document copy given provided contract departure embarkation",
    r"\bdeparture\b"                          : "departure embarkation sign-off prior departure leaving document copy",
    r"\bworking\s+hours\b|\bwork\s+hours\b|\bhours\b": "working hours eight regular daily standard per day",
    r"\bdue\s+process\b"                    : "due process dismiss hearing explain opportunity informed",
    r"\b120\b"                               : "120 days hundred treatment disability assessment period beyond",
    r"\ballowanc\w+"                         : "allowance subsistence vacation leave pay entitlement additional",
    r"\breimburs\w+"                         : "reimbursement reimburse expense cost shoulder pay",
    r"\bliable\b|\bliability\b"            : "liable liability responsible obligation penalty",
    r"\bbeneficiar\w+"                       : "beneficiary beneficiaries qualified spouse children dependent family",
    r"\bsickness\b|\bill\w*\b"            : "sickness sick illness injury allowance treatment basic wage rate",
    r"\bfee\w*"                              : "fee fees placement recruitment prohibited banned",
    r"\bobligat\w+"                          : "obligation obligations employer shall must provide duty responsible",
    r"\bpercent\b|\b%\b"                   : "percent percentage allotment mandatory minimum salary portion",
    r"\bfail\b|\bfails\b|\bfailed\b"     : "fail fails failed failure employer repatriation liable responsible",
    r"\bshoulder\w*"                         : "shoulder cost expense pay responsible employer repatriation",
    r"\bqualified\b"                         : "qualified beneficiary spouse children dependent entitled",
    r"\bstandard\b"                          : "standard regular normal working hours eight daily",
    r"\bcondit\w+"                           : "conditions terms when may circumstances case grounds",
    r"\bground\w*"                           : "grounds conditions cause reasons justification termination",
    r"\bobligation\w*"                       : "obligation duty shall provide employer required",
}

def expand_query(query: str) -> str:
    expanded = query
    for pattern, expansion in MARITIME_SYNONYMS.items():
        if re.search(pattern, query, re.IGNORECASE):
            expanded += " " + expansion
    return expanded


# ── Multi-Query: 3 template reformulations (Qwen — no extra LLM call needed) ─
def _multi_query(query: str) -> list:
    stems = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)
             if w.lower() not in _STOPWORDS]
    kw = " ".join(stems[:6])
    return [
        query,
        f"What does the POEA Standard Employment Contract state regarding {kw}?",
        f"Under the POEA SEC for Filipino seafarers, what are the provisions on {kw}?",
        expand_query(query),
    ]


# ── Hybrid retrieval: BM25 + FAISS → RRF ────────────────────────────────────
def _rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)

def retrieve_context(query: str, k_init: int = 80, k_final: int = 5) -> tuple:
    queries   = _multi_query(query)
    expanded  = expand_query(query)

    # ── FAISS: embed all query variants, union results ──
    q_embs    = embedder.encode(queries, normalize_embeddings=True).astype(np.float32)
    faiss_seen, faiss_ranks_union = set(), {}
    for qi, qe in enumerate(q_embs):
        _, I = index.search(qe.reshape(1, -1), min(k_init, len(chunk_texts)))
        for r, idx in enumerate(I[0]):
            if 0 <= idx < len(chunk_texts):
                c = chunk_texts[idx]
                if c not in faiss_seen:
                    faiss_ranks_union[c] = r + 1
                    faiss_seen.add(c)
    main_q_emb = q_embs[0]

    # ── BM25 sparse retrieval ──
    bm25_scores  = bm25_index.get_scores(expanded.lower().split())
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:k_init]
    bm25_ranks   = {chunk_texts[i]: r + 1
                    for r, i in enumerate(bm25_top_idx) if 0 <= i < len(chunk_texts)}

    # ── RRF fusion ──
    all_chunks = set(faiss_ranks_union) | set(bm25_ranks)
    rrf = {c: (_rrf_score(faiss_ranks_union.get(c, k_init + 1)) +
               _rrf_score(bm25_ranks.get(c, k_init + 1)))
           for c in all_chunks}
    pool = sorted(rrf, key=rrf.__getitem__, reverse=True)[:k_init]

    # ── Stem keyword boost ──
    query_stems = [w[:6].lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", query)
                   if w.lower() not in _STOPWORDS]
    seen = set(id(c) for c in pool)
    for chunk in chunk_texts:
        if id(chunk) not in seen and any(s in chunk.lower() for s in query_stems):
            pool.append(chunk)
            seen.add(id(chunk))

    # ── BGE Reranker v2-m3 ──
    if len(pool) > 1:
        rerank_pool = pool[:120]
        ce_scores   = reranker.predict([[query, c] for c in rerank_pool])
        ranked      = sorted(zip(ce_scores, rerank_pool), reverse=True)
        top_sim     = float(1.0 / (1.0 + math.exp(-ranked[0][0])))
        ce_pool     = [c for _, c in ranked[:k_final * 3]]
    else:
        ce_pool, top_sim = pool[:k_final], 0.5

    # ── MMR diversity (lambda=0.70) ──
    if len(ce_pool) > k_final:
        c_embs   = embedder.encode(ce_pool, normalize_embeddings=True)
        selected, remaining = [], list(range(len(ce_pool)))
        for _ in range(k_final):
            if not remaining: break
            if not selected:
                scores = [_cos(main_q_emb, c_embs[i]) for i in remaining]
            else:
                scores = [0.70 * _cos(main_q_emb, c_embs[i])
                          - 0.30 * max(_cos(c_embs[i], c_embs[j]) for j in selected)
                          for i in remaining]
            best = remaining[int(np.argmax(scores))]
            selected.append(best)
            remaining.remove(best)
        final_chunks = [ce_pool[i] for i in selected]
    else:
        final_chunks = ce_pool[:k_final]

    # ── Build metadata-tagged context string (Layer 2) ──
    parts = []
    for ch in final_chunks:
        try:
            idx = chunk_texts.index(ch)
            sec = chunk_meta[idx]["section"]
        except ValueError:
            sec = "General"
        parts.append(f"[Source: {sec}]\n{ch}")
    context_str = "\n\n---\n\n".join(parts)
    return context_str, final_chunks, top_sim


# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Philippine maritime labor law assistant specializing in "
    "the POEA Standard Employment Contract (SEC) for Filipino seafarers.\n"
    "Each context block begins with [Source: <section>] indicating which part of the "
    "contract it comes from. Use this metadata to cite the correct section in your answer.\n"
    "RULES:\n"
    "1. Answer ONLY using information explicitly stated in the provided context.\n"
    "2. NEVER say: 'I cannot find', 'not mentioned', 'context does not contain', "
    "   'not in context', 'not provided'. These phrases are forbidden.\n"
    "3. Cite the source section (e.g. 'Under Medical Benefits...') when visible.\n"
    "4. If the context gives partial information, use it to construct the best possible answer."
)

STRICT_PROMPT = (
    "You are a strict POEA SEC legal assistant. "
    "Use ONLY sentences directly verifiable in the provided context blocks. "
    "Each block starts with [Source: section]. Cite the section in your answer. "
    "No external knowledge. No speculation."
)

def _faith_inline(answer: str, chunks: list) -> float:
    if not answer.strip() or not chunks: return 0.0
    sents  = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 12]
    c_embs = embedder.encode(chunks, normalize_embeddings=True)
    if not sents:
        a_emb = embedder.encode([answer], normalize_embeddings=True)[0]
        return float(max(_cos(a_emb, c) for c in c_embs))
    return float(np.mean([
        max(_cos(embedder.encode([s], normalize_embeddings=True)[0], c) for c in c_embs)
        for s in sents
    ]))


def generate_answer(query: str) -> dict:
    if not query or not query.strip():
        return {"answer": "Please enter a valid question.", "context": "",
                "chunks": [], "latency_sec": 0.0, "confidence": 0.0}
    t_start = time.time()
    try:
        context, chunks, confidence = retrieve_context(query)
    except Exception as e:
        return {"answer": f"Retrieval error: {e}", "context": "", "chunks": [],
                "latency_sec": 0.0, "confidence": 0.0}

    def _call(q_text: str, sys_p: str) -> str:
        msgs   = [{"role": "system", "content": sys_p},
                  {"role": "user", "content": (
                      f"Context from official POEA documents:\n---\n{context}\n---\n\n"
                      f"Question: {q_text}\n\nAnswer:")}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=3072).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    try:
        answer = _call(query, SYSTEM_PROMPT)
    except Exception as e:
        return {"answer": f"Generation error: {e}", "context": context,
                "chunks": chunks, "latency_sec": 0.0, "confidence": confidence}

    # Layer 5: retry gate — only if faithfulness improves by >= 0.05
    faith_orig = _faith_inline(answer, chunks)
    if faith_orig < 0.35:
        try:
            alt       = _call(query, STRICT_PROMPT)
            faith_alt = _faith_inline(alt, chunks)
            if faith_alt >= faith_orig + 0.05:
                answer = alt
        except Exception:
            pass

    latency = round(time.time() - t_start, 2)
    return {"answer": answer, "context": context, "chunks": chunks,
            "latency_sec": latency, "confidence": confidence}

print(f"Pipeline ready — {MODEL_LABEL}")

# ── Gradio UI ─────────────────────────────────────────────────────────────────
import gradio as gr

def chat_fn(user_message, history):
    if not user_message or not user_message.strip():
        history.append((user_message, "⚠️ Please enter a valid question."))
        return history, "No query provided."
    if len(user_message.strip()) < 5:
        history.append((user_message, "⚠️ Question too short."))
        return history, "Query too short."
    result  = generate_answer(user_message.strip())
    answer  = result["answer"]
    context = result["context"]
    latency = result["latency_sec"]
    history.append((user_message, f"{answer}\n\n⏱️ {latency}s"))
    return history, f"📄 **Retrieved Context (Top 5):**\n\n{context}" if context else "No context retrieved."

def clear_fn():
    return [], ""

with gr.Blocks(title=f"SeamanBot — {MODEL_DISPLAY_NAME}", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # ⚓ SeamanBot — {MODEL_DISPLAY_NAME}
    ### RAG + CrossEncoder Reranker | POEA Maritime Labor Law
    Ask any question about the **POEA Standard Employment Contract for Filipino Seafarers**.
    """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Chat", height=500)
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="e.g. What are a seafarer's medical benefits if injured?",
                    label="Question", lines=2, scale=5
                )
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Send ▶", variant="primary")
                    clear_btn  = gr.Button("Clear")
        with gr.Column(scale=2):
            context_box = gr.Markdown(value="*Retrieved context will appear here.*")
    gr.Markdown("---\n**Example:** *What is the maximum contract duration?* | *Can a seafarer claim overtime?* | *What happens on death during contract?*")
    submit_btn.click(fn=chat_fn, inputs=[user_input, chatbot],
                     outputs=[chatbot, context_box]).then(lambda: "", outputs=user_input)
    user_input.submit(fn=chat_fn, inputs=[user_input, chatbot],
                      outputs=[chatbot, context_box]).then(lambda: "", outputs=user_input)
    clear_btn.click(fn=clear_fn, outputs=[chatbot, context_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
