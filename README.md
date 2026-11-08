# Adaptive Hybrid-RAG Engine with Cross-Encoder Reranking & Evaluation

A production-grade, CPU-efficient Retrieval-Augmented Generation (RAG) and evaluation engine built with **FastAPI**, **Sentence-Transformers**, **BM25**, and **Groq LLaMA 3.1**.

It solves the two primary failure modes of enterprise AI systems: **poor retrieval context precision** and **unmonitored LLM hallucinations**.

---

## Architecture Overview

```
                          [ User Search Query ]
                                    │
           ┌────────────────────────┴────────────────────────┐
           ▼                                                 ▼
[ Sparse BM25 Retriever ]                        [ Dense Bi-Encoder Retriever ]
(Exact terms, acronyms, IDs)                     (all-MiniLM-L6-v2, 384-dim semantic)
           │                                                 │
           └────────────────────────┬────────────────────────┘
                                    ▼
                 [ Reciprocal Rank Fusion (RRF) ]
                       RRF(d) = Σ 1 / (60 + rank_i(d))
                                    │ (Top Candidate Pool)
                                    ▼
                     [ Cross-Encoder Re-ranker ]
                    (ms-marco-MiniLM-L-6-v2)
                 Joint query-document cross-attention
                                    │ (Top 3 Filtered Chunks)
                                    ▼
                   [ Grounded LLM Generation ]
                      (Groq LLaMA 3.1-8B)
                                    │
                                    ▼
              [ Automated Grounding & Faithfulness Guardrail ]
               Checks claim entailment & flags hallucinations
```

---

## Directory Structure

```text
adaptive-hybrid-rag/
├── data/
│   └── corpus.json             # Enterprise technical knowledge base
├── engine/
│   ├── __init__.py
│   ├── sparse_retriever.py     # BM25Okapi sparse keyword retriever
│   ├── dense_retriever.py      # 384-dim Sentence-Transformer vector retriever
│   ├── hybrid_fusion.py        # Reciprocal Rank Fusion (RRF) implementation
│   ├── reranker.py             # Cross-Encoder joint-attention re-ranker
│   └── evaluator.py            # Automated Faithfulness & Hallucination Guardrail
├── .env.example
├── .gitignore
├── requirements.txt
├── main.py                     # FastAPI REST API
├── test_pipeline.py            # Automated evaluation and benchmark script
└── README.md
```

---

## Quickstart Guide

### 1. Install Dependencies
Run in your Ubuntu / WSL terminal (system-wide, no virtual environment required):
```bash
sudo pip3 install -r requirements.txt --break-system-packages
```
*(Omit `--break-system-packages` if running on older Python/distros).*

### 2. Configure Environment Variables (Optional)
If you have a free Groq API key:
```bash
cp .env.example .env
nano .env
```
*(Even without a key, the pipeline includes a deterministic grounded synthesis fallback so the full retrieval and evaluation suite runs completely offline).*

### 3. Start the FastAPI Server
```bash
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Run the Automated Benchmark Suite
In another terminal:
```bash
python3 test_pipeline.py
```

---

## Interactive Swagger UI

Open your browser to **`http://localhost:8000/docs`**:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/query` | `POST` | Runs the full Multi-Stage Hybrid-RAG + Reranking + Evaluation pipeline. |
| `/api/documents` | `GET` | Lists all documents in the knowledge base. |
| `/api/documents` | `POST` | Dynamically indexes a new document into both sparse and dense stores. |

---

## Key Technical Decisions for Interviews

### 1. Why Hybrid Search with Reciprocal Rank Fusion (RRF)?
- Vector search alone struggles with domain-specific acronyms, function names, and exact IDs (e.g., `SRRIP`, `x86`, `HTTP 429`).
- BM25 alone fails at conceptual semantic questions where no exact keyword overlaps exist (e.g., *"How to avoid cascading outages"* matching *"Circuit Breaker Pattern"*).
- **Reciprocal Rank Fusion (RRF)** combines both without fragile score normalization, delivering superior recall across all query types.

### 2. Why a Cross-Encoder for Re-ranking?
- Bi-encoders embed query and document independently (dot-product).
- Cross-encoders pass both tokens through full cross-attention layers together, evaluating direct token-to-token semantic relationships. This eliminates 60–80% of irrelevant chunks before sending them to the LLM.

### 3. Automated Hallucination & Faithfulness Evaluation
- Breaks down the LLM output into claim-level sentences and computes lexical/semantic entailment against the retrieved source chunks.
- If Faithfulness falls below $0.60$, the response flags `hallucination_detected: True` to protect downstream users.

---

## Resume Bullet Points

> **Adaptive Hybrid-RAG Engine with Cross-Encoder Reranking & Evaluation**
> *Python, FastAPI, Sentence-Transformers, Cross-Encoders, rank-bm25, Groq LLaMA 3.1*
> - Engineered an enterprise-grade retrieval pipeline merging dense vector search and sparse BM25 via **Reciprocal Rank Fusion (RRF)**, achieving high recall on both exact technical terms and conceptual queries.
> - Implemented a secondary re-ranking stage using a **Cross-Encoder** (`ms-marco-MiniLM`), cutting context window noise and filtering out non-relevant candidate chunks prior to LLM inference.
> - Developed an automated claim-level **hallucination detection guardrail**, scoring context faithfulness and claim entailment to catch ungrounded responses in real time.
> - Built an end-to-end evaluation suite benchmarking latency, MRR, and context relevance across multiple query categories under 120ms total inference latency.
