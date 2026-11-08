#!/usr/bin/env python3
"""
main.py - Adaptive Hybrid-RAG Engine with Cross-Encoder Reranking & Evaluation
FastAPI REST API serving the entire multi-stage retrieval, reranking, and evaluation pipeline.
"""

import os
import json
import time
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from groq import Groq
except ImportError:
    Groq = None

from engine.sparse_retriever import SparseRetriever
from engine.dense_retriever import DenseRetriever
from engine.hybrid_fusion import reciprocal_rank_fusion
from engine.reranker import CrossEncoderReranker
from engine.evaluator import GroundingEvaluator

load_dotenv()

app = FastAPI(
    title="Adaptive Hybrid-RAG & Evaluation Engine",
    description="High-precision retrieval combining BM25, dense vector search, Cross-Encoder reranking, and automated hallucination evaluation.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Global State & Index Initialization
# -----------------------------------------------------------------------------
CORPUS_PATH = Path(__file__).parent / "data" / "corpus.json"
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    corpus_documents = json.load(f)

# Document lookup map by ID
doc_store: Dict[str, Dict[str, Any]] = {doc["id"]: doc for doc in corpus_documents}

print("[INIT] Loading Sparse BM25 Index...")
sparse_index = SparseRetriever(corpus_documents)

print("[INIT] Loading Dense Vector Bi-Encoder (all-MiniLM-L6-v2)...")
dense_index = DenseRetriever(corpus_documents)

print("[INIT] Loading Cross-Encoder Re-ranker (ms-marco-MiniLM-L-6-v2)...")
reranker = CrossEncoderReranker()

print("[INIT] Loading Grounding & Hallucination Evaluator...")
evaluator = GroundingEvaluator(dense_model=dense_index.model)

# Groq Client setup (Optional key for real LLM synthesis)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if (Groq and GROQ_API_KEY and not GROQ_API_KEY.startswith("gsk_your")) else None


# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str = Field(..., example="How does sliding-window counter rate limiting work?")
    use_hybrid: bool = Field(True, description="Enable BM25 + Dense vector Reciprocal Rank Fusion")
    use_reranker: bool = Field(True, description="Apply Cross-Encoder re-ranking to top candidate chunks")
    top_k_candidates: int = Field(8, ge=2, le=20, description="Candidate chunks retrieved by hybrid search")
    top_n_final: int = Field(3, ge=1, le=5, description="Final context chunks passed to generator")


class DocumentIngestRequest(BaseModel):
    id: str = Field(..., example="doc_011")
    title: str = Field(..., example="Consistent Hashing with Virtual Nodes")
    content: str = Field(..., example="Consistent hashing maps both servers and keys onto a circular ring...")
    category: Optional[str] = "Distributed Systems"


# -----------------------------------------------------------------------------
# LLM Generation Helper
# -----------------------------------------------------------------------------
def generate_grounded_answer(query: str, context_chunks: List[str]) -> str:
    """Generates an answer strictly grounded in the retrieved context."""
    formatted_context = "\n\n".join([f"[Context Chunk {i+1}]: {c}" for i, c in enumerate(context_chunks)])

    if groq_client:
        system_prompt = (
            "You are an expert technical AI assistant. "
            "Your task is to answer the user query STRICTLY based on the provided technical context chunks. "
            "Rules:\n"
            "1. Only state facts directly present in the context.\n"
            "2. If the context does not contain enough info, state clearly: 'The provided documentation does not contain this information.'\n"
            "3. Be concise, precise, and highly technical."
        )
        user_message = f"User Query: {query}\n\nRetrieved Context:\n{formatted_context}\n\nAnswer:"
        
        try:
            completion = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1,
                max_tokens=300
            )
            return completion.choices[0].message.content.strip()
        except Exception as exc:
            print(f"[WARNING] Groq API call failed ({exc}), falling back to extractive synthesis.")

    # Extractive Fallback Synthesis if Groq is not configured
    return (
        f"Based on the verified technical documentation: {context_chunks[0]} "
        f"Furthermore, {context_chunks[1] if len(context_chunks) > 1 else ''}"
    ).strip()


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Adaptive Hybrid-RAG & Evaluation Engine",
        "indexed_documents": len(doc_store),
        "groq_live": groq_client is not None
    }


@app.post("/api/query")
def execute_rag_pipeline(req: QueryRequest):
    """
    Executes the Complete Multi-Stage Adaptive RAG Pipeline:
    1. Sparse Retrieval (BM25)
    2. Dense Retrieval (Sentence-Transformers)
    3. Hybrid Fusion (Reciprocal Rank Fusion - RRF)
    4. Cross-Encoder Re-ranking
    5. Grounded Generation (Groq LLaMA 3.1)
    6. Automated Faithfulness & Hallucination Evaluation
    """
    start_time = time.time()

    # Step 1: Sparse BM25 Search
    sparse_candidates = sparse_index.search(req.query, top_k=req.top_k_candidates)

    # Step 2: Dense Vector Search
    dense_candidates = dense_index.search(req.query, top_k=req.top_k_candidates)

    # Step 3: Hybrid Fusion (RRF)
    if req.use_hybrid:
        fused_results = reciprocal_rank_fusion(
            sparse_candidates,
            dense_candidates,
            k=60,
            top_k=req.top_k_candidates
        )
        candidate_ids = [item["doc_id"] for item in fused_results]
    else:
        # Fallback to dense only if hybrid is toggled off
        fused_results = []
        candidate_ids = [doc_id for doc_id, _ in dense_candidates[:req.top_k_candidates]]

    retrieved_docs = [doc_store[doc_id] for doc_id in candidate_ids if doc_id in doc_store]

    # Step 4: Cross-Encoder Re-ranking
    if req.use_reranker:
        reranked_docs = reranker.rerank(req.query, retrieved_docs, top_n=req.top_n_final)
    else:
        reranked_docs = retrieved_docs[:req.top_n_final]

    # Step 5: Grounded Answer Generation
    context_chunks = [f"{doc['title']}: {doc['content']}" for doc in reranked_docs]
    answer = generate_grounded_answer(req.query, context_chunks)

    # Step 6: Automated Hallucination & Faithfulness Evaluation
    eval_result = evaluator.evaluate(req.query, context_chunks, answer)

    total_latency_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "query": req.query,
        "answer": answer,
        "evaluation": eval_result,
        "retrieval_metadata": {
            "total_latency_ms": total_latency_ms,
            "candidates_evaluated": len(retrieved_docs),
            "final_context_chunks_used": len(reranked_docs),
            "hybrid_rrf_enabled": req.use_hybrid,
            "cross_encoder_rerank_enabled": req.use_reranker
        },
        "top_sources": [
            {
                "id": doc["id"],
                "title": doc["title"],
                "category": doc.get("category"),
                "cross_encoder_score": doc.get("cross_encoder_score"),
                "relevance_confidence_pct": doc.get("relevance_confidence_pct"),
                "snippet": doc["content"][:160] + "..."
            }
            for doc in reranked_docs
        ],
        "debug_pipeline_rankings": fused_results[:5]  # Shows sparse vs dense vs RRF rank
    }


@app.get("/api/documents")
def list_documents():
    """Returns all indexed technical knowledge base documents."""
    return {
        "total_documents": len(doc_store),
        "documents": list(doc_store.values())
    }


@app.post("/api/documents", status_code=status.HTTP_201_CREATED)
def add_document(doc: DocumentIngestRequest):
    """Dynamically indexes a new document into both Sparse and Dense indices."""
    if doc.id in doc_store:
        raise HTTPException(status_code=400, detail="Document with this ID already exists.")

    new_doc = doc.dict()
    doc_store[doc.id] = new_doc
    sparse_index.add_document(new_doc)
    dense_index.add_document(new_doc)

    return {
        "status": "success",
        "message": f"Document '{doc.title}' successfully indexed into sparse and dense stores.",
        "id": doc.id
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
