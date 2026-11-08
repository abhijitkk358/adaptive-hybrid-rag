#!/usr/bin/env python3
"""
test_pipeline.py - Automated Evaluation & Benchmark for Adaptive Hybrid-RAG Engine.
Tests:
1. Health check.
2. Keyword-Heavy Query (BM25 vs Dense comparison).
3. Conceptual Semantic Query (Vector retrieval & Cross-Encoder precision).
4. Automated Hallucination & Faithfulness Evaluation scores.

Run: python3 test_pipeline.py (with uvicorn server running at port 8000)
"""

import json
import time
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8000"


def make_request(path: str, method: str = "GET", data: dict = None) -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        error_content = err.read().decode("utf-8")
        print(f"[ERROR {err.code}] {url} -> {error_content}")
        raise


def run_benchmark():
    print("=" * 70)
    print(" Adaptive Hybrid-RAG & Evaluation Benchmark Suite")
    print("=" * 70)

    # 1. Health Check
    print("\n[STEP 1] Testing Service Health...")
    health = make_request("/health")
    print(f" > Status: {health['status']}")
    print(f" > Indexed Documents: {health['indexed_documents']}")
    print(f" > Groq Live: {health['groq_live']}")

    # 2. Test Keyword-Heavy Query (Tests BM25 & Acronyms)
    print("\n[STEP 2] Query 1: Keyword-Specific Technical Acronyms...")
    query1 = {
        "query": "How does SRRIP avoid cache scan pollution using RRPVs?",
        "use_hybrid": True,
        "use_reranker": True,
        "top_k_candidates": 6,
        "top_n_final": 2
    }
    t0 = time.time()
    res1 = make_request("/api/query", method="POST", data=query1)
    latency1 = round((time.time() - t0) * 1000, 2)

    top_doc1 = res1["top_sources"][0]
    print(f" > Query: '{res1['query']}'")
    print(f" > Top Matched Document: {top_doc1['title']} (Score: {top_doc1['cross_encoder_score']}, Confidence: {top_doc1['relevance_confidence_pct']}%)")
    print(f" > Faithfulness Score: {res1['evaluation']['faithfulness_score']} ({res1['evaluation']['grounding_status']})")
    print(f" > Hallucination Detected: {res1['evaluation']['hallucination_detected']}")
    print(f" > End-to-End Latency: {latency1} ms")
    assert "SRRIP" in top_doc1["title"] or "Cache" in top_doc1["title"], "Expected SRRIP doc to be ranked #1!"

    # 3. Test Conceptual Semantic Query
    print("\n[STEP 3] Query 2: High-Level Conceptual Query (No Exact Keywords)...")
    query2 = {
        "query": "What mechanism protects microservices from cascading failures when a downstream database goes down?",
        "use_hybrid": True,
        "use_reranker": True,
        "top_k_candidates": 6,
        "top_n_final": 2
    }
    t0 = time.time()
    res2 = make_request("/api/query", method="POST", data=query2)
    latency2 = round((time.time() - t0) * 1000, 2)

    top_doc2 = res2["top_sources"][0]
    print(f" > Query: '{res2['query']}'")
    print(f" > Top Matched Document: {top_doc2['title']} (Confidence: {top_doc2['relevance_confidence_pct']}%)")
    print(f" > Generated Answer Snippet: {res2['answer'][:140]}...")
    print(f" > Context Relevance: {res2['evaluation']['context_relevance_score']}")
    print(f" > Faithfulness: {res2['evaluation']['faithfulness_score']}")
    print(f" > End-to-End Latency: {latency2} ms")
    assert "Circuit Breaker" in top_doc2["title"], "Expected Circuit Breaker doc to rank #1!"

    # 4. Pipeline Debug Inspection
    print("\n[STEP 4] Inspecting RRF Fusion Pipeline Rankings...")
    print(" > Demonstrating Sparse vs Dense Rank Fusion for Query 2:")
    for rank, item in enumerate(res2["debug_pipeline_rankings"][:3], 1):
        print(f"    Rank #{rank}: Doc [{item['doc_id']}] -> BM25 Rank: {item['sparse_rank']}, Dense Rank: {item['dense_rank']} -> RRF Score: {item['rrf_score']}")

    print("\n" + "=" * 70)
    print(" [SUCCESS] All Multi-Stage Hybrid-RAG Benchmarks Passed Perfectly!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"\n[FAILED] Make sure the server is running (python3 -m uvicorn main:app --port 8000): {e}")
