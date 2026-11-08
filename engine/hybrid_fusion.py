"""
hybrid_fusion.py - Implements Reciprocal Rank Fusion (RRF).
Merges ranked lists from sparse (BM25) and dense (vector) retrievers
into a single robust candidate list without requiring score normalization.
"""

from typing import List, Dict, Any, Tuple


def reciprocal_rank_fusion(
    sparse_results: List[Tuple[str, float]],
    dense_results: List[Tuple[str, float]],
    k: int = 60,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Combines ranked lists using standard Reciprocal Rank Fusion (RRF).
    Formula: RRF(doc) = SUM(1 / (k + rank_i(doc))) for each retriever i.
    
    Args:
        sparse_results: List of (doc_id, bm25_score)
        dense_results: List of (doc_id, cosine_score)
        k: Smoothing constant (industry standard = 60)
        top_k: Number of unified candidates to return
    
    Returns:
        Sorted list of dicts with doc_id, rrf_score, sparse_rank, dense_rank.
    """
    scores: Dict[str, float] = {}
    metadata: Dict[str, Dict[str, Any]] = {}

    # 1. Process Sparse (BM25) Rankings
    for rank, (doc_id, bm25_score) in enumerate(sparse_results, start=1):
        if doc_id not in scores:
            scores[doc_id] = 0.0
            metadata[doc_id] = {
                "doc_id": doc_id,
                "sparse_rank": rank,
                "sparse_score": round(bm25_score, 4),
                "dense_rank": None,
                "dense_score": None
            }
        scores[doc_id] += 1.0 / (k + rank)
        metadata[doc_id]["sparse_rank"] = rank
        metadata[doc_id]["sparse_score"] = round(bm25_score, 4)

    # 2. Process Dense (Vector) Rankings
    for rank, (doc_id, dense_score) in enumerate(dense_results, start=1):
        if doc_id not in scores:
            scores[doc_id] = 0.0
            metadata[doc_id] = {
                "doc_id": doc_id,
                "sparse_rank": None,
                "sparse_score": None,
                "dense_rank": rank,
                "dense_score": round(dense_score, 4)
            }
        scores[doc_id] += 1.0 / (k + rank)
        metadata[doc_id]["dense_rank"] = rank
        metadata[doc_id]["dense_score"] = round(dense_score, 4)

    # 3. Compile and sort by RRF score descending
    combined = []
    for doc_id, score in scores.items():
        entry = metadata[doc_id]
        entry["rrf_score"] = round(score, 6)
        combined.append(entry)

    combined.sort(key=lambda x: x["rrf_score"], reverse=True)
    return combined[:top_k]
