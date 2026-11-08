"""
reranker.py - Implements Cross-Encoder Re-ranking.
Unlike bi-encoders which evaluate query and document in isolation,
cross-encoders process query-document pairs simultaneously with full cross-attention,
producing significantly higher precision rankings.
"""

from typing import List, Dict, Any
import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initializes the lightweight, CPU-efficient cross-encoder model."""
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Takes candidate documents and re-scores them using joint cross-attention.
        
        Args:
            query: The user search query
            candidates: List of document dicts with 'id', 'title', 'content'
            top_n: Number of top re-ranked documents to return
            
        Returns:
            List of candidate dicts with added 'cross_encoder_score' and calibrated 'confidence_pct'.
        """
        if not candidates:
            return []

        # Prepare (query, document) pairs
        pairs = [[query, f"{c['title']}: {c['content']}"] for c in candidates]
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Attach scores
        scored_candidates = []
        for cand, score in zip(candidates, raw_scores):
            entry = dict(cand)
            raw_val = float(score)
            entry["cross_encoder_score"] = round(raw_val, 4)
            # Sigmoid calibration: 1 / (1 + exp(-x)) -> percentage 0 to 100%
            calibrated_prob = 1.0 / (1.0 + np.exp(-raw_val))
            entry["relevance_confidence_pct"] = round(calibrated_prob * 100, 2)
            scored_candidates.append(entry)

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
        return scored_candidates[:top_n]
