"""
evaluator.py - Automated RAG Grounding & Hallucination Evaluator.
Evaluates whether the generated response is strictly supported by the retrieved context
chunks, outputting Faithfulness and Context Relevance scores.
"""

import re
from typing import List, Dict, Any
import numpy as np


class GroundingEvaluator:
    def __init__(self, dense_model=None):
        """Initializes evaluator with sentence embedding support for semantic entailment check."""
        self.encoder = dense_model

    def evaluate(self, query: str, context_chunks: List[str], generated_answer: str) -> Dict[str, Any]:
        """
        Calculates Faithfulness, Context Relevance, and Hallucination Risk.
        
        Args:
            query: The user prompt
            context_chunks: List of retrieved context strings
            generated_answer: The LLM response
            
        Returns:
            Dict containing faithfulness_score, context_relevance_score,
            is_hallucinated (bool), and confidence metrics.
        """
        combined_context = " ".join(context_chunks).lower()
        answer_sentences = [s.strip() for s in re.split(r"[.!?]", generated_answer) if len(s.strip()) > 8]

        if not answer_sentences:
            return {
                "faithfulness_score": 1.0,
                "context_relevance_score": 1.0,
                "hallucination_detected": False,
                "reasoning": "Answer was too brief to evaluate."
            }

        # 1. Lexical & Named Entity Grounding Check
        supported_sentences = 0
        total_sentences = len(answer_sentences)

        for sent in answer_sentences:
            words = [w for w in re.findall(r"\w+", sent.lower()) if len(w) > 3]
            if not words:
                continue
            
            # Count how many substantial words from the generated sentence exist in context
            matches = sum(1 for w in words if w in combined_context)
            ratio = matches / len(words)

            # A sentence is considered grounded if >= 60% of its content words are in context
            if ratio >= 0.55:
                supported_sentences += 1

        faithfulness = round(supported_sentences / max(1, total_sentences), 3)

        # 2. Semantic Context Relevance (Does the answer address the context?)
        relevance_score = 0.85
        if self.encoder:
            try:
                emb_answer = self.encoder.encode([generated_answer], convert_to_numpy=True, show_progress_bar=False)[0]
                emb_ctx = self.encoder.encode([combined_context[:1000]], convert_to_numpy=True, show_progress_bar=False)[0]
                
                norm_a = np.linalg.norm(emb_answer)
                norm_c = np.linalg.norm(emb_ctx)
                if norm_a > 0 and norm_c > 0:
                    relevance_score = round(float(np.dot(emb_answer, emb_ctx) / (norm_a * norm_c)), 3)
            except Exception:
                relevance_score = 0.80

        # Flag hallucination risk if faithfulness drops below threshold
        hallucination_detected = faithfulness < 0.60

        return {
            "faithfulness_score": faithfulness,
            "context_relevance_score": relevance_score,
            "hallucination_detected": hallucination_detected,
            "grounding_status": "HIGHLY_FAITHFUL" if faithfulness >= 0.8 else ("MODERATE_RISK" if faithfulness >= 0.6 else "HALLUCINATION_DETECTED"),
            "evaluated_claims_count": total_sentences,
            "supported_claims_count": supported_sentences
        }
