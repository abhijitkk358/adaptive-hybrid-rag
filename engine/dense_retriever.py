"""
dense_retriever.py - Implements dense vector retrieval via SentenceTransformers.
Encodes text into 384-dimensional dense vectors and performs cosine similarity search.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, documents: List[Dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.documents = documents
        self.corpus_ids = [doc["id"] for doc in documents]
        self.model = SentenceTransformer(model_name)
        
        # Precompute normalized embeddings for the corpus
        corpus_texts = [f"{doc['title']}: {doc['content']}" for doc in documents]
        raw_embeddings = self.model.encode(corpus_texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Normalize to unit length for fast dot-product cosine similarity
        self.embeddings = raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Encodes query and calculates cosine similarity across corpus vectors.
        Returns a list of tuples: (document_id, cosine_score) sorted descending.
        """
        query_vec = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []
        
        query_vec = query_vec / query_norm
        
        # Dot product with normalized vectors equals cosine similarity
        similarities = np.dot(self.embeddings, query_vec)
        scored_docs = list(zip(self.corpus_ids, [float(s) for s in similarities]))
        
        # Sort descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def add_document(self, document: Dict[str, Any]):
        """Dynamically appends a document and re-indexes embeddings."""
        self.documents.append(document)
        self.corpus_ids.append(document["id"])
        
        text = f"{document['title']}: {document['content']}"
        new_vec = self.model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
        new_vec = (new_vec / np.linalg.norm(new_vec)).reshape(1, -1)
        
        self.embeddings = np.vstack([self.embeddings, new_vec])
