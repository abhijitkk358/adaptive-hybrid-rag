"""
sparse_retriever.py - Implements BM25 sparse keyword retrieval.
Handles exact term matches, acronyms, and technical keyword queries.
"""

import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi


class SparseRetriever:
    def __init__(self, documents: List[Dict[str, Any]]):
        self.documents = documents
        self.corpus_ids = [doc["id"] for doc in documents]
        self.tokenized_corpus = [self._tokenize(doc["title"] + " " + doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text: str) -> List[str]:
        """Simple lowercase word tokenization."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Executes BM25 scoring for a query.
        Returns a list of tuples: (document_id, bm25_score) sorted in descending order.
        """
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        scored_docs = list(zip(self.corpus_ids, [float(s) for s in scores]))
        
        # Sort descending by score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

    def add_document(self, document: Dict[str, Any]):
        """Dynamically appends a document to the index."""
        self.documents.append(document)
        self.corpus_ids.append(document["id"])
        self.tokenized_corpus.append(self._tokenize(document["title"] + " " + document["content"]))
        self.bm25 = BM25Okapi(self.tokenized_corpus)
