import os
import re
import uuid
import numpy as np
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder

# Configuration
# Embedding model for Retrieval (Must match what was used for ingestion)
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# Reranking model (High accuracy, processes only top-k candidates)
# 'cross-encoder/ms-marco-MiniLM-L-12-v2' is a good balance, but for multilingual/arabic we need a compatible one.
# 'amberoad/bert-multilingual-passage-reranking-msmarco' is an option, or 'BAAI/bge-reranker-base'.
# Let's use BAAI/bge-reranker-v2-m3 which is SOTA for multilingual including Arabic, or 'BAAI/bge-reranker-base' used commonly.
# For simplicity and size, 'cross-encoder/ms-marco-MiniLM-L-6-v2' is standard but English focused.
# We will use 'BAAI/bge-reranker-base' as it supports multilingual well and is distinct for RAG.
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "legal_documents"

class RAGRetriever:
    """
    Production-Ready Retriever pipeline using Two-Stage Retrieval:
    1. Dense Vector Retrieval (Qdrant) - Fast candidate generation.
    2. Cross-Encoder Reranking - High precision sorting of candidates.
    
    This avoids loading the entire dataset into RAM.
    """

    def __init__(self, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
        print("Initializing Retrieval Pipeline (Production Mode)...")
        
        # 1. Connect to Qdrant (Disk-based)
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        
        # 2. Load Embedding Model (for Query Encoding)
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 3. Load Reranker Model (for Re-scoring)
        print(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)

    def get_query_embedding(self, query: str) -> List[float]:
        """
        Convert query to embedding using the E5 pattern 'query: '.
        """
        formatted_query = f"query: {query}"
        return self.encoder.encode(formatted_query, normalize_embeddings=True).tolist()

    def vector_search(self, query_vector: List[float], top_k: int = 50) -> List[Dict]:
        """
        Perform Dense Vector Search using Qdrant (NEW API).
        """
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
    
        return [
            {
                "id": point.id,
                "initial_score": point.score,
                "payload": point.payload
            }
            for point in results.points
        ]

    def rerank(self, query: str, initial_results: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Re-score the initial vector search results using the Cross-Encoder.
        """
        if not initial_results:
            return []
            
        # Prepare pairs for the Cross-Encoder [Query, Document Text]
        pairs = []
        for res in initial_results:
            doc_text = res['payload'].get('text', '')
            pairs.append([query, doc_text])
            
        # Predict scores (higher is better)
        scores = self.reranker.predict(pairs)
        
        # Attach scores to results
        for i, res in enumerate(initial_results):
            res['rerank_score'] = float(scores[i])
            
        # Sort by new score descending
        sorted_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        
        return sorted_results[:top_k]

    def retrieve(self, query: str, top_k: int = 10, search_k: int = 50) -> List[Dict]:
        """
        The Main Pipeline Function.
        
        Args:
            query (str): The user query.
            top_k (int): Number of final precise results to return.
            search_k (int): Number of candidates to fetch from Vector DB (should be > top_k).
        
        Returns:
            List[Dict]: Ordered list of top documents.
        """
        # Step 1: Retrieval (Dense Vector Search)
        # We fetch more candidates (search_k) than we need (top_k) to allow reranker to find the best ones.
        query_vector = self.get_query_embedding(query)
        candidates = self.vector_search(query_vector, top_k=search_k)
        
        # Step 2: Reranking (Cross-Encoder)
        # Re-sort the candidates based on actual relevance to the query.
        final_results = self.rerank(query, candidates, top_k=top_k)
        
        # Formatting output
        formatted = []
        for rank, res in enumerate(final_results):
            formatted.append({
                "rank": rank + 1,
                "score": res['rerank_score'],
                "initial_score": res['initial_score'],
                "content": res['payload'].get('text', ''),
                "metadata": {k:v for k,v in res['payload'].items() if k != 'text'}
            })
            
        return formatted

if __name__ == "__main__":
    retriever = RAGRetriever()
    results = retriever.retrieve("ما هي عقوبة السرقة؟", top_k=3)
    for res in results:
        print(f"Rank {res['rank']} (Score: {res['score']:.4f})")
        print(res['content'][:100] + "...")
        print("-" * 20)
