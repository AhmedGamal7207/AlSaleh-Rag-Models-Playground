import os
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# Configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
# No Reranker Model
# RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

DB_PATH = "./qdrant_db"
COLLECTION_NAME = "legal_documents"

class RAGRetrieverNoReranker:
    """
    RAG Retriever for Egyptian Legal Laws (No Reranker Version).
    - Single-stage retrieval: Vector Search Only.
    - Handles merging of sub-chunks into coherent article results.
    """

    def __init__(self, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
        print("Initializing Retrieval Pipeline (Vector Only)...")
        
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # No Reranker Initialization

    def get_query_embedding(self, query: str) -> List[float]:
        formatted_query = f"query: {query}"
        return self.encoder.encode(formatted_query, normalize_embeddings=True).tolist()

    def vector_search(self, query_vector: List[float], top_k: int = 50) -> List[Dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        # Normalize keys for the pipeline
        return [
            {
                "id": point.id,
                "score": point.score, # Use vector score as the main score
                "payload": point.payload
            }
            for point in results.points
        ]

    def group_results(self, results: List[Dict]) -> List[Dict]:
        """
        Group sub-chunks back into unique articles.
        """
        grouped = {}
        
        for res in results:
            # article_group_id is lawId_articleNum
            group_id = res['payload'].get('article_group_id')
            
            # Fallback for old data or missing keys
            if not group_id:
                group_id = res['id'] 
            
            if group_id not in grouped:
                grouped[group_id] = {
                    "best_score": res['score'],
                    "primary_chunk": res,
                    "all_chunks": [res]
                }
            else:
                grouped[group_id]["all_chunks"].append(res)
                # Keep the chunk with the highest similarity score
                if res['score'] > grouped[group_id]["best_score"]:
                    grouped[group_id]["best_score"] = res['score']
                    grouped[group_id]["primary_chunk"] = res
        
        # Flatten back to list
        final_output = []
        for g_id, data in grouped.items():
            primary = data['primary_chunk']
            
            final_output.append({
                "score": data["best_score"],
                "payload": primary["payload"],
                "group_count": len(data["all_chunks"])
            })
            
        # Re-sort by best score
        final_output.sort(key=lambda x: x['score'], reverse=True)
        return final_output

    def retrieve(self, query: str, top_k: int = 5, search_k: int = 50) -> List[Dict]:
        # 1. Dense Search
        query_vector = self.get_query_embedding(query)
        
        # We fetch search_k results to group them, then slice to top_k
        candidates = self.vector_search(query_vector, top_k=search_k)
        
        # 2. No Reranking Step
        
        # 3. Group by Article
        grouped = self.group_results(candidates)
        
        return grouped[:top_k]

if __name__ == "__main__":
    retriever = RAGRetrieverNoReranker()
    # Test query
    results = retriever.retrieve("ما هي عقوبة السرقة بالإكراه؟", top_k=3)
    
    print(f"\nFound {len(results)} relevant articles (Vector Only):\n")
    for i, res in enumerate(results):
        p = res['payload']
        print(f"{i+1}. المادة: {p.get('article_title', 'بدون')} (Score: {res['score']:.4f})")
        print(f"   القانون: {p.get('law_name', '')}")
        print(f"   الحالة: {p.get('status', '')}")
        print(f"   النص: {p.get('text_content', '')[:150]}...")
        print("-" * 30)
