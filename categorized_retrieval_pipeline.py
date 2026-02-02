import os
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer, CrossEncoder

# Reuse configuration
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"
DB_PATH = "./qdrant_db"
COLLECTION_NAME = "legal_documents"

class CategorizedRAGRetriever:
    """
    Categorized RAG Retriever.
    - Filters search space by metadata category BEFORE vector search.
    - Two-stage retrieval: Filtered Vector Search + Cross-Encoder Reranking.
    """

    def __init__(self, db_path: str = DB_PATH, collection_name: str = COLLECTION_NAME):
        print("Initializing Categorized Retrieval Pipeline...")
        
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)
        
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        print(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME, max_length=512)

    def get_query_embedding(self, query: str) -> List[float]:
        formatted_query = f"query: {query}"
        return self.encoder.encode(formatted_query, normalize_embeddings=True).tolist()

    def vector_search(self, query_vector: List[float], category: str = None, top_k: int = 50) -> List[Dict]:
        """
        Perform vector search with optional category filtering.
        """
        query_filter = None
        
        if category:
            # Create a filter to checking if 'categories' list in payload contains the value 'category'
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="categories",
                        match=models.MatchAny(any=[category])
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter, 
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
        if not initial_results:
            return []
            
        pairs = []
        for res in initial_results:
            p = res['payload']
            law_name = p.get('law_name', '')
            details = p.get('article_title', '')
            status = p.get('status', '')
            content = p.get('text_content', '')
            
            rich_text = f"القانون: {law_name}\nالمادة: {details} ({status})\nنص المادة: {content}"
            pairs.append([query, rich_text])
            
        scores = self.reranker.predict(pairs)
        
        for i, res in enumerate(initial_results):
            res['rerank_score'] = float(scores[i])
            
        sorted_results = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)
        return sorted_results[:top_k]

    def group_results(self, results: List[Dict]) -> List[Dict]:
        grouped = {}
        
        for res in results:
            group_id = res['payload'].get('article_group_id')
            if not group_id:
                group_id = res['id'] 
            
            if group_id not in grouped:
                grouped[group_id] = {
                    "best_score": res['rerank_score'],
                    "primary_chunk": res,
                    "all_chunks": [res]
                }
            else:
                grouped[group_id]["all_chunks"].append(res)
                if res['rerank_score'] > grouped[group_id]["best_score"]:
                    grouped[group_id]["best_score"] = res['rerank_score']
                    grouped[group_id]["primary_chunk"] = res
        
        final_output = []
        for g_id, data in grouped.items():
            primary = data['primary_chunk']
            final_output.append({
                "score": data["best_score"],
                "payload": primary["payload"],
                "group_count": len(data["all_chunks"])
            })
            
        final_output.sort(key=lambda x: x['score'], reverse=True)
        return final_output

    def retrieve(self, query: str, category: str = None, top_k: int = 5, search_k: int = 50) -> List[Dict]:
        """
        Main retrieval function with category support.
        """
        # 1. Dense Search with Filter
        query_vector = self.get_query_embedding(query)
        candidates = self.vector_search(query_vector, category=category, top_k=search_k)
        
        # 2. Rerank
        reranked = self.rerank(query, candidates, top_k=search_k)
        
        # 3. Group
        grouped = self.group_results(reranked)
        
        return grouped[:top_k]

if __name__ == "__main__":
    retriever = CategorizedRAGRetriever()
    # Test
    cat = "القانون الجنائي"
    q = "ما هي عقوبة السرقة بالإكراه؟"
    print(f"Searching for '{q}' in category '{cat}'...")
    results = retriever.retrieve(q, category=cat, top_k=3)
    
    for i, res in enumerate(results):
        print(f"Result {i+1}: {res['payload'].get('article_title')} (Score: {res['score']:.4f})")
