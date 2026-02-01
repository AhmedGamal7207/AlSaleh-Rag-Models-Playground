import os
import json
from typing import List, Dict, Generator
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import sys

# Configuration
# "intfloat/multilingual-e5-large" is State-of-the-Art for multilingual retrieval
# It maps query/passage to 1024-dim vectors. 
# Requires "passage: " prefix for docs and "query: " for queries.
MODEL_NAME = "intfloat/multilingual-e5-large" 
VECTOR_DIM = 1024 # Dimension for e5-large
COLLECTION_NAME = "legal_documents"
DB_PATH = "./qdrant_db" # Local storage path
BATCH_SIZE = 32 # Increased slightly as text processing is lighter now

class VectorDBBuilder:
    def __init__(self, model_name: str, db_path: str, collection_name: str, vector_dim: int):
        self.model_name = model_name
        self.db_path = db_path
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        
        print(f"Loading embedding model: {self.model_name}...")
        self.encoder = SentenceTransformer(self.model_name)
        
        print(f"Initializing Qdrant at: {self.db_path}")
        self.client = QdrantClient(path=self.db_path)
        
        self._init_collection()

    def _init_collection(self):
        """Initialize or recreate the Qdrant collection."""
        # Check if collection exists
        if self.client.collection_exists(self.collection_name):
            print(f"Collection '{self.collection_name}' exists. It will be appended to.")
            # Note: For a clean rebuild, one might want to delete it first. 
            # self.client.delete_collection(self.collection_name)
        else:
            print(f"Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_dim,
                    distance=models.Distance.COSINE
                )
            )

    def read_processed_files(self, source_dir: str) -> Generator[Dict, None, None]:
        """
        Yields structured chunks from the processed local JSONL files.
        """
        files = [f for f in os.listdir(source_dir) if f.endswith('.jsonl')]
        total_files = len(files)
        print(f"Found {total_files} JSONL files to process.")

        for file_idx, file_name in enumerate(files):
            file_path = os.path.join(source_dir, file_name)
            
            try:
                print(f"Reading {file_name}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            # Parse JSON line
                            chunk_data = json.loads(line)
                            yield chunk_data
                        except json.JSONDecodeError as e:
                            print(f"Error decoding JSON at line {line_num} in {file_name}: {e}")
                            
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

            if (file_idx + 1) % 1 == 0:
                print(f"Finished file {file_idx + 1}/{total_files}...", end='\r')

    def embed_and_upsert(self, source_dir: str):
        """
        Main loop to read chunks, generate vectors, and upload to Qdrant.
        """
        batch_docs = []
        batch_texts = []
        
        print("Starting ingestion...")
        
        # Generator for all chunks
        chunk_stream = self.read_processed_files(source_dir)
        
        count = 0
        
        for doc in chunk_stream:
            # Prepare text for e5 model (passage prefix)
            # Use 'vector_text' which contains the rich context
            formatted_text = f"passage: {doc['vector_text']}"
            
            batch_docs.append(doc)
            batch_texts.append(formatted_text)
            
            if len(batch_docs) >= BATCH_SIZE:
                self._process_batch(batch_docs, batch_texts)
                count += len(batch_docs)
                print(f"Ingested {count} chunks...", end='\r')
                batch_docs = []
                batch_texts = []
        
        # Process remaining
        if batch_docs:
            self._process_batch(batch_docs, batch_texts)
            count += len(batch_docs)
        
        print(f"\nFinished ingesting {count} chunks into '{self.collection_name}'.")

    def _process_batch(self, docs: List[Dict], texts: List[str]):
        """Generate vectors and upload batch."""
        try:
            # Generate embeddings
            embeddings = self.encoder.encode(
                texts, 
                normalize_embeddings=True, 
                show_progress_bar=False,
                batch_size=len(texts)
            )
            
            # Prepare points for Qdrant
            points = []
            for i, doc in enumerate(docs):
                # Use the pre-calculated deterministic hash ID
                point_id = doc["id"]
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=doc["payload"]
                ))
            
            # Upload
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
        except Exception as e:
            print(f"Error processing batch: {e}")

def main():
    source_dir = "processed_docs"
    
    if not os.path.exists(source_dir):
        print(f"Error: Directory {source_dir} not found.")
        print("Please run json_to_rag_chunks.py first.")
        return

    # Initialize Builder
    builder = VectorDBBuilder(
        model_name=MODEL_NAME,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        vector_dim=VECTOR_DIM # 1024 for e5-large
    )
    
    # Run Ingestion
    builder.embed_and_upsert(source_dir)
    
    # Verification
    info = builder.client.get_collection(COLLECTION_NAME)
    print("\nCollection Info:")
    print(f"- Vectors Count: {info.points_count}")
    print(f"- Status: {info.status}")

if __name__ == "__main__":
    main()
