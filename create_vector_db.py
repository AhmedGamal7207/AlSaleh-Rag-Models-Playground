import os
import re
import math
from typing import List, Dict, Generator
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import sys
import uuid

# Configuration
# "intfloat/multilingual-e5-large" is State-of-the-Art for multilingual retrieval
# It maps query/passage to 1024-dim vectors. 
# Requires "passage: " prefix for docs and "query: " for queries.
MODEL_NAME = "intfloat/multilingual-e5-large" 
VECTOR_DIM = 1024 # Dimension for e5-large
COLLECTION_NAME = "legal_documents"
DB_PATH = "./qdrant_db" # Local storage path
BATCH_SIZE = 16 # Adjust based on VRAM/RAM

class VectorDBBuilderFixed:
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

    def extract_metadata_from_chunk(self, chunk_text: str) -> Dict[str, Any]:
        """
        Naive extraction of metadata from the chunk text itself (Header).
        In the fixed pipeline, metadata is encoded in the text header.
        We can use Regex to pull it out for the payload.
        """
        metadata = {}
        
        # Extract patterns
        # Decision Name: Decision: <...>
        # Address: Address: <...>
        # Categories: Categories: <...> (We added this in json_to_rag_chunks.py)
        
        # Arabic regex patterns based on header format:
        # header = f"وثيقة رقم: {e_id}\n"
        # header += f"القرار: {name}\n"
        # header += f"العنوان: {address}\n"
        # header += f"التصنيف: {cat_str}\n" 
        
        id_match = re.search(r"وثيقة رقم:\s*(.*)", chunk_text)
        if id_match: metadata['doc_id'] = id_match.group(1).strip()
        
        dec_match = re.search(r"القرار:\s*(.*)", chunk_text)
        if dec_match: metadata['law_name'] = dec_match.group(1).strip()
        
        cat_match = re.search(r"التصنيف:\s*(.*)", chunk_text)
        if cat_match:
            cat_str = cat_match.group(1).strip()
            # Convert string list "['A', 'B']" or plain "A, B" to list
            # Usually safe to just store as list of strings
            # If it looks like a python list string, eval it safely or parse it
            if cat_str.startswith('[') and cat_str.endswith(']'):
                try:
                    # simplistic parse
                    metadata['categories'] = eval(cat_str) 
                except:
                    metadata['categories'] = [cat_str]
            else:
                metadata['categories'] = [c.strip() for c in cat_str.split(',') if c.strip()]

        # We can also store the Law/header info
        # But crucially we need categories for the filter to work
        return metadata

    def read_processed_files(self, source_dir: str) -> Generator[Dict, None, None]:
        """
        Yields structured chunks from the processed text files.
        """
        files = [f for f in os.listdir(source_dir) if f.endswith('.txt')]
        total_files = len(files)
        print(f"Found {total_files} files to process.")

        for file_idx, file_name in enumerate(files):
            file_path = os.path.join(source_dir, file_name)
            
            # Extract basic doc ID from filename (e.g. 1081.txt -> 1081)
            doc_id_base = os.path.splitext(file_name)[0]
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Split content into chunks based on separator
                # Regex looks for "--- CHUNK n ---"
                raw_chunks = re.split(r'--- CHUNK \d+ ---\n', content)
                
                # Filter out empty strings from split
                chunks = [c.strip() for c in raw_chunks if c.strip()]
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    # We treat the entire chunk text (Header + Content) as the 'text' to embed,
                    # as the header contains critical context (Title, Law Name).
                    # 'e5' models expect "passage: " prefix for optimal performance.
                    
                    # Extract metadata for Payload
                    extracted_meta = self.extract_metadata_from_chunk(chunk_text)
                    
                    yield {
                        "id": f"{doc_id_base}_{chunk_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "source_file": file_name,
                            "doc_id": doc_id_base,
                            "chunk_index": chunk_idx,
                            "text_content": chunk_text, # Store text in payload for retrieval display
                            **extracted_meta
                        }
                    }
                    
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

            if (file_idx + 1) % 100 == 0:
                print(f"Processed {file_idx + 1}/{total_files} files...", end='\r')

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
            # The model requires 'passage: ' before the text for indexing tasks
            formatted_text = f"passage: {doc['text']}"
            
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
                batch_size=len(texts) # Since we strictly control batch size in the loop
            )
            
            # Prepare points for Qdrant
            points = []
            for i, doc in enumerate(docs):
                # UUID generation
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc["id"]))
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=embeddings[i].tolist(),
                    payload=doc["metadata"]
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
        return

    # Initialize Builder
    builder = VectorDBBuilderFixed(
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
