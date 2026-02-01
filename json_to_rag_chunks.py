import ijson
import os
import json
from typing import Iterator, Dict, Any, List
import hashlib

# Configuration
INPUT_FILES = ["القوانين.json"]
OUTPUT_DIR = "processed_docs"
# intfloat/multilingual-e5-large has a limit of 512 tokens.
# Arabic roughly 1 token ~= 3-4 chars? Safe limit 1000-1200 chars to check for splitting.
# We will use a soft limit to split long articles.
MAX_CHUNK_CHARS = 1000 
OVERLAP_CHARS = 200

class DocumentProcessor:
    """
    A class to process legal documents from JSON and convert them into structured chunks
    (one article per chunk) suitable for RAG embeddings.
    """

    def __init__(self, output_dir: str, max_chunk_chars: int = 1000, overlap: int = 200):
        self.output_dir = output_dir
        self.max_chunk_chars = max_chunk_chars
        self.overlap = overlap
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_json_file(self, file_path: str):
        print(f"Processing file: {file_path}...")
        
        # Output file name based on input
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = os.path.join(self.output_dir, f"{base_name}_chunks.jsonl")
        
        try:
            # We open the output file in write mode to clear previous runs
            with open(file_path, 'rb') as f_in, open(output_file_path, 'w', encoding='utf-8') as f_out:
                documents = ijson.items(f_in, 'item')
                
                count = 0
                article_count = 0
                
                for doc in documents:
                    chunks = self.process_document(doc)
                    for chunk in chunks:
                        f_out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                        article_count += 1
                        
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} documents (generated {article_count} chunks)...", end='\r')
                
                print(f"\nFinished processing {count} documents from {file_path}. Total chunks: {article_count}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            import traceback
            traceback.print_exc()

    def process_document(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single document object into a list of chunk objects.
        """
        chunks_out = []
        
        # 1. Extract Law-Level Metadata
        law_id = str(doc.get("element_id", "unknown"))
        law_metadata = {
            "law_id": law_id,
            "law_name": doc.get("decision_name", ""),
            "law_number": doc.get("decision_number", ""),
            "law_year": doc.get("decision_year", ""),
            "law_date": doc.get("decision_date", ""),
            "law_type": doc.get("type", ""), # e.g. "law", "regulation"
            "law_address": doc.get("law_address", ""),
            "law_status": doc.get("status", ""), # If available at top level
            
            # Additional Categories & Keywords
            "categories": doc.get("categories", []),
            "keywords_level_1": doc.get("keywords_level_1", []),
            "keywords_level_2": doc.get("keywords_level_2", [])
        }

        articles = doc.get("articles", [])
        if not articles:
            return []

        # 2. Process each article
        for art in articles:
            # Extract Article-Level Metadata
            article_number = art.get("article_number", "")
            article_title = art.get("article_title", "")
            
            # Status Logic
            is_canceled = str(art.get("is_canceled", "0"))
            status_str = "ملغاة" if is_canceled == "1" else "سارية"
            
            working_date = art.get("working_date") or "غير محدد"
            canceling_date = art.get("canceling_date") or "غير محدد"
            
            article_metadata = {
                "article_number": article_number,
                "article_title": article_title,
                "status": status_str,
                "is_canceled": is_canceled,
                "working_date": working_date,
                "canceling_date": canceling_date,
                "article_id_local": art.get("article_id", "") # Local DB ID if exists
            }
            
            # Content Logic
            # "If an article is canceled and original content is not null that means that content of it is inside the original content"
            raw_content = art.get("article_content", "")
            original_content = art.get("original_content")
            
            content_text = ""
            if is_canceled == "1" and original_content:
                content_text = original_content
            else:
                content_text = raw_content

            if not content_text:
                content_text = ""
            
            content_text = content_text.strip()
            if not content_text:
                continue # Skip empty articles

            # 3. Handle Chunking (Split if too long)
            text_chunks = self._split_text(content_text)
            
            for i, sub_chunk_text in enumerate(text_chunks):
                # Construct composite ID for the vector
                # Using hash of law_id + article_title to be robust
                unique_str = f"{law_id}_{article_title}_{i}"
                chunk_id_hash = hashlib.md5(unique_str.encode()).hexdigest()
                
                # Shared Article ID (for grouping sub-chunks later)
                # Group by Law + Article Number/Title
                article_unique_id = f"{law_id}_{article_number}"
                
                # Prepare text for embedding (Rich Context)
                # Combine metadata + content into the string that will be vectorized
                # Provide explicit context to the model
                embedding_text = (
                    f"القانون: {law_metadata['law_name']}\n"
                    f"العنوان: {law_metadata['law_address']}\n"
                    f"المادة: {article_title} ({status_str})\n"
                    f"التصنيف: {', '.join(law_metadata['categories']) if isinstance(law_metadata['categories'], list) else str(law_metadata['categories'])}\n"
                    f"نص المادة: {sub_chunk_text}"
                )

                chunk_obj = {
                    "id": chunk_id_hash, # Unique ID for Qdrant Point
                    "payload": {
                        # Metadata Fields
                        "article_group_id": article_unique_id, # For merging results
                        "chunk_index": i,
                        "total_chunks_in_article": len(text_chunks),
                        "text_content": sub_chunk_text, # The actual text part
                        
                        # Law Metadata
                        **law_metadata,
                        
                        # Article Metadata
                        **article_metadata
                    },
                    "vector_text": embedding_text # What we will actually embed
                }
                
                chunks_out.append(chunk_obj)

        return chunks_out

    def _split_text(self, text: str) -> List[str]:
        """
        Split text into chunks if it exceeds MAX_CHUNK_CHARS.
        Try to split on newlines or spaces.
        """
        if len(text) <= self.max_chunk_chars:
            return [text]
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + self.max_chunk_chars, text_len)
            
            if end < text_len:
                # Try to find a safe split point (newline > space)
                # Look back from 'end' to find a split point
                split_point = -1
                
                # Look for newline in the last 100 chars of the window
                for i in range(end, max(start, end - 100), -1):
                    if text[i] == '\n':
                        split_point = i + 1 # Include newline in previous chunk
                        break
                
                # If no newline, look for space
                if split_point == -1:
                    for i in range(end, max(start, end - 50), -1):
                        if text[i] == ' ':
                            split_point = i + 1
                            break
                
                if split_point != -1:
                    end = split_point
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Overlap logic
            # If we split exactly at 'end', next chunk starts at end - overlap
            # But we must ensure we don't go backwards or stay stuck
            if end >= text_len:
                break
                
            start = end - self.overlap
            if start < 0: start = 0
            
            # Avoid infinite loops if overlap is too big / chunk too small
            if start >= end:
                start = end 

        return chunks

def main():
    processor = DocumentProcessor(
        output_dir=OUTPUT_DIR,
        max_chunk_chars=MAX_CHUNK_CHARS,
        overlap=OVERLAP_CHARS
    )
    
    for input_file in INPUT_FILES:
        if os.path.exists(input_file):
            processor.process_json_file(input_file)
        else:
            print(f"Warning: File {input_file} not found.")

if __name__ == "__main__":
    main()
