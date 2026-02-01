import ijson
import os
import json
from typing import Iterator, Dict, Any, List

# Configuration
INPUT_FILES = ["القوانين.json", "اللوائح.json"]
OUTPUT_DIR = "processed_docs"
CHUNK_SIZE = 1500  # Target characters per chunk
OVERLAP_SIZE = 200  # Overlap characters between chunks
SEPARATOR = "\n" + "="*40 + "\n"

class DocumentProcessor:
    """
    A class to process legal documents from JSON and convert them into text chunks
    suitable for RAG embeddings.
    """

    def __init__(self, output_dir: str, chunk_size: int = 1500, overlap: int = 200):
        """
        Initialize the processor.

        Args:
            output_dir (str): Directory to save output .txt files.
            chunk_size (int): approximate size of each text chunk in characters.
            overlap (int): number of characters to overlap between chunks.
        """
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def process_json_file(self, file_path: str):
        """
        Stream process a large JSON file containing a list of documents.
        
        Args:
            file_path (str): Path to the JSON file.
        """
        print(f"Processing file: {file_path}...")
        
        try:
            with open(file_path, 'rb') as f:
                # ijson.items yields complete objects from the array
                # 'item' refers to each object in the root list
                documents = ijson.items(f, 'item')
                
                count = 0
                for doc in documents:
                    self.process_document(doc)
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} documents...", end='\r')
                
                print(f"\nFinished processing {count} documents from {file_path}.")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def process_document(self, doc: Dict[str, Any]):
        """
        Process a single document object, generate chunks, and write to a text file.
        
        Args:
            doc (Dict): The document dictionary.
        """
        element_id = doc.get("element_id", "unknown")
        
        # 1. Prepare Global Header (Metadata that appears in every chunk)
        header_text = self._create_header(doc)
        
        # 2. Flatten Articles into a single text stream with their own metadata
        full_text_content = self._flatten_articles(doc.get("articles", []))
        
        # 3. Generate Chunks (Sliding Window)
        chunks = self._chunk_text(header_text, full_text_content)
        
        # 4. Write to file
        self._write_chunks(element_id, chunks)

    def _create_header(self, doc: Dict[str, Any]) -> str:
        """Create the standardized header for the document."""
        # Using .get to handle potential missing fields gracefully
        name = doc.get("decision_name", "")
        address = doc.get("law_address", "")
        e_id = doc.get("element_id", "")
        
        header = f"وثيقة رقم: {e_id}\n"
        header += f"القرار: {name}\n"
        header += f"العنوان: {address}\n"
        header += "-" * 20 + "\n"
        return header

    def _flatten_articles(self, articles: List[Dict[str, Any]]) -> str:
        """
        Convert list of articles into a single formatted string.
        Handles canceled articles and dates.
        """
        if not articles:
            return ""

        text_parts = []
        for art in articles:
            title = art.get("article_title", "")
            
            # Determine status and dates
            is_canceled = art.get("is_canceled", 0)
            working_date = art.get("working_date") or "غير محدد"
            
            # Logic for content selection
            # If canceled, prefer original_content. If not available, use article_content.
            # Note: User said "if the article is canceled and original content is not null that means that content of it is inside the original content"
            raw_content = art.get("article_content", "")
            original_content = art.get("original_content")
            
            date_info = f"تاريخ العمل: {working_date}"
            
            if is_canceled == 1 or is_canceled == "1":
                status = "ملغاة"
                canceling_date = art.get("canceling_date") or "غير محدد"
                date_info += f" | تاريخ الإلغاء: {canceling_date}"
                
                if original_content:
                    content_text = original_content
                else:
                    content_text = raw_content
            else:
                status = "سارية"
                content_text = raw_content

            # Clean content (remove excessive newlines if needed, but keeping structure is usually good)
            if content_text is None:
                content_text = ""
            content_text = content_text.strip()

            # Construct Article Block
            article_block = f"\n[{title} ({status}) - {date_info}]\n"
            article_block += f"{content_text}\n"
            
            text_parts.append(article_block)
            
        return "".join(text_parts)

    def _chunk_text(self, header: str, body: str) -> List[str]:
        """
        Split body text into overlapping windows and prepend header to each.
        """
        chunks = []
        body_len = len(body)
        
        if body_len == 0:
            # Even if empty body, we might want one chunk with just metadata? 
            # Or skip. Let's return one chunk with header.
            return [header.strip()]

        start = 0
        while start < body_len:
            end = min(start + self.chunk_size, body_len)
            
            # Refinement: Try not to split words?
            # Find the nearest space after chunk_size IF we are not at end
            if end < body_len:
                # Look ahead for a space/newline to avoid cutting words
                # Limit lookahead to avoid extending chunk too much (e.g. 100 chars)
                next_space = -1
                for i in range(0, 100):
                    if end + i < body_len and body[end+i] in [' ', '\n', '\t']:
                        next_space = end + i
                        break
                if next_space != -1:
                    end = next_space

            chunk_content = body[start:end]
            full_chunk = header + chunk_content
            chunks.append(full_chunk)
            
            # Optimization: If we reached the end, break
            if end >= body_len:
                break
                
            # Move start pointer (Sliding Window)
            start += (self.chunk_size - self.overlap)
            
            # Ensure we don't get stuck if overlap >= chunk_size (bad config)
            if (self.chunk_size - self.overlap) <= 0:
                start += self.chunk_size  # Fallback to no overlap to prevent infinite loop
                
        return chunks

    def _write_chunks(self, element_id: str, chunks: List[str]):
        """Write formattted chunks to a file."""
        # Sanitize filename
        safe_id = "".join(c for c in str(element_id) if c.isalnum() or c in ('-', '_'))
        filename = f"{safe_id}.txt"
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                f.write(f"--- CHUNK {i+1} ---\n")
                f.write(chunk)
                f.write("\n\n")

def main():
    processor = DocumentProcessor(
        output_dir=OUTPUT_DIR,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP_SIZE
    )
    
    for input_file in INPUT_FILES:
        if os.path.exists(input_file):
            processor.process_json_file(input_file)
        else:
            print(f"Warning: File {input_file} not found.")

if __name__ == "__main__":
    main()
