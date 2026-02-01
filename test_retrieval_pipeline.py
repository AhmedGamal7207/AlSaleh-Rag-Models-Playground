from retrieval_pipeline import RAGRetriever
import time

# Queries to test specific legal knowledge
# Using queries that require finding specific articles among thousands
QUERIES = [
    # Query 1: Pension/Social Security Law
    "ما هي شروط استحقاق المعاش المبكر؟", 
    # (Translation: What are the conditions for early retirement pension eligibility?)

    # Query 2: Criminal Law / Theft
    "ما هي عقوبة السرقة بالإكراه في القانون المصري؟",
    # (Translation: What is the penalty for theft by coercion in Egyptian law?)

    # Query 3: Real Estate / Land
    "كيف يتم تسجيل الشهر العقاري للأراضي الزراعية؟"
    # (Translation: How is real estate registration done for agricultural lands?)
]

OUTPUT_FILE = "retrieval_result.txt"

def main():
    print("--- Starting Retrieval Agent Test (Production Mode) ---")
    
    # Initialize the retriever
    try:
        start_load = time.time()
        retriever = RAGRetriever()
        print(f"Pipeline loaded in {time.time() - start_load:.2f} seconds.")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Retrieval Pipeline Test Results ===\n")
        f.write("Method: Two-Stage Retrieval (Dense Vector Search + Cross-Encoder Reranking)\n")
        f.write("=======================================\n\n")

        for i, query in enumerate(QUERIES):
            print(f"\nProcessing Query {i+1}: {query}")
            f.write(f"QUERY {i+1}: {query}\n")
            f.write("-" * 40 + "\n")
            
            start_search = time.time()
            # Retrieve top 5 final results, searching top 50 candidates first
            results = retriever.retrieve(query, top_k=5, search_k=50)
            duration = time.time() - start_search
            
            f.write(f"Time taken: {duration:.4f}s\n\n")
            
            if not results:
                f.write("No results found.\n")
            
            for res in results:
                f.write(f"Rank: {res['rank']} | CE Score: {res['score']:.5f} (Initial: {res['initial_score']:.4f})\n")
                f.write(f"Source: {res['metadata'].get('source_file', 'unknown')}\n")
                f.write("Content Snippet:\n")
                # Indent content for readability
                content = res['content'].strip()
                f.write(f"{content}\n") 
                f.write("\n" + "*"*30 + "\n\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"\nTest finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
