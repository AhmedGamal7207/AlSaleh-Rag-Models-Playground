import time
from retrieval_pipeline import RAGRetriever

# CELL 3
QUERIES = [
    "ما هي حالات استحقاق المعاش قبل سن الستين؟",
    "هل يجوز الجمع بين المعاش والأجر في القانون المصري؟",
    "ما هي مدة الاشتراك المطلوبة لاستحقاق معاش الشيخوخة؟",
    "متى يسقط الحق في صرف المعاش؟",
    "ما هي حقوق الورثة في معاش المتوفى؟",

    "ما الفرق بين السرقة البسيطة والسرقة المشددة؟",
    "ما عقوبة السرقة المقترنة بحمل سلاح؟",
    "ما هي أركان جريمة السرقة في القانون المصري؟",
    "متى تتحول السرقة إلى جناية؟",
    "هل الشروع في السرقة يعاقب عليه؟",

    "ما الفرق بين التسجيل في الشهر العقاري وصحة التوقيع؟",
    "ما المستندات المطلوبة لتسجيل شقة في الشهر العقاري؟",
    "هل يجوز تسجيل عقد عرفي في الشهر العقاري؟",
    "ما هي إجراءات التسجيل وفقًا لقانون الشهر العقاري الجديد؟",
    "ما الحالات التي يُرفض فيها تسجيل العقار؟",

    "هل يجوز البناء على الأراضي الزراعية؟",
    "ما عقوبة التعدي على الأراضي الزراعية؟",
    "كيف يتم إثبات ملكية الأرض الزراعية؟",
    "هل يجوز بيع الأرض الزراعية بعقد عرفي؟",
    "ما الفرق بين وضع اليد والملكية المسجلة؟",

    "ما هي حقوق الزوجة بعد الطلاق؟",
    "متى تستحق الزوجة نفقة العدة؟",
    "ما شروط الخلع في القانون المصري؟",
    "هل يجوز إسقاط النفقة باتفاق الطرفين؟",
    "ما الحالات التي يسقط فيها حق الحضانة؟",

    "ما حقوق العامل عند الفصل التعسفي؟",
    "ما مدة الإخطار قبل إنهاء عقد العمل؟",
    "هل يجوز فصل العامل دون تحقيق؟",
    "ما الحالات التي يجوز فيها إنهاء عقد العمل؟"
]

OUTPUT_FILE = "retrieval_result_long.txt"

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
        f.write("Method: Main RAG Pipeline (Vector + Reranker)\n")
        f.write("=======================================\n\n")

        for i, query in enumerate(QUERIES):
            print(f"\nProcessing Query {i+1}: {query}")
            f.write(f"QUERY {i+1}: {query}\n")
            f.write("-" * 40 + "\n")
            
            start_search = time.time()
            # Retrieve top 5 final results
            results = retriever.retrieve(query, top_k=5)
            duration = time.time() - start_search
            
            f.write(f"Time taken: {duration:.4f}s\n\n")
            
            if not results:
                f.write("No results found.\n")
            
            for rank, res in enumerate(results):
                payload = res['payload']
                score = res.get('score', 0.0)
                
                f.write(f"Rank: {rank + 1} | Score: {score:.5f}\n")
                f.write(f"Law: {payload.get('law_name', 'unknown')}\n")
                f.write(f"Article: {payload.get('article_title', 'unknown')} ({payload.get('status', '')})\n")
                f.write("Content Snippet:\n")
                
                # Indent content for readability
                content = payload.get('text_content', '').strip()
                f.write(f"{content}\n") 
                f.write("\n" + "*"*30 + "\n\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"\nTest finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()