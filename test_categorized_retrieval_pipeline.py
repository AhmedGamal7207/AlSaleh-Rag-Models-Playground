import time
from categorized_retrieval_pipeline import CategorizedRAGRetriever

QUERIES = [
    {"question": "ما هي حالات استحقاق المعاش قبل سن الستين؟", "category": "قانون العمل"},
    {"question": "هل يجوز الجمع بين المعاش والأجر في القانون المصري؟", "category": "قانون العمل"},
    {"question": "ما هي مدة الاشتراك المطلوبة لاستحقاق معاش الشيخوخة؟", "category": "قانون العمل"},
    {"question": "متى يسقط الحق في صرف المعاش؟", "category": "قانون العمل"},
    {"question": "ما هي حقوق الورثة في معاش المتوفى؟", "category": "قانون العمل"},

    {"question": "ما الفرق بين السرقة البسيطة والسرقة المشددة؟", "category": "القانون الجنائي"},
    {"question": "ما عقوبة السرقة المقترنة بحمل سلاح؟", "category": "القانون الجنائي"},
    {"question": "ما هي أركان جريمة السرقة في القانون المصري؟", "category": "القانون الجنائي"},
    {"question": "متى تتحول السرقة إلى جناية؟", "category": "القانون الجنائي"},
    {"question": "هل الشروع في السرقة يعاقب عليه؟", "category": "القانون الجنائي"},

    {"question": "ما الفرق بين التسجيل في الشهر العقاري وصحة التوقيع؟", "category": "القانون المدني"},
    {"question": "ما المستندات المطلوبة لتسجيل شقة في الشهر العقاري؟", "category": "القانون المدني"},
    {"question": "هل يجوز تسجيل عقد عرفي في الشهر العقاري؟", "category": "القانون المدني"},
    {"question": "ما هي إجراءات التسجيل وفقًا لقانون الشهر العقاري الجديد؟", "category": "القانون المدني"},
    {"question": "ما الحالات التي يُرفض فيها تسجيل العقار؟", "category": "القانون المدني"},

    {"question": "هل يجوز البناء على الأراضي الزراعية؟", "category": "القانون الإداري"},
    {"question": "ما عقوبة التعدي على الأراضي الزراعية؟", "category": "القانون الجنائي"},
    {"question": "كيف يتم إثبات ملكية الأرض الزراعية؟", "category": "القانون المدني"},
    {"question": "هل يجوز بيع الأرض الزراعية بعقد عرفي؟", "category": "القانون المدني"},
    {"question": "ما الفرق بين وضع اليد والملكية المسجلة؟", "category": "القانون المدني"},

    {"question": "ما هي حقوق الزوجة بعد الطلاق؟", "category": "قانون الأسرة"},
    {"question": "متى تستحق الزوجة نفقة العدة؟", "category": "قانون الأسرة"},
    {"question": "ما شروط الخلع في القانون المصري؟", "category": "قانون الأسرة"},
    {"question": "هل يجوز إسقاط النفقة باتفاق الطرفين؟", "category": "قانون الأسرة"},
    {"question": "ما الحالات التي يسقط فيها حق الحضانة؟", "category": "قانون الأسرة"},

    {"question": "ما حقوق العامل عند الفصل التعسفي؟", "category": "قانون العمل"},
    {"question": "ما مدة الإخطار قبل إنهاء عقد العمل؟", "category": "قانون العمل"},
    {"question": "هل يجوز فصل العامل دون تحقيق؟", "category": "قانون العمل"},
    {"question": "ما الحالات التي يجوز فيها إنهاء عقد العمل؟", "category": "قانون العمل"}
]


OUTPUT_FILE = "retrieval_result_categorized.txt"

def main():
    print("--- Starting Categorized Retrieval Test ---")
    
    try:
        start_load = time.time()
        retriever = CategorizedRAGRetriever()
        print(f"Pipeline loaded in {time.time() - start_load:.2f} seconds.")
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("=== Categorized Retrieval Test Results ===\n")
        f.write("Method: Vector Search + Category Filter + Reranker\n")
        f.write("==========================================\n\n")

        for i, item in enumerate(QUERIES):
            query_text = item["question"]
            category = item["category"]
            
            print(f"\nProcessing Query {i+1}: {query_text} (Category: {category})")
            f.write(f"QUERY {i+1}: {query_text}\n")
            f.write(f"Category Filter: {category}\n")
            f.write("-" * 40 + "\n")
            
            start_search = time.time()
            # Pass the category to the retrieve method
            results = retriever.retrieve(query_text, category=category, top_k=5)
            duration = time.time() - start_search
            
            f.write(f"Time taken: {duration:.4f}s\n\n")
            
            if not results:
                f.write("No results found matching this category.\n")
            
            for rank, res in enumerate(results):
                payload = res['payload']
                score = res.get('score', 0.0)
                
                f.write(f"Rank: {rank + 1} | Score: {score:.5f}\n")
                f.write(f"Law: {payload.get('law_name', 'unknown')}\n")
                f.write(f"Article: {payload.get('article_title', 'unknown')} ({payload.get('status', '')})\n")
                # Show matched categories to verify
                cats = payload.get('categories', [])
                f.write(f"Categories: {cats}\n")
                
                f.write("Content Snippet:\n")
                content = payload.get('text_content', '').strip()
                f.write(f"{content}\n") 
                f.write("\n" + "*"*30 + "\n\n")
            
            f.write("\n" + "="*50 + "\n\n")
            
    print(f"\nTest finished. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()