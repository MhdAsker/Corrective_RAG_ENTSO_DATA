"""
Simplified demo evaluation for Corrective RAG - uses BM25 only (no HF API needed).
Shows the complete flow and metrics system working.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from retrieval.bm25_index import BM25Index
from eval.metrics import MetricsComputer, QueryResult

# Test queries
TEST_QUERIES = [
    "What is the FCR activation threshold?",
    "How are imbalance settlement periods defined?",
    "What is the required capacity for FRR?",
]


def retrieve_bm25_only(query: str) -> list[dict]:
    """Retrieve using BM25 only (no embeddings needed)"""
    bm25 = BM25Index.load("data/embeddings/bm25_index.pkl")
    return bm25.search(query, top_k=5)


def generate_answer(query: str, chunks: list[dict]) -> str:
    """Generate answer from chunks"""
    context = "\n\n".join(
        [f"Source: {c.get('slug', 'unknown')}\n{c['text']}" for c in chunks]
    )

    prompt = f"""You are an expert assistant for energy grid regulations. Given the following extracted parts of a document and a question, create a final answer.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.

Question: {query}
=========
Context:
{context}
=========
Answer:"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )

    response = llm.invoke(prompt)
    return response.content


def run_demo_evaluation():
    """Run simplified demo evaluation"""
    computer = MetricsComputer()

    print("\n" + "="*60)
    print("CORRECTIVE RAG - DEMO EVALUATION")
    print("="*60)
    print("\nUsing BM25 retrieval only (demo mode)")
    print(f"Evaluating {len(TEST_QUERIES)} queries\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"[{i}/{len(TEST_QUERIES)}] {query}")

        try:
            # Retrieve chunks
            chunks = retrieve_bm25_only(query)
            print(f"  Retrieved: {len(chunks)} chunks")

            # For demo: assume first retrieval is "naive", second is "corrective"
            # In real setup, would compare actual naive vs CRAG
            naive_chunks = chunks[:3]  # Mock: first 3 chunks as "naive"
            crag_chunks = chunks[1:4]  # Mock: different selection as "CRAG"

            # Generate answers
            naive_gen = generate_answer(query, naive_chunks)
            crag_gen = generate_answer(query, crag_chunks)

            print(f"  Generated answers (naive: {len(naive_gen.split())} words, crag: {len(crag_gen.split())} words)")

            # Mock scores (in real eval, these come from LLM grader)
            naive_relevance = [0.6, 0.7, 0.4]
            crag_relevance = [0.8, 0.75, 0.85]

            avg_naive = sum(naive_relevance) / len(naive_relevance)
            avg_crag = sum(crag_relevance) / len(crag_relevance)

            print(f"  Relevance: naive={avg_naive:.2f}, crag={avg_crag:.2f}")

            # Store result
            result = QueryResult(
                query=query,
                naive_chunks=naive_chunks,
                naive_generation=naive_gen,
                crag_chunks=crag_chunks,
                crag_generation=crag_gen,
                corrections_triggered=(avg_naive < 0.7),  # Mock trigger
                retry_count=1 if avg_naive < 0.7 else 0,
                web_search_used=False,
                naive_relevance_scores=naive_relevance,
                crag_relevance_scores=crag_relevance,
                avg_naive_relevance=avg_naive,
                avg_crag_relevance=avg_crag,
            )

            computer.add_result(result)
            time.sleep(1)

        except Exception as e:
            print(f"  ERROR: {e}")

    # Print report
    print("\n" + "="*60 + "\n")
    if computer.results:
        metrics = computer.compute_overall_metrics()
        computer.print_report(metrics)
        print("\nMetrics Table:")
        print(computer.export_table(metrics))
    else:
        print("No results collected!")


if __name__ == "__main__":
    run_demo_evaluation()
