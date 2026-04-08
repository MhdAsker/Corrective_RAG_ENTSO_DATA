#!/usr/bin/env python3
"""
Quick system health check for Corrective RAG.
Verifies all components are working correctly.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("CORRECTIVE RAG - SYSTEM HEALTH CHECK")
print("=" * 60)

# 1. Check environment variables
print("\n[1/6] Checking environment variables...")
required_keys = ["GROQ_API_KEY"]
optional_keys = ["HF_API_KEY"]

for key in required_keys:
    if key in os.environ:
        print(f"  OK: {key} is set")
    else:
        print(f"  ERROR: {key} is NOT set")
        sys.exit(1)

for key in optional_keys:
    if key in os.environ:
        print(f"  OK: {key} is set (optional)")
    else:
        print(f"  WARN: {key} is not set (optional)")

# 2. Check data files
print("\n[2/6] Checking data files...")
data_files = [
    "data/embeddings/all_chunks_embedded.json",
    "data/embeddings/bm25_index.pkl",
    "data/qdrant/meta.json",
]

for file in data_files:
    if Path(file).exists():
        print(f"  OK: {file}")
    else:
        print(f"  WARN: {file} not found")

# 3. Import core modules
print("\n[3/6] Importing core modules...")
try:
    from retrieval.bm25_index import BM25Index
    print("  OK: BM25Index imported")
except Exception as e:
    print(f"  ERROR: Failed to import BM25Index: {e}")
    sys.exit(1)

try:
    from retrieval.vector_store import GridKnowledgeStore
    print("  OK: GridKnowledgeStore imported")
except Exception as e:
    print(f"  ERROR: Failed to import GridKnowledgeStore: {e}")
    sys.exit(1)

try:
    from grader.relevance_grader import grade_chunk
    print("  OK: Relevance grader imported")
except Exception as e:
    print(f"  ERROR: Failed to import relevance grader: {e}")
    sys.exit(1)

try:
    from grader.query_rewriter import rewrite_query
    print("  OK: Query rewriter imported")
except Exception as e:
    print(f"  ERROR: Failed to import query rewriter: {e}")
    sys.exit(1)

try:
    from eval.metrics import MetricsComputer
    print("  OK: Metrics computer imported")
except Exception as e:
    print(f"  ERROR: Failed to import metrics computer: {e}")
    sys.exit(1)

# 4. Test BM25 retrieval
print("\n[4/6] Testing BM25 retrieval...")
try:
    bm25 = BM25Index.load("data/embeddings/bm25_index.pkl")
    results = bm25.search("FCR activation", top_k=3)
    if results:
        print(f"  OK: Retrieved {len(results)} documents")
        print(f"       First result: {results[0]['text'][:60]}...")
    else:
        print("  WARN: No results from BM25 search")
except Exception as e:
    print(f"  ERROR: BM25 search failed: {e}")
    sys.exit(1)

# 5. Test Groq LLM
print("\n[5/6] Testing Groq LLM connection...")
try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    response = llm.invoke("Say 'System OK' in exactly 2 words.")
    if response:
        print(f"  OK: Groq response received")
        print(f"       Response: {response.content[:50]}")
    else:
        print("  ERROR: No response from Groq")
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: Groq connection failed: {e}")
    sys.exit(1)

# 6. Test metrics computation
print("\n[6/6] Testing metrics computation...")
try:
    from eval.metrics import QueryResult

    computer = MetricsComputer()
    result = QueryResult(
        query="Test query",
        naive_chunks=[{"text": "chunk1"}],
        naive_generation="Test generation",
        crag_chunks=[{"text": "chunk2"}],
        crag_generation="Better generation",
        corrections_triggered=True,
        retry_count=1,
        web_search_used=False,
        naive_relevance_scores=[0.5],
        crag_relevance_scores=[0.8],
        avg_naive_relevance=0.5,
        avg_crag_relevance=0.8,
    )
    computer.add_result(result)
    metrics = computer.compute_overall_metrics()

    print(f"  OK: Metrics computed successfully")
    print(f"       Correction rate: {metrics.correction_rate:.0f}%")
    print(f"       Relevance improvement: {metrics.relevance_improvement:+.1f}%")
except Exception as e:
    print(f"  ERROR: Metrics computation failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL CHECKS PASSED - System is ready!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run demo: python eval/demo_evaluation.py")
print("  2. Run web UI: streamlit run app.py")
print("  3. View README: cat README.md")
