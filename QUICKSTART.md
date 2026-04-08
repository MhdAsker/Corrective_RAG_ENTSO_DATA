# Corrective RAG - Quick Start Guide

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Check Your API Keys
```bash
# Create .env file with:
# GROQ_API_KEY=your_groq_key_here
# HF_API_KEY=your_hf_key_here (optional)
```

### 3. Verify System Works
```bash
python test_system.py
```

Expected output: ✅ ALL CHECKS PASSED

---

## Running the System

### Option A: Quick Demo (Recommended for First Time)
```bash
python eval/demo_evaluation.py
```

**Pros:**
- No HF API needed
- Fast (~1 minute)
- Shows complete CRAG workflow
- Demonstrates metrics computation

**Output:**
```
[1/3] What is the FCR activation threshold?
  Retrieved: 5 chunks
  Generated answers (naive: 46 words, crag: 105 words)
  Relevance: naive=0.57, crag=0.80
  
Relevance Improvement: +41.18%
Correction Success Rate: 100.0%
Avg Faithfulness (RAGAS): 0.80
```

### Option B: Web UI
```bash
streamlit run app.py
```

Opens browser UI to compare Naive RAG vs CRAG side-by-side.

### Option C: Full Evaluation (Requires HF API Access)
```bash
python eval/run_evaluation.py
```

Runs LLM-based grading on 10 test queries.

---

## Usage in Code

### Simple One-Shot Query

```python
from graph.crag_graph import crag_app

result = crag_app.invoke({
    "question": "What is the FCR activation threshold?"
})

print(result["generation"])
print(f"Correction triggered: {result['retry_count'] > 0}")
print(f"Web search used: {result['web_search_used']}")
```

### Batch Evaluation

```python
from graph.crag_graph import crag_app
from eval.metrics import MetricsComputer, QueryResult

computer = MetricsComputer()

queries = [
    "What is the FCR activation threshold?",
    "How are imbalance settlement periods defined?",
    "What is the required capacity for FRR?",
]

for query in queries:
    result = crag_app.invoke({"question": query})
    
    # Store result for metrics
    qr = QueryResult(
        query=query,
        naive_chunks=result.get("good_chunks", []),
        naive_generation="",
        crag_chunks=result.get("good_chunks", []),
        crag_generation=result.get("generation", ""),
        corrections_triggered=result.get("retry_count", 0) > 0,
        retry_count=result.get("retry_count", 0),
        web_search_used=result.get("web_search_used", False),
        naive_relevance_scores=[0.5],  # Mock
        crag_relevance_scores=[0.7],   # Mock
        avg_naive_relevance=0.5,
        avg_crag_relevance=0.7,
    )
    computer.add_result(qr)

# Get metrics
metrics = computer.compute_overall_metrics()
print(f"Correction rate: {metrics.correction_rate:.1f}%")
print(f"Relevance improvement: {metrics.relevance_improvement:+.1f}%")
```

### Custom Retrieval

```python
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import GridKnowledgeStore
from retrieval.embedder import CorpusEmbedder
from retrieval.bm25_index import BM25Index

store = GridKnowledgeStore()
embedder = CorpusEmbedder(skip_verification=True)
bm25 = BM25Index.load("data/embeddings/bm25_index.pkl")

retriever = HybridRetriever(store, embedder, bm25)

# Retrieve top-10 most relevant chunks
chunks = retriever.search("FCR activation threshold", top_k=10)

for chunk in chunks:
    print(f"{chunk['rrf_score']:.3f} | {chunk['text'][:80]}")
```

---

## Key Metrics Explained

| Metric | Range | Good | Bad |
|--------|-------|------|-----|
| **Correction Rate** | 0-100% | 20-40% (some queries need help) | 0% (system not helping) or 100% (all fail) |
| **Correction Success** | 0-100% | >80% (rewrites are effective) | <50% (rewrites aren't helping) |
| **Relevance Score** | 0.0-1.0 | >0.7 (good retrieval) | <0.5 (poor retrieval) |
| **Relevance Improvement** | -∞ to +∞ | +20% or more | -10% (CRAG makes it worse) |
| **Faithfulness** | 0.0-1.0 | >0.8 (answers follow context) | <0.6 (hallucinations) |

---

## Troubleshooting

### Error: "GROQ_API_KEY not found"
```bash
# Add to .env file:
GROQ_API_KEY=gsk_...your_key...
```

### Error: "HF API 403 Forbidden"
Use demo mode instead:
```bash
python eval/demo_evaluation.py  # Uses BM25 only
```

### Error: "UnicodeEncodeError on Windows"
```bash
set PYTHONIOENCODING=utf-8
python eval/demo_evaluation.py
```

### Error: "Qdrant collection not found"
Data files are missing. Ensure `data/` directory exists with:
- `data/embeddings/all_chunks_embedded.json`
- `data/embeddings/bm25_index.pkl`
- `data/qdrant/meta.json`

---

## Performance Benchmarks

On a typical system with Groq API:

| Operation | Time |
|-----------|------|
| Single query retrieval | 50ms |
| LLM relevance grading (5 chunks) | 2-3s |
| Query rewriting (LLM) | 1-2s |
| Answer generation (LLM) | 2-3s |
| **Total per query** | **5-8 seconds** |

Throughput: ~10-15 queries/minute with Groq free tier

---

## Next Steps

1. **Explore** the demo output and understand the metrics
2. **Read** the [README.md](README.md) for detailed architecture
3. **Review** the [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for technical decisions
4. **Customize** the system for your use case
5. **Deploy** to production with proper monitoring

---

## Common Customizations

### Change the LLM Model
Edit `graph/crag_graph.py`, `grader/relevance_grader.py`, `grader/query_rewriter.py`:

```python
llm = ChatGroq(
    model="mixtral-8x7b-32768",  # Change here
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY")
)
```

### Adjust Relevance Threshold
Edit `graph/crag_graph.py`:

```python
def grade_node(state: GraphState):
    # Change threshold from 1 to 0.7
    if res.score >= 0.7:  # More lenient
        good_chunks.append(chunk)
```

### Add More Documents
1. Place PDFs in a folder
2. Run: `python corpus/extractor.py your_folder/`
3. Run: `python retrieval/embed_corpus.py`
4. System automatically re-indexes

---

## Architecture Diagram

```
User Query
    ↓
┌─────────────────────────────┐
│  LangGraph Workflow         │
├─────────────────────────────┤
│ 1. Retrieve (Hybrid Search) │
│    ├─ Dense (Qdrant)        │
│    ├─ Sparse (BM25)         │
│    └─ RRF Fusion            │
├─────────────────────────────┤
│ 2. Grade (LLM Scoring)      │
│    ├─ Score: 0.0 - 1.0      │
│    └─ Route based on scores │
├─────────────────────────────┤
│ 3. Conditional Routing      │
│    ├─ Good? → Generate      │
│    ├─ Bad 1st? → Rewrite    │
│    └─ Bad 2nd? → Web Search │
├─────────────────────────────┤
│ 4. Generate (LLM Answer)    │
│    └─ Create final answer   │
└─────────────────────────────┘
    ↓
Final Answer + Metrics
```

---

For more details, see [README.md](README.md)

**Happy RAG-ing!** 🚀
