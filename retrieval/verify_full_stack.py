"""
Phase 3 end-to-end stack verification.

Boots every retrieval component, runs a German query through the full pipeline,
prints each intermediate result, then runs the MRR@5 benchmark.

Run with:
    python -m retrieval.verify_full_stack
"""

from dotenv import load_dotenv
load_dotenv()

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

EMBEDDINGS_PATH = "data/embeddings/all_chunks_embedded.json"
BM25_PATH       = "data/embeddings/bm25_index.pkl"
QDRANT_PATH     = "data/qdrant"

TEST_QUERY = (
    "Was sind die TSO-Pflichten gemaess Artikel 14 bezueglich "
    "N-1 Sicherheit und Echtzeit-Ueberwachung?"
)


def _check_files() -> None:
    missing = [p for p in [EMBEDDINGS_PATH, BM25_PATH] if not Path(p).exists()]
    if missing:
        print(
            f"\nERROR: Missing required files: {missing}\n"
            "Run `python -m retrieval.embed_corpus` first to generate embeddings.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    _check_files()

    # -- Boot all components ----------------------------------------------------
    print(f"\n{'='*60}")
    print("  Phase 3 Full Stack Verification")
    print(f"{'='*60}\n")

    print("Loading components...")

    from retrieval.embedder       import CorpusEmbedder
    from retrieval.bm25_index     import BM25Index
    from retrieval.vector_store   import GridKnowledgeStore
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker       import CrossEncoderReranker
    from retrieval.context_builder import ContextBuilder
    from retrieval.benchmark      import run_benchmark

    embedder  = CorpusEmbedder()
    bm25      = BM25Index.load(BM25_PATH)
    store     = GridKnowledgeStore(path=QDRANT_PATH)
    store.load_from_embeddings(EMBEDDINGS_PATH)
    retriever = HybridRetriever(store, embedder, bm25)
    reranker  = CrossEncoderReranker()
    builder   = ContextBuilder(max_tokens=3000)

    print(f"\nQuery: \"{TEST_QUERY}\"\n")

    # --------------------------------------------------------------------------
    # STEP 1 -- HybridRetriever
    # --------------------------------------------------------------------------
    print("--- STEP 1 -- HybridRetriever (top-20) ------------------------------")
    hybrid_results = retriever.search(TEST_QUERY, top_k=20)

    print(f"  Retrieved {len(hybrid_results)} candidates via RRF fusion\n")
    print(f"  {'Rank':<6}{'chunk_id':<20}{'rrf_score':<12}heading_path")
    print(f"  {'-'*4}  {'-'*18}  {'-'*9}  {'-'*30}")
    for i, r in enumerate(hybrid_results[:3], 1):
        heading = (r.get("heading_path") or "--")[:40]
        print(f"  [{i}]   {r['chunk_id']:<20}{r['rrf_score']:<12.6f}{heading}")

    # --------------------------------------------------------------------------
    # STEP 2 -- CrossEncoderReranker
    # --------------------------------------------------------------------------
    print("\n--- STEP 2 -- CrossEncoderReranker (top-5) ---------------------------")
    analysis = reranker.rerank_with_analysis(TEST_QUERY, hybrid_results, top_k=5)
    reranked = analysis["reranked"]

    print(f"  {'Rank':<6}{'chunk_id':<20}{'rerank_score':<14}{'rank_delta':<12}heading_path")
    print(f"  {'-'*4}  {'-'*18}  {'-'*11}  {'-'*9}  {'-'*30}")
    for r in reranked[:3]:
        delta     = r.get("rank_delta", 0)
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        heading   = (r.get("heading_path") or "--")[:30]
        print(
            f"  [{r['new_rank']}]   {r['chunk_id']:<20}"
            f"{r.get('rerank_score', 0.0):<14.4f}"
            f"{delta_str:<12}{heading}"
        )

    if analysis["biggest_mover_up"]:
        mu = analysis["biggest_mover_up"]
        print(f"\n  Biggest mover  ^  {mu['chunk_id']}  (Δrank = +{mu['rank_delta']})")
    if analysis["biggest_mover_down"]:
        md = analysis["biggest_mover_down"]
        print(f"  Biggest mover  v  {md['chunk_id']}  (Δrank = {md['rank_delta']})")

    # --------------------------------------------------------------------------
    # STEP 3 -- ContextBuilder
    # --------------------------------------------------------------------------
    print("\n--- STEP 3 -- ContextBuilder (max 3000 tokens) -----------------------")
    meta = builder.build_with_metadata(reranked)

    print(f"  chunks_used    : {meta['chunks_used']}")
    print(f"  chunks_dropped : {meta['chunks_dropped']}")
    print(f"  total_tokens   : {meta['total_tokens']}")
    print(f"  sources        : {meta['sources']}")

    # --------------------------------------------------------------------------
    # STEP 4 -- Full context string (what the LLM will see in Phase 4)
    # --------------------------------------------------------------------------
    print("\n--- STEP 4 -- LLM Context String -------------------------------------")
    print(meta["context"])

    # --------------------------------------------------------------------------
    # MRR@5 BENCHMARK
    # --------------------------------------------------------------------------
    print("\n--- MRR@5 Benchmark -------------------------------------------------")
    run_benchmark(retriever, reranker)

    # --------------------------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------------------------
    sep = "=" * 60
    print(sep)
    print("  Phase 3 Verification Complete")
    print(sep)
    print(f"  Components verified  : CorpusEmbedder, BM25Index,")
    print(f"                         GridKnowledgeStore (Qdrant),")
    print(f"                         HybridRetriever (RRF),")
    print(f"                         CrossEncoderReranker,")
    print(f"                         ContextBuilder")
    print(f"  Query language       : German")
    print(f"  Corpus vectors       : {store.count()}")
    print(f"  BM25 documents       : {bm25.index.corpus_size if bm25.index else 'N/A'}")
    print(f"  Hybrid candidates    : {len(hybrid_results)}")
    print(f"  After reranking      : {len(reranked)}")
    print(f"  Context tokens       : {meta['total_tokens']} / 3000")
    print(f"  Sources in context   : {', '.join(meta['sources'])}")
    print(sep)
    print("  [OK] Stack ready for Phase 4 -- Groq LLM agent")
    print(f"{sep}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error("Unexpected error in verify_full_stack", exc_info=True)
        raise
