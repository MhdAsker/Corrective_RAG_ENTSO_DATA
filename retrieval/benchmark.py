"""
MRR@5 benchmark -- hybrid-only vs hybrid+rerank.

Measures whether the cross-encoder reranker actually improves retrieval
quality over hybrid BM25+dense retrieval alone.

Metric: MRR@5 (Mean Reciprocal Rank at 5)
  For each query, find the first result in the top-5 whose chunk_id
  contains the expected keyword. MRR@5 = mean of 1/rank across all queries.
  MRR@5 = 1.0 means the right chunk is always rank-1.
  MRR@5 = 0.0 means the right chunk never appears in the top-5.
"""

import logging

# 10 test (query, relevant_keyword) pairs.
# relevant_keyword is matched case-insensitively against chunk_id.
TEST_QUERIES = [
    ("N-1 Sicherheitskriterium Uebertragungsnetz",          "article"),
    ("N-1 security criterion transmission lines",           "article"),
    ("FCR frequency containment reserve sizing",            "article"),
    ("Frequenzhaltungsreserve Aktivierungsschwelle",        "article"),
    ("aFRR automatic frequency restoration activation",     "article"),
    ("Ausgleichsenergie Bilanzkreis Abrechnung",            "article"),
    ("TSO Betriebssicherheitspflichten Echtzeitueberwachung","article"),
    ("operational security real-time monitoring TSO",       "article"),
    ("Schwarzstart Wiederversorgung nach Blackout",         "article"),
    ("restoration reconnection procedure after blackout",   "article"),
]


def mrr_at_k(results: list[dict], keyword: str, k: int = 5) -> float:
    """
    Return 1/rank of the first result in top-k whose chunk_id contains
    `keyword` (case-insensitive). Returns 0.0 if not found.
    """
    keyword_lower = keyword.lower()
    for rank, result in enumerate(results[:k], 1):
        if keyword_lower in result.get("chunk_id", "").lower():
            return 1.0 / rank
    return 0.0


def run_benchmark(retriever, reranker) -> None:
    """
    Run MRR@5 benchmark comparing hybrid-only vs hybrid+rerank.

    Prints a results table and mean MRR values.
    """
    COL_Q  = 41
    COL_MH = 12
    COL_MR = 13
    COL_D  = 7

    border_top    = "+" + "-" * COL_Q + "+" + "-" * COL_MH + "+" + "-" * COL_MR + "+" + "-" * COL_D + "+"
    border_head   = "+" + "-" * COL_Q + "+" + "-" * COL_MH + "+" + "-" * COL_MR + "+" + "-" * COL_D + "+"
    border_bottom = "+" + "-" * COL_Q + "+" + "-" * COL_MH + "+" + "-" * COL_MR + "+" + "-" * COL_D + "+"

    print(f"\n{'='*78}")
    print("  MRR@5 Benchmark -- Hybrid vs Hybrid + Rerank")
    print(f"{'='*78}")
    print(border_top)
    print(
        f"| {'Query (40 chars)':<{COL_Q-2}} "
        f"| {'MRR hybrid':>{COL_MH-2}} "
        f"| {'MRR +rerank':>{COL_MR-2}} "
        f"| {'Delta':>{COL_D-2}} |"
    )
    print(border_head)

    hybrid_scores:  list[float] = []
    rerank_scores:  list[float] = []

    for query, keyword in TEST_QUERIES:
        try:
            hybrid_20  = retriever.search(query, top_k=20)
            reranked_5 = reranker.rerank(query, hybrid_20, top_k=5)
            hybrid_5   = hybrid_20[:5]

            mrr_h = mrr_at_k(hybrid_5,   keyword, k=5)
            mrr_r = mrr_at_k(reranked_5, keyword, k=5)
        except Exception as e:
            logging.warning(f"Benchmark query failed: {query!r} -- {e}")
            mrr_h, mrr_r = 0.0, 0.0

        hybrid_scores.append(mrr_h)
        rerank_scores.append(mrr_r)

        delta = mrr_r - mrr_h
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        q_display = (query[:38] + "..") if len(query) > 40 else query

        print(
            f"| {q_display:<{COL_Q-2}} "
            f"| {mrr_h:>{COL_MH-2}.2f} "
            f"| {mrr_r:>{COL_MR-2}.2f} "
            f"| {delta_str:>{COL_D-2}} |"
        )

    print(border_bottom)

    mean_h = sum(hybrid_scores) / len(hybrid_scores)
    mean_r = sum(rerank_scores) / len(rerank_scores)
    mean_d = mean_r - mean_h
    mean_d_str = f"+{mean_d:.2f}" if mean_d >= 0 else f"{mean_d:.2f}"

    print(
        f"\n  MEAN MRR@5 :  hybrid = {mean_h:.4f}  |  +rerank = {mean_r:.4f}  |  improvement = {mean_d_str}"
    )

    if mean_d > 0:
        print("  [OK] Reranker improves retrieval quality")
    elif mean_d == 0:
        print("  ➖ Reranker has no net effect on this corpus")
    else:
        print("  [WARN]️  Reranker degrades results -- check model API connectivity")

    print()
