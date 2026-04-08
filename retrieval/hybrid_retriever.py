import logging
from retrieval.embedder import CorpusEmbedder
from retrieval.bm25_index import BM25Index
from retrieval.vector_store import GridKnowledgeStore


class HybridRetriever:
    """
    Combines dense (Qdrant cosine) + sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF).

    RRF formula:  score(d) = Σ  1 / (k + rank_i(d))
    where k = 60 is a standard smoothing constant, and rank_i is the
    position of document d in the i-th ranked list (1-indexed).

    Why RRF over weighted sum of scores:
    - Score scales differ wildly: BM25 ~ [0, 20], cosine ~ [0, 1]
    - RRF uses only rank positions -- scale-invariant and robust
    - No hyperparameter tuning needed beyond k (60 is the standard default)
    - Well-studied: consistently outperforms naive score fusion in BEIR benchmarks

    Pipeline position:
      CorpusEmbedder.embed_query()  ->  GridKnowledgeStore.search()  -+
                                                                      +- RRF -> top-20
      BM25Index.search()                                             -+
    """

    RRF_K = 60

    def __init__(
        self,
        store: GridKnowledgeStore,
        embedder: CorpusEmbedder,
        bm25: BM25Index,
    ):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Hybrid retrieval: dense + BM25 fused with RRF.

        Returns top_k results sorted by rrf_score descending.
        Each result dict: {chunk_id, rrf_score, text, heading_path, slug, page_start, page_end}
        """
        # -- Dense retrieval ----------------------------------------------------
        query_vec = self.embedder.embed_query(query)
        dense_results = self.store.search(query_vec, top_k=top_k)

        # -- Sparse retrieval ---------------------------------------------------
        sparse_results = self.bm25.search(query, top_k=top_k)

        # -- Merge chunk metadata into a single lookup --------------------------
        chunks_map: dict[str, dict] = {}
        for r in dense_results:
            chunks_map[r["chunk_id"]] = r
        for r in sparse_results:
            if r["chunk_id"] not in chunks_map:
                chunks_map[r["chunk_id"]] = r

        # -- RRF score accumulation ---------------------------------------------
        rrf_scores: dict[str, float] = {}

        for rank, r in enumerate(dense_results, 1):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)

        for rank, r in enumerate(sparse_results, 1):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)

        # -- Sort and assemble results ------------------------------------------
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            chunk = chunks_map[cid].copy()
            chunk["rrf_score"] = round(rrf_scores[cid], 6)
            chunk.setdefault("page_end", chunk.get("page_start", 0))
            # Remove raw retrieval scores -- only expose rrf_score downstream
            chunk.pop("score", None)
            results.append(chunk)

        logging.info(
            f"HybridRetriever: {len(dense_results)} dense + {len(sparse_results)} BM25 "
            f"-> {len(results)} fused (RRF k={self.RRF_K})"
        )
        return results
