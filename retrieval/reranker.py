import os, logging, time
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

# BAAI/bge-reranker-base: available on HF router, supports text-ranking pipeline,
# multilingual -- handles DE/EN mixed queries better than MS-MARCO MiniLM.
RERANKER_MODEL = "BAAI/bge-reranker-base"


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a cross-encoder via HF InferenceClient.
    No local model download -- runs on HF servers.

    Why cross-encoder re-ranking over bi-encoder retrieval alone:
    - Bi-encoder (dense retrieval): encodes query and document SEPARATELY,
      then compares vectors -- fast but less accurate because query and document
      tokens never see each other during encoding
    - Cross-encoder: encodes (query, document) JOINTLY as a single sequence --
      every query token can attend to every document token (full cross-attention)
      -- far more accurate relevance scoring
    - Typical improvement: MRR@5 increases 15-25% after re-ranking on
      legal and technical document collections
    - BAAI/bge-reranker-base: multilingual cross-encoder, strong on DE/EN text,
      available on HF router text-ranking pipeline

    Degraded mode: if the API is unavailable, rerank() returns chunks in their
    original RRF order with dummy scores rather than crashing the pipeline.

    Pipeline position:
      HybridRetriever (top-20 candidates) -> CrossEncoderReranker (top-5) -> LLM context
    """

    def __init__(self):
        self.available = False
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ.get("HF_API_KEY", ""),
        )
        self._verify_connection()

    def _verify_connection(self) -> None:
        try:
            scores = self._score_batch("test query", ["test passage"])
            if isinstance(scores, list) and len(scores) > 0:
                self.available = True
                logging.info(f"Reranker ready -- model: {RERANKER_MODEL}, backend: HF API")
            else:
                logging.warning(f"Reranker returned unexpected response -- running in passthrough mode")
        except Exception as e:
            logging.warning(f"Reranker unavailable ({e}) -- running in passthrough mode (RRF order preserved)")

    def _score_batch(self, query: str, texts: list[str]) -> list[float]:
        # InferenceClient.text_ranking returns list of {"score": float, "index": int}
        for attempt in range(2):
            try:
                results = self.client.text_ranking(
                    query=query,
                    texts=texts,
                    model=RERANKER_MODEL,
                )
                # results is list of RankedDocument objects or dicts with .score
                scores_by_index = {}
                for item in results:
                    idx = item.index if hasattr(item, "index") else item["index"]
                    sc  = item.score  if hasattr(item, "score")  else item["score"]
                    scores_by_index[idx] = float(sc)
                return [scores_by_index.get(i, 0.0) for i in range(len(texts))]

            except Exception as e:
                err = str(e)
                if "503" in err or "loading" in err.lower():
                    logging.info("Reranker model loading, waiting 20s...")
                    time.sleep(20)
                elif "429" in err or "rate" in err.lower():
                    logging.warning("Reranker rate limited, waiting 60s...")
                    time.sleep(60)
                elif attempt == 1:
                    raise
                else:
                    time.sleep(5)

        raise RuntimeError("Reranker failed after retries")

    def rerank(self, query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
        """
        Score all chunks against the query, sort by score, return top_k.

        Each returned chunk gains:
          rerank_score   float  -- cross-encoder relevance score
          original_rank  int    -- 1-indexed position before reranking
          new_rank       int    -- 1-indexed position after reranking
          rank_delta     int    -- original_rank - new_rank
                                  positive = moved up, negative = moved down
        """
        try:
            texts = [c["text"][:512] for c in chunks]
            scores = self._score_batch(query, texts)

            # Annotate original rank
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i]) if i < len(scores) else 0.0
                chunk["original_rank"] = i + 1

            # Sort descending by rerank_score
            reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

            # Annotate new rank and delta
            for i, chunk in enumerate(reranked):
                chunk["new_rank"] = i + 1
                chunk["rank_delta"] = chunk["original_rank"] - chunk["new_rank"]

            return reranked[:top_k]

        except Exception as e:
            logging.warning(f"Reranking failed ({e}), returning top_k unranked")
            return chunks[:top_k]

    def rerank_with_analysis(self, query: str, chunks: list[dict], top_k: int = 5) -> dict:
        """
        Rerank and return a structured analysis dict:
        {
            "reranked":           list[dict],   top_k chunks after reranking
            "dropped":            list[dict],   chunks outside top_k
            "biggest_mover_up":   dict | None,  chunk with highest rank_delta
            "biggest_mover_down": dict | None,  chunk with lowest rank_delta
        }
        """
        reranked = self.rerank(query, chunks, top_k=top_k)

        reranked_ids = {c["chunk_id"] for c in reranked}
        dropped = [c for c in chunks if c["chunk_id"] not in reranked_ids]

        # Movers are meaningful only when reranking actually ran (rank_delta present)
        annotated = [c for c in reranked if "rank_delta" in c]

        biggest_mover_up = (
            max(annotated, key=lambda c: c["rank_delta"]) if annotated else None
        )
        biggest_mover_down = (
            min(annotated, key=lambda c: c["rank_delta"]) if annotated else None
        )

        return {
            "reranked":           reranked,
            "dropped":            dropped,
            "biggest_mover_up":   biggest_mover_up,
            "biggest_mover_down": biggest_mover_down,
        }
