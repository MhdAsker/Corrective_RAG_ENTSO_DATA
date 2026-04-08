import os, json, logging, time
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

MODEL_ID = "intfloat/multilingual-e5-large"


class CorpusEmbedder:
    """
    Uses HuggingFace InferenceClient for embeddings -- no local model download needed.

    Why intfloat/multilingual-e5-large over Ollama/sentence-transformers:
    - Top multilingual embedding model on MTEB leaderboard
    - Native German + English support -- critical for ENTSO-E/EnWG documents
    - 1024-dim embeddings -- higher quality than mpnet (768-dim) or qwen3-embedding (768-dim)
    - Free via HF Inference API -- no GPU, no RAM overhead, runs on HF servers
    - multilingual-e5 trained specifically on multilingual retrieval pairs
    - IMPORTANT: requires "query: " prefix for queries and "passage: " prefix for documents
      This is a model requirement -- skipping it degrades retrieval quality significantly

    Uses huggingface_hub.InferenceClient (not raw requests) -- handles the HF router
    URL and token scope automatically.
    """

    EMBEDDING_DIM = 1024

    def __init__(self, skip_verification=False):
        if "HF_API_KEY" not in os.environ:
            raise RuntimeError("HF_API_KEY not set in .env")
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=os.environ["HF_API_KEY"],
        )
        if not skip_verification:
            try:
                self._verify_connection()
            except RuntimeError as e:
                logging.warning(f"Embedder connection verification skipped: {e}")
        logging.info(
            f"Embedder ready -- model: {MODEL_ID}, dim: {self.EMBEDDING_DIM}, backend: HF InferenceClient"
        )

    def _verify_connection(self) -> None:
        try:
            res = self._embed_batch(["passage: connection test"])
            if not isinstance(res, list) or len(res) == 0:
                raise RuntimeError(f"Unexpected response: {res}")
            logging.info("HF API connection verified -- multilingual-e5-large ready")
        except Exception as e:
            raise RuntimeError(f"HF API verify failed: {e}")

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(2):
            try:
                result = self.client.feature_extraction(texts, model=MODEL_ID)
                # result is a numpy array of shape (n_texts, dim) or a list
                if hasattr(result, "tolist"):
                    result = result.tolist()
                # Ensure it's list[list[float]]
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], (int, float)):
                        # Single text returned as flat list -- wrap it
                        return [result]
                    return result
                raise RuntimeError(f"Unexpected response shape: {type(result)}")
            except Exception as e:
                err = str(e)
                if "503" in err or "loading" in err.lower():
                    logging.info("Model loading, waiting 20s...")
                    time.sleep(20)
                elif "429" in err or "rate" in err.lower():
                    logging.warning("Rate limited, waiting 60s...")
                    time.sleep(60)
                elif attempt == 1:
                    raise RuntimeError(f"HF API error: {e}")
                else:
                    time.sleep(5)
        raise RuntimeError("Failed to embed batch after retries")

    def embed_chunks(self, chunks: list[dict], batch_size: int = 8) -> list[dict]:
        texts = [f"passage: {chunk.get('text', '')}" for chunk in chunks]

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks", unit="batch"):
            batch_texts = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch_texts)

            for j, emb in enumerate(batch_embeddings):
                chunks[i + j]["embedding"] = emb if isinstance(emb, list) else list(emb)

            time.sleep(0.5)

        logging.info(f"Embedded {len(chunks)} chunks, dim={self.EMBEDDING_DIM}")
        return chunks

    def embed_query(self, query: str) -> list[float]:
        res = self._embed_batch([f"query: {query}"])
        emb = res[0]
        return emb if isinstance(emb, list) else list(emb)

    def save_embeddings(self, chunks: list[dict], output_path: str) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        logging.info(f"Saved {len(chunks)} embeddings -> {output_path}")

    def load_embeddings(self, path: str) -> list[dict]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
