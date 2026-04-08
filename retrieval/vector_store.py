import json, logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION_NAME = "grid_ops_chunks"
VECTOR_DIM = 1024


class GridKnowledgeStore:
    """
    Wraps a local Qdrant vector store for dense retrieval over ENTSO-E chunk embeddings.

    Uses Qdrant's on-disk mode -- no Docker, no server, no RAM overhead.
    Data is persisted at `data/qdrant/` between runs.

    Why Qdrant over FAISS / numpy dot-product:
    - Native filtering support (by slug, page, heading) -- useful for scoped queries
    - Persistent on disk -- no need to re-index on every run
    - Cosine similarity built in -- correct for normalised multilingual-e5 vectors
    - Simple Python client -- no C++ extensions or platform headaches
    """

    def __init__(self, path: str = "data/qdrant"):
        self.path = path
        Path(path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=path)
        self._ensure_collection()
        logging.info(
            f"GridKnowledgeStore ready -- collection: {COLLECTION_NAME}, dim: {VECTOR_DIM}, path: {path}"
        )

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            logging.info(f"Created Qdrant collection: {COLLECTION_NAME}")

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    def index_chunks(self, chunks: list[dict]) -> None:
        """Upsert chunks that have an 'embedding' field into Qdrant."""
        points = []
        for i, chunk in enumerate(chunks):
            if "embedding" not in chunk:
                continue
            payload = {
                "chunk_id":     chunk.get("chunk_id", f"chunk_{i}"),
                "slug":         chunk.get("slug", ""),
                "heading_path": chunk.get("heading_path", ""),
                "text":         chunk.get("text", ""),
                "page_start":   chunk.get("page_start", 0),
                "page_end":     chunk.get("page_end", chunk.get("page_start", 0)),
                "token_count":  chunk.get("token_count", 0),
            }
            points.append(PointStruct(id=i, vector=chunk["embedding"], payload=payload))

        # Upsert in batches of 100 to avoid request size limits
        for i in range(0, len(points), 100):
            self.client.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 100])

        logging.info(f"Indexed {len(points)} chunks into Qdrant collection '{COLLECTION_NAME}'")

    def load_from_embeddings(self, embeddings_path: str) -> None:
        """
        Load chunks from all_chunks_embedded.json and index them.
        Skips indexing if collection already contains vectors (idempotent).
        """
        if self.count() > 0:
            logging.info(f"Collection already has {self.count()} vectors -- skipping index")
            return

        with open(embeddings_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        self.index_chunks(chunks)

    def search(self, query_vector: list[float], top_k: int = 20) -> list[dict]:
        """
        Dense cosine similarity search.
        Returns top_k results sorted by score descending.
        """
        results = self.client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            {
                "chunk_id":     r.payload["chunk_id"],
                "score":        r.score,
                "text":         r.payload["text"],
                "heading_path": r.payload["heading_path"],
                "slug":         r.payload["slug"],
                "page_start":   r.payload["page_start"],
                "page_end":     r.payload["page_end"],
            }
            for r in results
        ]
