from dotenv import load_dotenv
load_dotenv()

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

from retrieval.embedder import CorpusEmbedder
from retrieval.bm25_index import BM25Index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    # -- Validate API key ------------------------------------------------------
    if not os.environ.get("HF_API_KEY"):
        print("ERROR: HF_API_KEY is not set. Add it to your .env file and retry.", file=sys.stderr)
        sys.exit(1)

    # -- Args ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Embed corpus chunks via HF API and build a BM25 index."
    )
    parser.add_argument("--chunks-dir", default="data/chunks/",
                        help="Directory containing JSON chunk files")
    parser.add_argument("--output-dir", default="data/embeddings/",
                        help="Directory to write embeddings and BM25 index")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    output_dir = Path(args.output_dir)

    # -- Load chunks -----------------------------------------------------------
    if not chunks_dir.exists():
        logging.error(f"Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    chunk_files = sorted(chunks_dir.glob("*.json"))
    all_chunks: list[dict] = []
    for fp in chunk_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                all_chunks.extend(data)
        except Exception as e:
            logging.error(f"Failed to load {fp}: {e}")

    if not all_chunks:
        logging.error("No chunks loaded -- aborting.")
        sys.exit(1)

    logging.info(f"Loaded {len(all_chunks)} chunks from {len(chunk_files)} files")

    # -- Create output dir -----------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "all_chunks_embedded.json"
    bm25_path = output_dir / "bm25_index.pkl"

    # -- Embed -----------------------------------------------------------------
    start = time.time()
    embedder = CorpusEmbedder()
    embedded_chunks = embedder.embed_chunks(all_chunks)
    elapsed = time.time() - start

    embedder.save_embeddings(embedded_chunks, str(embeddings_path))

    # -- BM25 index ------------------------------------------------------------
    bm25 = BM25Index()
    bm25.build(embedded_chunks)
    bm25.save(str(bm25_path))

    # -- Summary ---------------------------------------------------------------
    sep = "=" * 36
    print(f"\n{sep}")
    print("Corpus Embedding Complete")
    print(sep)
    print(f"Total chunks embedded : {len(embedded_chunks)}")
    print(f"Embedding dimension   : 1024")
    print(f"Time taken            : {elapsed:.1f}s")
    print(f"Output JSON           : {embeddings_path}")
    print(f"BM25 index            : {bm25_path}")
    print(f"{sep}\n")

    # -- Smoke tests -----------------------------------------------------------
    print("--- TEST 1 -- BM25 German query -----------------------------------")
    results_de = bm25.search("N-1 Sicherheitskriterium Uebertragungsnetz", top_k=3)
    if results_de:
        for rank, r in enumerate(results_de, 1):
            print(f"  [{rank}] {r['chunk_id']}  score={r['score']:.4f}  \"{r['text'][:60]}\"")
    else:
        print("  (no results)")

    print("\n--- TEST 2 -- BM25 English query ----------------------------------")
    results_en = bm25.search("N-1 security criterion transmission", top_k=3)
    if results_en:
        for rank, r in enumerate(results_en, 1):
            print(f"  [{rank}] {r['chunk_id']}  score={r['score']:.4f}  \"{r['text'][:60]}\"")
        top_de_ids = {r["chunk_id"] for r in results_de}
        top_en_ids = {r["chunk_id"] for r in results_en}
        overlap = top_de_ids & top_en_ids
        print(f"  Overlap with TEST 1 top results: {overlap if overlap else 'none'}")
    else:
        print("  (no results)")

    print("\n--- TEST 3 -- Dense embedding -------------------------------------")
    vec = embedder.embed_query("Was ist das N-1 Sicherheitskriterium?")
    norm = sum(x ** 2 for x in vec) ** 0.5
    print(f"  Embedding dim  : {len(vec)}")
    print(f"  First 3 values : {vec[:3]}")
    print(f"  Vector norm    : {norm:.4f}")

    print("\n--- TEST 4 -- Cross-lingual cosine similarity ---------------------")
    a = embedder.embed_query("N-1 security criterion transmission")
    b = embedder.embed_query("N-1 Sicherheitskriterium Uebertragungsnetz")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x ** 2 for x in a) ** 0.5
    norm_b = sum(x ** 2 for x in b) ** 0.5
    sim = dot / (norm_a * norm_b)
    print(f"  Cross-lingual similarity (EN<->DE): {sim:.4f}")
    if sim >= 0.80:
        print("  [OK] Cross-lingual retrieval confirmed")
    elif sim < 0.70:
        print("  WARN️  WARNING: Low similarity -- verify HF_API_KEY and model name")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.error("Unexpected error", exc_info=True)
        raise
