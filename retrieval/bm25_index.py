import pickle, re, logging
from rank_bm25 import BM25Okapi

class BM25Index:
    def __init__(self):
        self.index = None
        self.chunk_ids = []
        self.chunks_map = {}  # chunk_id -> full chunk dict

    def _tokenize(self, text: str) -> list[str]:
        if not text:
            return []
        
        text = text.lower()
        # Split on whitespace and punctuation, including section signs (§) correctly
        tokens = re.split(r'[\s\.,;:!?()\[\]§]+', text)
        return [t for t in tokens if len(t) >= 2]

    def build(self, chunks: list[dict]) -> None:
        tokenized_corpus = []
        
        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{idx}")
            chunk["chunk_id"] = chunk_id
            
            text = chunk.get("text", "")
            tokens = self._tokenize(text)
            
            tokenized_corpus.append(tokens)
            self.chunk_ids.append(chunk_id)
            self.chunks_map[chunk_id] = chunk

        self.index = BM25Okapi(tokenized_corpus)
        logging.info(f"BM25 index built: {len(chunks)} documents")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if not self.index:
            logging.error("Cannot search. BM25Index is not built yet.")
            return []
            
        tokenized_query = self._tokenize(query)
        doc_scores = self.index.get_scores(tokenized_query)
        
        # Argsort descending manually, filter 0 scores explicitly
        scored_docs = [(self.chunk_ids[i], score) for i, score in enumerate(doc_scores) if score > 0.0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in scored_docs[:top_k]:
            chunk_ref = self.chunks_map[chunk_id]
            res = {
                "chunk_id": chunk_id,
                "score": float(score),
                "text": chunk_ref.get("text", ""),
                "heading_path": chunk_ref.get("heading_path", "Preamble"),
                "slug": chunk_ref.get("slug", "unknown"),
                "page_start": chunk_ref.get("page_start", chunk_ref.get("page_number", 0))
            }
            results.append(res)
            
        return results

    def save(self, path: str) -> None:
        from pathlib import Path
        output_file = Path(path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'wb') as f:
            pickle.dump(self, f)
        logging.info(f"BM25 index saved → {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logging.info(f"BM25 index loaded from {path}: {len(obj.chunk_ids)} documents")
        return obj
