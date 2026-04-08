import logging
import tiktoken


class ContextBuilder:
    """
    Formats re-ranked chunks into a clean context string for the LLM prompt.

    Uses "Quelle" (German: source) as the section header -- signals to the
    Groq/LLaMA LLM that this is a German-domain system and primes it for
    mixed DE/EN regulatory content.

    Token budget strategy:
    - Counts tokens for the entire formatted string (headers + text), not just text
    - Adds chunks greedily in ranked order until the budget would be exceeded
    - Always includes at least 1 chunk -- truncating its text if necessary
    - This ensures the LLM never receives an empty context
    """

    CHUNK_TEMPLATE = "---\n[Quelle: {slug} | {heading_path} | Seite {page_start}]\n{text}\n---\n"

    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def _format_chunk(self, chunk: dict, text_override: str | None = None) -> str:
        return self.CHUNK_TEMPLATE.format(
            slug=chunk.get("slug", "unknown"),
            heading_path=chunk.get("heading_path", "--"),
            page_start=chunk.get("page_start", "?"),
            text=text_override if text_override is not None else chunk.get("text", ""),
        )

    def build(self, chunks: list[dict]) -> str:
        """
        Build context string from chunks, respecting max_tokens budget.

        Greedy fill: add chunks in order until the next one would push
        the total over max_tokens. Always includes at least 1 chunk.
        """
        if not chunks:
            return ""

        parts: list[str] = []
        total_tokens = 0

        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(chunk)
            chunk_tokens = self._count_tokens(formatted)

            if i == 0:
                # Always include the first chunk -- truncate if necessary
                if chunk_tokens > self.max_tokens:
                    # Binary-search-free truncation: trim text until it fits
                    text = chunk.get("text", "")
                    while text and self._count_tokens(self._format_chunk(chunk, text)) > self.max_tokens:
                        # Drop last 50 chars at a time -- fast convergence
                        text = text[: max(0, len(text) - 50)]
                    formatted = self._format_chunk(chunk, text)
                    chunk_tokens = self._count_tokens(formatted)

                parts.append(formatted)
                total_tokens += chunk_tokens

            else:
                if total_tokens + chunk_tokens > self.max_tokens:
                    break
                parts.append(formatted)
                total_tokens += chunk_tokens

        return "\n".join(parts)

    def build_with_metadata(self, chunks: list[dict]) -> dict:
        """
        Build context and return it with usage metadata.

        Returns:
        {
            "context":        str,        formatted context string
            "chunks_used":    int,        number of chunks included
            "chunks_dropped": int,        number of chunks excluded (budget)
            "total_tokens":   int,        token count of context string
            "sources":        list[str],  unique document slugs in context
        }
        """
        if not chunks:
            return {
                "context":        "",
                "chunks_used":    0,
                "chunks_dropped": 0,
                "total_tokens":   0,
                "sources":        [],
            }

        context = self.build(chunks)
        total_tokens = self._count_tokens(context)

        # Determine which chunks made it in by re-counting
        chunks_used = 0
        running = 0
        for i, chunk in enumerate(chunks):
            formatted = self._format_chunk(chunk)
            tok = self._count_tokens(formatted)
            if i == 0:
                chunks_used += 1
                running += min(tok, self.max_tokens)
            else:
                if running + tok > self.max_tokens:
                    break
                chunks_used += 1
                running += tok

        chunks_dropped = len(chunks) - chunks_used

        sources = list(dict.fromkeys(
            chunk.get("slug", "unknown")
            for chunk in chunks[:chunks_used]
        ))

        logging.info(
            f"ContextBuilder: {chunks_used}/{len(chunks)} chunks, "
            f"{total_tokens} tokens, sources: {sources}"
        )

        return {
            "context":        context,
            "chunks_used":    chunks_used,
            "chunks_dropped": chunks_dropped,
            "total_tokens":   total_tokens,
            "sources":        sources,
        }
