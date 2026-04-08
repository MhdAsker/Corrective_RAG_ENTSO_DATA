import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MAX_TOKENS = 512
CHARS_PER_TOKEN = 4  # heuristic
MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN
OVERLAP_CHARS = 200

def parse_args():
    parser = argparse.ArgumentParser(description="Chunk extracted PDF JSON into smaller semantic chunks for retrieval.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing extracted JSON files")
    parser.add_argument("--output", type=str, required=True, help="Directory to output chunk JSON files")
    return parser.parse_args()

def chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Splits text into chunks of max_chars size with overlap_chars overlap."""
    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # If we can fit the remainder, just take it
        if text_len - start <= max_chars:
            chunks.append(text[start:])
            break

        # Find a good breaking point
        end = start + max_chars
        
        # Try to break on a double newline (paragraph boundary)
        break_point = text.rfind('\n\n', start, end)
        
        # If no double newline, try single newline
        if break_point == -1 or break_point <= start + overlap_chars:
            break_point = text.rfind('\n', start, end)

        # If no newline, try a period
        if break_point == -1 or break_point <= start + overlap_chars:
            break_point = text.rfind('. ', start, end)

        # If no period, just break at max_chars
        if break_point == -1 or break_point <= start + overlap_chars:
            break_point = end
        else:
            # Include the breaking character
            break_point += 1

        chunks.append(text[start:break_point].strip())
        
        # Set start for the next chunk, effectively overlapping
        start = break_point - overlap_chars

    return chunks

def process_file(input_file: Path, output_dir: Path):
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            pages = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load {input_file}: {e}")
        return

    if not pages:
        return

    # Group pages by heading
    # Because a section can span multiple pages, we first accumulate all text for a particular heading.
    sections = []
    current_section = None

    for page in pages:
        # Ignore TOC pages for retrieval purposes if possible
        if page.get("is_toc_page"):
            continue

        page_heading = page.get("detected_heading")
        # If no heading was found on this page, inherit the last heading
        if not page_heading and current_section:
            heading = current_section["heading"]
        else:
            heading = page_heading

        if current_section and current_section["heading"] == heading:
            current_section["text"] += "\n" + page.get("raw_text", "")
            # We track the starting page of the section
        else:
            if current_section:
                sections.append(current_section)
            current_section = {
                "heading": heading,
                "text": page.get("raw_text", ""),
                "page_start": page.get("page_number"),
                "slug": page.get("slug")
            }
            
    if current_section:
        sections.append(current_section)

    all_chunks = []
    chunk_idx = 0

    for section in sections:
        heading = section["heading"]
        if not heading:
            heading = "Preamble"
            
        full_text = section["text"].strip()
        if not full_text:
            continue

        raw_chunks = chunk_text(full_text, MAX_CHARS, OVERLAP_CHARS)
        
        for i, chunk_txt in enumerate(raw_chunks):
            # Estimate token count
            token_count = len(chunk_txt) // CHARS_PER_TOKEN
            
            # Simple heuristic to extract potential cross-references
            cross_refs = []
            if "Article " in chunk_txt:
                # Basic mock logic for extracting referenced articles
                import re
                refs = re.findall(r"Article\s+(\d+)", chunk_txt)
                if refs:
                    cross_refs.extend([f"Article {r}" for r in set(refs)])

            all_chunks.append({
                "slug": section["slug"],
                "chunk_id": f"chunk_{chunk_idx}",
                "text": chunk_txt,
                "heading_path": heading,
                "page_start": section["page_start"],
                "token_count": token_count,
                "is_overlap": True if i > 0 else False,
                "cross_references": cross_refs
            })
            chunk_idx += 1

    output_file = output_dir / input_file.name
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
        logging.info(f"Chunked {input_file.name}: {len(all_chunks)} chunks generated.")
    except Exception as e:
        logging.error(f"Failed to write chunks to {output_file}: {e}")

def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logging.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    extracted_files = list(input_dir.glob("*.json"))
    for fpath in extracted_files:
        process_file(fpath, output_dir)

if __name__ == "__main__":
    main()
