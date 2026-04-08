import logging
import argparse
import json
import re
from pathlib import Path
from collections import Counter
import pdfplumber

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Heading regex rules in priority order
HEADING_RULES = [
    (re.compile(r"^(CHAPTER\s+(?:[IVX]+|\d+))", re.IGNORECASE), 1),
    (re.compile(r"^(Article\s+\d+)", re.IGNORECASE), 2),
    (re.compile(r"^(\d+\.\d+(?:\.\d+)?)\s*"), 3)
]

TOC_PATTERN = re.compile(r"(?:Article\s+\d+).*\d{1,3}$", re.IGNORECASE)
PAGE_NUM_PATTERN = re.compile(r"^(\d+|Page\s+\d+\s+of\s+\d+)$", re.IGNORECASE)

def extract_document(pdf_path: str, slug: str) -> list[dict]:
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        logging.error(f"File not found: {pdf_path}")
        return []

    # First pass: collect all lines to identify headers/footers
    all_lines = []
    page_lines_map = {}
    
    with pdfplumber.open(pdf_path_obj) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                page_lines_map[i] = lines
                
                # Only add lines >= 4 chars to our global frequency count to avoid stripping common short words
                for line in lines:
                    if len(line) >= 4:
                        all_lines.append(line)
            else:
                page_lines_map[i] = []

    # Frequency analysis for headers/footers (> 30% of pages)
    line_counts = Counter(all_lines)
    header_footer_threshold = num_pages * 0.30
    
    # Only filter common lines if we have more than 1 page
    common_lines = set()
    if num_pages > 1:
        common_lines = {line for line, count in line_counts.items() if count > header_footer_threshold}

    extracted_pages = []

    # Second pass: clean and structure
    with pdfplumber.open(pdf_path_obj) as pdf:
        for page_num_0_idx, page in enumerate(pdf.pages):
            page_number = page_num_0_idx + 1
            raw_lines = page_lines_map.get(page_num_0_idx, [])
            
            cleaned_lines = []
            toc_matches = 0
            
            detected_heading = None
            heading_level = None

            for line in raw_lines:
                # Strip headers/footers, page numbers, and very short lines
                if len(line) < 4:
                    continue
                if line in common_lines:
                    continue
                if PAGE_NUM_PATTERN.match(line):
                    continue

                cleaned_lines.append(line)
                
                # Check for TOC entries
                if TOC_PATTERN.search(line):
                    toc_matches += 1
                
                # Check for headings (only take the first highest-priority heading on the page)
                if not detected_heading:
                    for pattern, level in HEADING_RULES:
                        match = pattern.search(line)
                        if match:
                            detected_heading = match.group(1).strip()
                            heading_level = level
                            break
            
            # Combine cleaned lines into raw text
            raw_text = "\n".join(cleaned_lines)
            word_count = len(raw_text.split())
            is_toc_page = toc_matches > 8
            
            extracted_pages.append({
                "slug": slug,
                "page_number": page_number,
                "raw_text": raw_text,
                "detected_heading": detected_heading,
                "heading_level": heading_level,
                "is_toc_page": is_toc_page,
                "word_count": word_count
            })

    return extracted_pages

def main():
    parser = argparse.ArgumentParser(description="Extract text and identify structure from regulatory PDFs.")
    parser.add_argument("--input", type=str, required=True, help="Directory containing raw PDFs and manifest.json")
    parser.add_argument("--output", type=str, required=True, help="Directory to save extracted JSONs")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = input_dir.parent / "manifest.json"
    if not manifest_path.exists():
        logging.error(f"Manifest not found at {manifest_path}")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    for doc in manifest:
        if doc.get("status") != "ok":
            logging.warning(f"Skipping {doc['slug']} due to failed download status.")
            continue
            
        slug = doc["slug"]
        pdf_path = doc["local_path"]
        
        logging.info(f"Processing {slug}...")
        extracted_data = extract_document(pdf_path, slug)
        
        if extracted_data:
            n_pages = len(extracted_data)
            n_headings = sum(1 for p in extracted_data if p["detected_heading"] is not None)
            
            out_file = output_dir / f"{slug}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Extracted {slug}: {n_pages} pages, {n_headings} headings detected. Saved to {out_file}")
        else:
            logging.warning(f"No pages extracted for {slug}")

if __name__ == "__main__":
    main()
