import re
from typing import List
import pypdf


def read_text_file(filepath: str) -> str:
    """Reads content from a text file with resilient encoding fallbacks."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _clean_extracted_pdf_page(page_text: str) -> str:
    """Normalizes single-newline PDF text into clean paragraphs and strips standalone page numbers."""
    if not page_text:
        return ""
    lines = page_text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip standalone page numbers like "14", "- 14 -", "Page 14 of 220"
        if re.match(r"^(?:[-–—]\s*)?(?:page\s+)?\d+(?:\s+of\s+\d+)?(?:\s*[-–—])?$", stripped, re.IGNORECASE):
            continue
        cleaned_lines.append(stripped)

    # Reconstruct paragraphs: join wrapped lines unless a line ends with sentence punctuation
    # followed by an uppercase/quote start or an empty line.
    paragraphs = []
    buffer = []
    for line in cleaned_lines:
        if not line:
            if buffer:
                paragraphs.append(" ".join(buffer))
                buffer = []
            continue
        buffer.append(line)
        # Heuristic paragraph boundary for PDFs without blank lines
        if len(" ".join(buffer)) > 280 and re.search(r'[.!?]["”\']?$', line):
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer))

    return "\n\n".join(paragraphs)


def read_pdf_file(filepath: str) -> str:
    """Extracts and cleans text from a PDF file safely."""
    pages_out = []
    try:
        reader = pypdf.PdfReader(filepath)
        for page in reader.pages:
            raw_page = page.extract_text() or ""
            cleaned = _clean_extracted_pdf_page(raw_page)
            if cleaned.strip():
                pages_out.append(cleaned)
    except Exception as e:
        raise RuntimeError(f"Error reading PDF '{filepath}': {e}") from e
    return "\n\n".join(pages_out)


def _split_oversized_unit(text: str, max_chars: int) -> List[str]:
    """Splits a long paragraph by single newlines, then sentences, then words so no chunk exceeds max_chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Try single newlines first
    if "\n" in text:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        if len(lines) > 1:
            return _pack_units(lines, max_chars, joiner="\n")

    # Split by sentence boundaries (preserving quotes after punctuation)
    sentences = [
        s.strip()
        for s in re.split(r'(?<=[.!?]["”\'])\s+|(?<=[.!?])\s+', text)
        if s.strip()
    ]
    if len(sentences) > 1:
        return _pack_units(sentences, max_chars, joiner=" ")

    # Hard fallback: split by words
    words = text.split()
    return _pack_units(words, max_chars, joiner=" ")


def _pack_units(units: List[str], max_chars: int, joiner: str = "\n\n") -> List[str]:
    """Packs text units greedily without ever emitting empty chunks or exceeding max_chars."""
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    sep_len = len(joiner)

    for unit in units:
        unit = unit.strip()
        if not unit:
            continue

        if len(unit) > max_chars:
            if current:
                chunks.append(joiner.join(current))
                current = []
                current_len = 0
            sub_chunks = _split_oversized_unit(unit, max_chars)
            chunks.extend(sub_chunks)
            continue

        added_len = len(unit) if not current else sep_len + len(unit)
        if current and (current_len + added_len > max_chars):
            chunks.append(joiner.join(current))
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += added_len

    if current:
        chunks.append(joiner.join(current))

    return [c for c in chunks if c.strip()]


def chunk_text(text: str, max_chars: int = 2800) -> List[str]:
    """
    Splits raw book text into clean, non-empty chunks guaranteed <= max_chars,
    respecting paragraph and sentence boundaries even on single-newline PDFs.
    """
    if not text or not text.strip():
        return []
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    return _pack_units(paragraphs, max_chars=max_chars, joiner="\n\n")
