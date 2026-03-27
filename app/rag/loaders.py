"""
rag/loaders.py
--------------
Source-specific file loaders for the Interview Prep RAG pipeline.

Each loader reads a file in its native format and returns a list of
raw document dicts. The dicts are NOT yet normalized — that happens
in normalization.py.

Supported formats:
  - Markdown (.md)  — one document per file
  - PDF (.pdf)      — one document per page

Teaching point:
    Loaders are the "adapters" between the outside world and your system.
    Keep them simple: one loader per format, minimal transformation.
"""

import os
import re
from typing import List, Dict
from pypdf import PdfReader


def _clean_markdown(text: str) -> str:
    """
    Lightweight cleanup for raw markdown text.

    This is NOT full markdown parsing — just enough to show students
    that loading is not the same as normalization.

    Steps:
      1. Convert markdown links [text](url) to just "text"
      2. Strip excessive heading markers (### -> clean text)
      3. Collapse 3+ consecutive blank lines into 2
    """
    # Convert markdown links to plain text: [display text](url) -> display text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Clean heading markers: "## Heading" -> "Heading"
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # Collapse excessive blank lines (3+ newlines -> 2)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def load_markdown(filepath: str) -> List[Dict]:
    """
    Reads a markdown file, applies lightweight cleanup, and returns
    a single-item list with the processed content.

    The company name is inferred from the filename (e.g., "amazon.md" -> "Amazon").

    Cleanup includes:
      - Stripping markdown links to plain text
      - Normalizing heading markers
      - Collapsing excessive blank lines

    Args:
        filepath: Path to the .md file.

    Returns:
        List containing one dict with keys: source, source_type, company, text
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Markdown file not found: {filepath}")

    filename = os.path.basename(filepath)
    company = os.path.splitext(filename)[0].capitalize()

    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text = _clean_markdown(raw_text)

    print(
        f"  [Loader] Markdown: {filename} "
        f"({len(raw_text)} raw -> {len(text)} cleaned chars, company={company})"
    )

    return [{
        "source": filename,
        "source_type": "public_repo",
        "company": company,
        "page": None,
        "text": text,
    }]



def load_pdf(filepath: str) -> List[Dict]:
    """
    Reads a PDF file page-by-page and returns one dict per page.

    Args:
        filepath: Path to the .pdf file.

    Returns:
        List of dicts, one per page, with keys: source, source_type, company, page, text
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"PDF file not found: {filepath}")

    filename = os.path.basename(filepath)
    reader = PdfReader(filepath)
    documents = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()

        if not text:
            continue  # Skip empty pages

        documents.append({
            "source": filename,
            "source_type": "private_pdf",
            "company": _detect_company_from_text(text),
            "page": page_num,
            "text": text,
        })

    print(
        f"  [Loader] PDF: {filename} "
        f"({len(reader.pages)} pages, {len(documents)} with content)"
    )
    return documents


def _detect_company_from_text(text: str) -> str:
    """
    Simple heuristic to detect which company a PDF page is about.
    Looks for known company names near the top of the page text.

    This is intentionally simple — a real system might use:
      - Structured PDF metadata
      - Explicit section markers in the document
      - Named Entity Recognition (NER)

    For teaching: deterministic approaches (explicit markers) are
    almost always better than smart heuristics in production.
    """
    # Check the first 300 characters for company names
    header = text[:300].upper()
    companies = {
        "AMAZON": "Amazon",
        "GOOGLE": "Google",
        "MICROSOFT": "Microsoft",
        "ADOBE": "Adobe",
        "ORACLE": "Oracle",
    }
    for keyword, name in companies.items():
        if keyword in header:
            return name

    # Log the fallback — helps debug PDF parsing issues during demos
    preview = text[:60].replace("\n", " ")
    print(f"    [Loader] WARNING: No company detected, defaulting to 'General' — \"{preview}...\"")
    return "General"



def load_directory(data_dir: str) -> List[Dict]:
    """
    Scans a directory and loads all supported files (.md and .pdf).

    This is the main entry point for the ingestion pipeline.
    It delegates to the appropriate loader based on file extension.

    Args:
        data_dir: Path to the directory containing source files.

    Returns:
        Combined list of raw document dicts from all files.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    all_documents = []
    supported_extensions = {".md": load_markdown, ".pdf": load_pdf}

    print(f"\n[Loaders] Scanning directory: {data_dir}")

    for filename in sorted(os.listdir(data_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        filepath = os.path.join(data_dir, filename)
        loader = supported_extensions[ext]
        documents = loader(filepath)
        all_documents.extend(documents)

    if not all_documents:
        raise ValueError(f"No supported files (.md, .pdf) found in {data_dir}")

    print(f"[Loaders] Total raw documents loaded: {len(all_documents)}")
    return all_documents
