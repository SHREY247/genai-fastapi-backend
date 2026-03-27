"""
rag/chunking.py
---------------
Two strategies for creating retrieval records:

1. build_whole_document_records()  - treats each document as a single retrieval unit
2. build_chunks()                  - splits each document into overlapping chunks

Each record is a dict with:
  - source:   the original filename
  - chunk_id: a string identifier (e.g. "doc-0" or "chunk-3")
  - text:     the content of the chunk
"""

from typing import List, Dict


def build_whole_document_records(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Returns one record per document (no splitting).
    This is the 'naive' baseline for demonstrating weaker retrieval.
    """
    records = []
    for i, doc in enumerate(documents):
        records.append({
            "source": doc["source"],
            "chunk_id": f"doc-{i}",
            "text": doc["text"],
        })

    print(f"[Chunking] Created {len(records)} whole-document record(s)")
    return records


def build_chunks(
    documents: List[Dict[str, str]],
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict[str, str]]:
    """
    Splits each document into word-based chunks with overlap.

    Args:
        documents:  list of {"source": ..., "text": ...} dicts
        chunk_size: number of words per chunk
        overlap:    number of overlapping words between consecutive chunks

    Returns:
        List of {"source", "chunk_id", "text"} dicts.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    records = []

    for doc in documents:
        words = doc["text"].split()
        start = 0
        doc_chunk_index = 0
        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            records.append({
                "source": doc["source"],
                "chunk_id": f"chunk-{doc_chunk_index}",
                "text": chunk_text,
            })
            doc_chunk_index += 1
            start += chunk_size - overlap

    print(
        f"[Chunking] Created {len(records)} chunk(s) "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )
    return records


# ---------------------------------------------------------------------------
# Session 8: Metadata-aware chunking for multi-source ingestion
# ---------------------------------------------------------------------------

def build_chunks_with_metadata(
    documents,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    """
    Splits Document objects into chunks while preserving source metadata.

    This is the Session 8 evolution of build_chunks(). Instead of only
    carrying 'source' and 'chunk_id', each chunk also carries:
      - source_type (e.g., "public_repo" or "private_pdf")
      - company (e.g., "Amazon", "Google")
      - page (for PDF sources)

    Args:
        documents:  list of Document objects (from normalization.py)
        chunk_size: number of words per chunk
        overlap:    overlapping words between consecutive chunks

    Returns:
        List of dicts with: source, source_type, company, page,
                            chunk_id, text
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    records = []
    global_chunk_index = 0

    for doc in documents:
        # Support both Document objects and plain dicts
        text = doc.text if hasattr(doc, "text") else doc["text"]
        source = doc.source if hasattr(doc, "source") else doc.get("source", "unknown")
        source_type = doc.source_type if hasattr(doc, "source_type") else doc.get("source_type", "unknown")
        company = doc.company if hasattr(doc, "company") else doc.get("company", "Unknown")
        page = doc.page if hasattr(doc, "page") else doc.get("page")

        words = text.split()
        start = 0
        doc_chunk_index = 0

        while start < len(words):
            end = start + chunk_size
            chunk_text = " ".join(words[start:end])
            records.append({
                "source": source,
                "source_type": source_type,
                "company": company,
                "page": page,
                "chunk_id": f"chunk-{global_chunk_index}",
                "text": chunk_text,
            })
            global_chunk_index += 1
            doc_chunk_index += 1
            start += chunk_size - overlap

    print(
        f"[Chunking] Created {len(records)} metadata-aware chunk(s) "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )
    return records
