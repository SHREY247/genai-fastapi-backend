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
