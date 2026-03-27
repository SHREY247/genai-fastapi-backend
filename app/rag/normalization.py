"""
rag/normalization.py
--------------------
Converts raw loader output into a unified Document schema.

Every document — regardless of whether it came from a markdown file,
a PDF page, or any future source — gets normalized into the same shape.
This makes the rest of the pipeline (chunking, embedding, retrieval)
completely source-agnostic.

Teaching point:
    Normalize early, benefit everywhere. A schema is a contract
    between your ingestion layer and everything downstream.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class Document:
    """
    Unified internal document representation.

    Every piece of content in the system becomes a Document,
    regardless of its original format.

    Fields:
        source:      filename or identifier (e.g., "amazon.md")
        source_type: category of the source (e.g., "public_repo", "private_pdf")
        company:     which company this content relates to
        text:        the actual text content
        page:        page number (for PDFs; None for other formats)
        metadata:    any additional key-value pairs
    """
    source: str
    source_type: str
    company: str
    text: str
    page: Optional[int] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Converts the Document to a plain dict for downstream use."""
        return asdict(self)


def normalize_documents(raw_docs: List[Dict]) -> List[Document]:
    """
    Converts a list of raw loader dicts into normalized Document objects.

    This function:
      1. Validates that required fields are present
      2. Fills in defaults for optional fields
      3. Returns a clean list of Document objects

    Args:
        raw_docs: list of dicts from loaders (load_markdown, load_pdf, etc.)

    Returns:
        List of Document objects.
    """
    documents = []

    for i, raw in enumerate(raw_docs):
        # Validate required fields
        if "text" not in raw or not raw["text"].strip():
            print(f"  [Normalization] Skipping empty document at index {i}")
            continue

        doc = Document(
            source=raw.get("source", "unknown"),
            source_type=raw.get("source_type", "unknown"),
            company=raw.get("company", "Unknown"),
            text=raw["text"].strip(),
            page=raw.get("page"),
            metadata=raw.get("metadata", {}),
        )
        documents.append(doc)

    print(
        f"[Normalization] Normalized {len(documents)} document(s) "
        f"from {len(raw_docs)} raw input(s)"
    )

    # Print a summary by source type
    by_type = {}
    for doc in documents:
        by_type[doc.source_type] = by_type.get(doc.source_type, 0) + 1
    for src_type, count in sorted(by_type.items()):
        print(f"  - {src_type}: {count} document(s)")

    return documents
