"""
rag/ingestion.py
----------------
Loads all .txt files from a given directory and returns
structured document records.

Each record is a dict with:
  - source: the filename
  - text:   the full file content
"""

import os
from typing import List, Dict


def load_documents(data_dir: str) -> List[Dict[str, str]]:
    """
    Reads every .txt file in `data_dir` and returns a list of document dicts.

    Args:
        data_dir: Path to the folder containing .txt files.

    Returns:
        List of {"source": filename, "text": full_content} dicts.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"source": filename, "text": text})

    if not documents:
        raise ValueError(f"No .txt files found in {data_dir}")

    print(f"[Ingestion] Loaded {len(documents)} document(s) from '{data_dir}'")
    for doc in documents:
        print(f"  - {doc['source']} ({len(doc['text'])} chars)")

    return documents
