"""
rag/prompt_builder.py
---------------------
Builds grounded prompts for the LLM from retrieved context.

Two functions:
1. build_context()  - formats retrieved chunks into a readable context block
2. build_prompt()   - wraps the context and query into a grounded instruction prompt
"""

from typing import List, Dict

GROUNDED_INSTRUCTION = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context below. Do not use any outside knowledge.\n\n"
    "If the answer is not present in the context, respond with exactly:\n"
    '"I don\'t know based on the provided documents."\n\n'
)


def build_context(results: List[Dict]) -> str:
    """
    Formats retrieved results into a numbered context block.

    Args:
        results: list of dicts from the retriever (rank, source, chunk_id, text, score)

    Returns:
        A formatted context string.
    """
    context_parts = []
    for r in results:
        header = f"[Source: {r['source']} | Chunk: {r['chunk_id']} | Score: {r['score']:.4f}]"
        context_parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_prompt(query: str, context: str) -> str:
    """
    Combines the grounded instruction, context, and query into a final prompt.

    Args:
        query:   the user's question
        context: the formatted context string from build_context()

    Returns:
        The complete prompt string to send to the LLM.
    """
    prompt = (
        f"{GROUNDED_INSTRUCTION}"
        f"CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION:\n"
        f"{query}\n\n"
        f"ANSWER:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Session 8: Source-aware prompt building for multi-source RAG
# ---------------------------------------------------------------------------

INTERVIEW_INSTRUCTION = (
    "You are an Interview Prep Assistant. Answer the user's question using ONLY the "
    "provided context below. Each piece of context is labeled with its source type "
    "(Public or Private) and the company it relates to.\n\n"
    "When answering:\n"
    "- Cite the source type and company for each piece of information\n"
    "- If public and private sources conflict, prefer the private/2025 update "
    "and mention that the information has been updated\n"
    "- If the answer is not present in the context, respond with exactly:\n"
    '"I don\'t have enough information to answer this based on the provided sources."\n\n'
)


def build_interview_context(results: List[Dict]) -> str:
    """
    Formats retrieved results with source-type and company labels.

    Each chunk is labeled like:
        [Public: Amazon | Source: amazon.md | Chunk: chunk-3 | Score: 0.7812]
    or:
        [Private: Google | Source: company_hiring_updates_2025.pdf | Page: 3 | Score: 0.8123]

    This labeling helps the LLM distinguish between sources in its answer.

    Args:
        results: list of dicts from the retriever

    Returns:
        A formatted context string with source labels.
    """
    context_parts = []

    for r in results:
        # Determine the display label based on source type
        source_type = r.get("source_type", "unknown")
        company = r.get("company", "Unknown")
        label = "Private" if "private" in source_type.lower() else "Public"

        # Build the header
        header_parts = [
            f"{label}: {company}",
            f"Source: {r.get('source', 'unknown')}",
        ]

        # Add page number for PDFs
        page = r.get("page")
        if page is not None:
            header_parts.append(f"Page: {page}")

        header_parts.append(f"Chunk: {r.get('chunk_id', 'N/A')}")
        header_parts.append(f"Score: {r.get('score', 0):.4f}")

        header = "[" + " | ".join(header_parts) + "]"
        context_parts.append(f"{header}\n{r['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_interview_prompt(query: str, context: str) -> str:
    """
    Builds a source-aware prompt for the interview prep assistant.

    Args:
        query:   the user's question
        context: the formatted context from build_interview_context()

    Returns:
        The complete prompt string to send to the LLM.
    """
    prompt = (
        f"{INTERVIEW_INSTRUCTION}"
        f"CONTEXT:\n"
        f"{context}\n\n"
        f"QUESTION:\n"
        f"{query}\n\n"
        f"ANSWER:"
    )
    return prompt
