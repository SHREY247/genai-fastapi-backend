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
