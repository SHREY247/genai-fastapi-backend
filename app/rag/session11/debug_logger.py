"""
session11/debug_logger.py
--------------------------
Pretty-printing utilities for RAG pipeline observability.

Provides classroom-friendly terminal output for inspecting every stage
of a RAG pipeline run: query, rewriting, retrieval, prompt, and answer.

Design goals:
    1. Clean, scannable output with clear section headers
    2. Truncation helpers so long chunks/prompts don't flood the terminal
    3. Color-free (works in any terminal, Jupyter, or IDE console)
    4. Stateless functions — no globals, no side effects beyond printing

Usage:
    from app.rag.session11.debug_logger import print_debug_report
    print_debug_report(result_dict)
"""

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Truncation helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 300) -> str:
    """
    Truncates text to max_chars, preserving word boundaries.

    Args:
        text:      the string to truncate.
        max_chars: maximum character length.

    Returns:
        Truncated string with '...' appended if truncated.
    """
    if not text or len(text) <= max_chars:
        return text or ""
    cut = text[:max_chars].rsplit(" ", 1)[0]
    return cut + "..."


def truncate_lines(text: str, max_lines: int = 10) -> str:
    """
    Truncates text to max_lines, showing a count of omitted lines.

    Args:
        text:      the string to truncate.
        max_lines: maximum number of lines to keep.

    Returns:
        Truncated string with omitted line count.
    """
    if not text:
        return ""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = lines[:max_lines]
    omitted = len(lines) - max_lines
    kept.append(f"  ... ({omitted} more lines omitted)")
    return "\n".join(kept)


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------

SEPARATOR = "─" * 70
THICK_SEPARATOR = "═" * 70


def _print_header(title: str) -> None:
    """Prints a section header."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def print_query_section(query: str, rewritten_query: Optional[str] = None) -> None:
    """
    Prints the QUERY section with original and rewritten query.

    Args:
        query:           the original user query.
        rewritten_query: the rewritten version (None if not rewritten).
    """
    _print_header("📝 QUERY")
    print(f"  Original : {query}")
    if rewritten_query and rewritten_query != query:
        print(f"  Rewritten: {rewritten_query}")
    else:
        print("  Rewritten: (not applied)")


def print_strategy_section(strategy: str) -> None:
    """
    Prints the STRATEGY section.

    Args:
        strategy: name of the retrieval strategy used.
    """
    _print_header("⚙️  STRATEGY")
    print(f"  {strategy}")


def print_chunks_section(
    results: List[Dict],
    max_chunks: int = 5,
    max_text_chars: int = 200,
) -> None:
    """
    Prints the RETRIEVED CHUNKS section with rank, source, score, and text.

    Design: each chunk is a clearly delineated block so it reads naturally
    on a projector. Rank, source, and score are on one line; text on the next.

    Args:
        results:        list of chunk dicts from retrieval.
        max_chunks:     maximum number of chunks to display.
        max_text_chars: maximum characters per chunk text preview.
    """
    _print_header(f"📦 RETRIEVED CHUNKS  ({len(results)} returned, showing {min(len(results), max_chunks)})")

    display = results[:max_chunks]
    for i, chunk in enumerate(display, start=1):
        rank   = chunk.get("rank", i)
        source = chunk.get("source", "unknown")
        score  = chunk.get("score", 0.0)
        via    = chunk.get("retriever", "?")
        text   = truncate(chunk.get("text", ""), max_text_chars)

        # One header line, one text line — easy to read on a projector
        print(f"\n  Rank {rank}  |  {source}  |  score={score:.4f}  |  via {via}")
        print(f"  {'─'*66}")
        print(f"  {text}")

    if len(results) > max_chunks:
        remaining = len(results) - max_chunks
        print(f"\n  ... {remaining} more chunk(s) not shown (increase max_chunks to see them)")


def print_prompt_section(prompt: str, max_chars: int = 600) -> None:
    """
    Prints the PROMPT PREVIEW section with clean truncation.

    Shows the actual final prompt sent to the LLM. Long prompts are
    truncated with a clear indicator of how much was cut.

    Args:
        prompt:    the full prompt sent to the LLM.
        max_chars: maximum characters to display (default 600 ~= ~10 lines).
    """
    _print_header("📋 FINAL PROMPT  (sent to LLM)")
    if not prompt:
        print("  (no prompt — retrieval-only mode)")
        return

    lines = prompt.split("\n")
    char_count = 0
    output_lines = []
    for line in lines:
        if char_count + len(line) > max_chars:
            output_lines.append(f"  ... [{len(prompt) - max_chars}+ chars truncated — full prompt has {len(prompt)} chars]")
            break
        output_lines.append(line)
        char_count += len(line) + 1

    print("\n".join(output_lines))


def print_answer_section(answer: str, retrieval_only: bool = False, max_chars: int = 500) -> None:
    """
    Prints the ANSWER section.

    Args:
        answer:         the LLM-generated answer.
        retrieval_only: if True, shows a notice instead of an empty answer.
        max_chars:      maximum characters to display.
    """
    _print_header("💬 ANSWER")
    if retrieval_only:
        print("  [RETRIEVAL-ONLY MODE — LLM generation was skipped]")
        print("  Run without --no-answers to see the generated answer.")
    elif not answer or answer.isspace():
        print("  (no answer returned)")
    else:
        print(f"  {truncate(answer, max_chars)}")


def print_timing_section(timing: Dict) -> None:
    """
    Prints the TIMING section with per-phase breakdown.

    Args:
        timing: dict with keys like rewrite_ms, retrieval_ms, answer_ms, total_ms.
    """
    _print_header("⏱️  TIMING")
    if timing.get("rewrite_ms", 0) > 0:
        print(f"  Rewrite:   {timing['rewrite_ms']:>8.0f} ms")
    print(f"  Retrieval: {timing.get('retrieval_ms', 0):>8.0f} ms")
    print(f"  Answer:    {timing.get('answer_ms', 0):>8.0f} ms")
    print(f"  Total:     {timing.get('total_ms', 0):>8.0f} ms")


def print_metadata_section(results: List[Dict], max_items: int = 5) -> None:
    """
    Prints source metadata for retrieved chunks.

    Args:
        results:   list of chunk dicts.
        max_items: maximum number to display.
    """
    _print_header("🏷️  SOURCE METADATA")
    display = results[:max_items]
    for chunk in display:
        source = chunk.get("source", "unknown")
        source_type = chunk.get("source_type", "unknown")
        company = chunk.get("company", "unknown")
        page = chunk.get("page", None)

        parts = [f"source={source}", f"type={source_type}", f"company={company}"]
        if page is not None:
            parts.append(f"page={page}")
        print(f"  {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# Full debug report
# ---------------------------------------------------------------------------

def print_debug_report(result: Dict, verbose: bool = True) -> None:
    """
    Prints a complete debug report for a single pipeline run.

    This is the main entry point for debug output. It prints every
    observable stage of the RAG pipeline in a clean, readable format.

    Args:
        result:  structured result dict from observability.run_strategy().
                 Expected keys: query, rewritten_query, strategy, contexts,
                 context_metadata, prompt, answer, timing_ms
        verbose: if True, shows prompt preview and metadata sections.
    """
    print(f"\n{THICK_SEPARATOR}")
    print("  🔍 RAG PIPELINE DEBUG REPORT")
    print(THICK_SEPARATOR)

    print_query_section(
        result.get("query", ""),
        result.get("rewritten_query"),
    )

    print_strategy_section(result.get("strategy", "unknown"))

    print_chunks_section(result.get("contexts", []))

    if verbose:
        print_metadata_section(result.get("contexts", []))

    if result.get("prompt"):
        print_prompt_section(result["prompt"])
    elif verbose:
        print_prompt_section("")  # shows the retrieval-only notice

    print_answer_section(
        result.get("answer", ""),
        retrieval_only=result.get("retrieval_only", False),
    )

    if result.get("timing_ms"):
        print_timing_section(result["timing_ms"])

    print(f"\n{THICK_SEPARATOR}\n")


def print_comparison_table(results: List[Dict]) -> None:
    """
    Prints a side-by-side comparison table of multiple strategy runs.

    Args:
        results: list of result dicts from running multiple strategies.
    """
    print(f"\n{THICK_SEPARATOR}")
    print("  📊 STRATEGY COMPARISON")
    print(THICK_SEPARATOR)

    # Header
    header = f"  {'Strategy':<18s} {'Chunks':>6s} {'Rewrite':>9s} {'Retrieve':>9s} {'Answer':>9s} {'Total':>9s}"
    print(header)
    print(f"  {'─'*18} {'─'*6} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

    for r in results:
        strategy = r.get("strategy", "?")
        n_chunks = len(r.get("contexts", []))
        t = r.get("timing_ms", {})
        rewrite = t.get("rewrite_ms", 0)
        retrieval = t.get("retrieval_ms", 0)
        answer = t.get("answer_ms", 0)
        total = t.get("total_ms", 0)

        print(
            f"  {strategy:<18s} {n_chunks:>6d} {rewrite:>8.0f}ms {retrieval:>8.0f}ms "
            f"{answer:>8.0f}ms {total:>8.0f}ms"
        )

    print(f"\n{THICK_SEPARATOR}\n")
