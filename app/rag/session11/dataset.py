"""
session11/dataset.py
---------------------
Evaluation dataset loader for Session 11.

Loads curated question/ground-truth pairs from a JSON file and provides
them in a format ready for evaluation pipelines.

Each evaluation item contains:
    - question:         the user's query
    - ground_truth:     the expected correct answer (for faithfulness/relevance)
    - reference_answer: optional longer reference (for answer quality checks)
    - expected_sources: optional list of expected source filenames

The default dataset lives at data/session11/eval_questions.json.
You can also pass questions as a Python list directly.

Usage:
    from app.rag.session11.dataset import load_eval_dataset, get_questions

    dataset = load_eval_dataset()
    questions = get_questions()  # just the question strings
"""

import json
import os
from typing import List, Dict, Optional


# Default path relative to project root
DEFAULT_DATASET_PATH = os.path.join("data", "session11", "eval_questions.json")


def load_eval_dataset(path: Optional[str] = None) -> List[Dict]:
    """
    Loads the evaluation dataset from a JSON file.

    Each item in the returned list has:
        question:         str  — the evaluation question
        ground_truth:     str  — the expected correct answer
        reference_answer: str  — optional longer reference answer
        expected_sources: list — optional list of expected source filenames

    Args:
        path: path to the JSON file. Defaults to data/session11/eval_questions.json.

    Returns:
        List of evaluation item dicts.

    Raises:
        FileNotFoundError: if the dataset file doesn't exist.
        json.JSONDecodeError: if the file contains invalid JSON.
    """
    dataset_path = path or DEFAULT_DATASET_PATH

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Evaluation dataset not found at: {dataset_path}\n"
            f"  Expected location: {os.path.abspath(dataset_path)}\n"
            f"  Hint: Make sure you're running from the project root directory."
        )

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Validate structure
    for i, item in enumerate(dataset):
        if "question" not in item:
            raise ValueError(f"Dataset item {i} is missing required 'question' field.")
        if "ground_truth" not in item:
            raise ValueError(f"Dataset item {i} is missing required 'ground_truth' field.")

    print(f"[Dataset] Loaded {len(dataset)} evaluation questions from {dataset_path}")
    return dataset


def get_questions(path: Optional[str] = None) -> List[str]:
    """
    Returns just the question strings from the evaluation dataset.

    Convenience wrapper for cases where you only need the questions
    (e.g., running strategies without evaluation).

    Args:
        path: optional path to the dataset file.

    Returns:
        List of question strings.
    """
    dataset = load_eval_dataset(path)
    return [item["question"] for item in dataset]


def get_ground_truths(path: Optional[str] = None) -> Dict[str, str]:
    """
    Returns a mapping of question → ground_truth.

    Useful for looking up the expected answer for a given question.

    Args:
        path: optional path to the dataset file.

    Returns:
        Dict mapping question strings to ground truth strings.
    """
    dataset = load_eval_dataset(path)
    return {item["question"]: item["ground_truth"] for item in dataset}


def create_sample_dataset() -> List[Dict]:
    """
    Returns the built-in sample evaluation dataset as a Python list.

    Use this if you want to work with the dataset in-memory without
    loading from a file. This returns the same data that's in
    data/session11/eval_questions.json.

    Returns:
        List of evaluation item dicts.
    """
    return [
        {
            "question": "What are the interview rounds at Amazon for an SDE role?",
            "ground_truth": (
                "Amazon's SDE interview process includes an online assessment, "
                "phone screen, and onsite loop with 4-5 rounds covering coding, "
                "system design, and behavioral questions based on Leadership Principles."
            ),
            "expected_sources": ["amazon.md"],
        },
        {
            "question": "How does Google evaluate problem-solving skills during interviews?",
            "ground_truth": (
                "Google evaluates problem-solving through coding interviews that "
                "test algorithmic thinking, data structure knowledge, and the ability "
                "to optimize solutions."
            ),
            "expected_sources": ["google.md"],
        },
        {
            "question": "What is Oracle OCI and how is it relevant to Oracle interviews?",
            "ground_truth": (
                "Oracle Cloud Infrastructure (OCI) is Oracle's cloud computing platform. "
                "OCI knowledge is increasingly relevant in Oracle interviews, especially "
                "for cloud engineering and SDE roles."
            ),
            "expected_sources": ["oracle.md"],
        },
        {
            "question": "How should I prepare for a coding assessment?",
            "ground_truth": (
                "Practice data structures and algorithms, focus on time complexity, "
                "use platforms like LeetCode, and practice writing clean code under "
                "time constraints."
            ),
            "expected_sources": ["amazon.md", "google.md", "microsoft.md"],
        },
        {
            "question": "What are Amazon's Leadership Principles and how are they used in interviews?",
            "ground_truth": (
                "Amazon's Leadership Principles are guidelines that define Amazon's culture. "
                "Behavioral interview questions are structured around LPs like Customer "
                "Obsession, Ownership, and Dive Deep."
            ),
            "expected_sources": ["amazon.md"],
        },
    ]


def print_dataset_summary(dataset: Optional[List[Dict]] = None) -> None:
    """
    Prints a readable summary of the evaluation dataset.

    Args:
        dataset: optional list of eval items. If None, loads from file.
    """
    if dataset is None:
        dataset = load_eval_dataset()

    print(f"\n{'='*70}")
    print(f"  EVALUATION DATASET — {len(dataset)} questions")
    print(f"{'='*70}")

    for i, item in enumerate(dataset, start=1):
        q = item["question"]
        gt = item["ground_truth"][:80] + "..." if len(item["ground_truth"]) > 80 else item["ground_truth"]
        sources = ", ".join(item.get("expected_sources", ["?"]))

        print(f"\n  [{i}] {q}")
        print(f"      Ground truth: {gt}")
        print(f"      Sources: {sources}")

    print(f"\n{'='*70}\n")
