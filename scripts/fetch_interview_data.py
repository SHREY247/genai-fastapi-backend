"""
scripts/fetch_interview_data.py
-------------------------------
Prepares an interview prep dataset from repo-inspired sources + curated
company summaries.

How it works:
  1. Fetches the list of interview experience links from the public repo
     (realabbas/big-companies-interview-questions) for each company.
  2. Parses the link TITLES to identify common interview themes.
  3. Combines those themes with curated, structured interview prep content
     (hand-written summaries of publicly known interview patterns).
  4. Saves the result as clean markdown files ready for RAG ingestion.
  5. Generates a private PDF with fictional 2025 hiring updates.

Why not raw extraction?
  The GitHub repo contains only link titles pointing to GeeksforGeeks —
  no actual question content. Scraping GfG would be fragile and legally
  questionable. Instead, this script demonstrates a realistic preprocessing
  pattern: take a real source, extract what's usable (link titles), and
  enrich it with curated knowledge.

  This is a deliberate teaching moment: real-world ingestion almost always
  requires transformation, not just loading.

Usage:
    python -m scripts.fetch_interview_data

Output:
    data/interview_prep/amazon.md
    data/interview_prep/google.md
    data/interview_prep/microsoft.md
    data/interview_prep/adobe.md
    data/interview_prep/oracle.md
    data/interview_prep/company_hiring_updates_2025.pdf
"""


import os
import re
import urllib.request
from fpdf import FPDF

# ── Configuration ─────────────────────────────────────────────────────────

REPO_BASE = (
    "https://raw.githubusercontent.com/realabbas/"
    "big-companies-interview-questions/master/companies"
)

COMPANIES = ["amazon", "google", "microsoft", "adobe", "oracle"]

OUTPUT_DIR = os.path.join("data", "interview_prep")

# ── Company-specific interview prep content ───────────────────────────────
# The GitHub repo only contains link titles. We enrich each company with
# realistic interview preparation content based on publicly known patterns.

COMPANY_CONTENT = {
    "amazon": {
        "title": "Amazon SDE Interview Preparation Guide",
        "overview": (
            "Amazon's Software Development Engineer (SDE) interview process is one of "
            "the most structured in the industry. The process typically consists of an "
            "online assessment (OA) followed by 4-5 on-site interview rounds."
        ),
        "process": [
            "Online Assessment: 2 coding problems (70 minutes) + Work Simulation",
            "Phone Screen: 1 coding problem + Leadership Principles behavioral question",
            "On-site Loop: 4 rounds (2 coding + 1 system design + 1 Bar Raiser)",
            "Bar Raiser Round: Cross-team interviewer who ensures hiring bar is maintained",
        ],
        "topics": [
            "Arrays and Strings — sliding window, two pointers, prefix sums",
            "Trees and Graphs — BFS, DFS, lowest common ancestor, graph traversal",
            "Dynamic Programming — knapsack variations, longest subsequence problems",
            "System Design — design a URL shortener, design an e-commerce cart system",
            "Object-Oriented Design — parking lot, library management system",
            "Leadership Principles — STAR format stories for all 16 principles",
        ],
        "tips": [
            "Amazon asks about Leadership Principles in EVERY round, not just behavioral",
            "The Bar Raiser has veto power — they evaluate culture fit and thinking depth",
            "For coding, practice explaining your thought process out loud",
            "System design questions often relate to Amazon-scale distributed systems",
            "Prepare 2-3 STAR stories for each Leadership Principle",
        ],
        "sample_questions": [
            "Design a system like Amazon's recommendation engine",
            "Given a list of reviews, find the most helpful review using a scoring algorithm",
            "Implement an LRU cache with O(1) get and put operations",
            "Tell me about a time you had to make a decision with incomplete data (LP: Bias for Action)",
            "How would you design a distributed rate limiter for API throttling?",
            "Find the k most frequent elements in an array",
            "Tell me about a time you disagreed with your manager (LP: Have Backbone)",
        ],
    },
    "google": {
        "title": "Google SDE Interview Preparation Guide",
        "overview": (
            "Google's interview process is known for being rigorous and algorithm-heavy. "
            "The process typically involves a recruiter screen, 1-2 phone interviews, and "
            "an on-site loop of 5 coding/design interviews."
        ),
        "process": [
            "Recruiter Screen: resume review and initial fit assessment",
            "Phone Interview 1: 45-min coding on Google Docs (no IDE)",
            "Phone Interview 2: another 45-min coding round (for some candidates)",
            "On-site Loop: 5 rounds — 4 coding + 1 Googleyness & Leadership (G&L)",
            "Hiring Committee Review: packet review by committee (candidate is not present)",
        ],
        "topics": [
            "Graph Algorithms — shortest path, topological sort, connected components",
            "Dynamic Programming — complex state transitions, memoization patterns",
            "String Manipulation — regex matching, edit distance, palindrome problems",
            "Concurrency — thread safety, producer-consumer, deadlock prevention",
            "System Design (L5+) — design Google Maps, design YouTube recommendation",
            "Googleyness — navigating ambiguity, collaboration, intellectual humility",
        ],
        "tips": [
            "Google interviews are on Google Docs, not a real IDE — practice without autocomplete",
            "They value clean, bug-free code over brute-force solutions",
            "Algorithmic complexity analysis is critical — always state time/space complexity",
            "The Hiring Committee values consistency — strong performance across all rounds",
            "For system design, start with requirements clarification and capacity estimation",
        ],
        "sample_questions": [
            "Design the backend for Google Calendar",
            "Implement a trie with insert, search, and startsWith operations",
            "Given a 2D grid of characters, find all words from a dictionary (word search II)",
            "How would you design a distributed file storage system like Google Drive?",
            "Find the median of two sorted arrays in O(log(m+n)) time",
            "Design a type-ahead search suggestion system",
            "How do you handle disagreements in a team with no clear hierarchy?",
        ],
    },
    "microsoft": {
        "title": "Microsoft SDE Interview Preparation Guide",
        "overview": (
            "Microsoft's interview process balances coding skills, system design, and "
            "behavioral assessment. The process typically includes an initial phone screen "
            "followed by 4-5 on-site rounds, with the final round conducted by a hiring manager."
        ),
        "process": [
            "Phone Screen: 30-45 min coding problem on a shared editor",
            "On-site Round 1: Coding — data structures and algorithms",
            "On-site Round 2: Coding — more complex problem, often with follow-ups",
            "On-site Round 3: System Design (for experienced) or Coding (for new grads)",
            "On-site Round 4 ('As Appropriate'): Hiring manager round — fit and career goals",
        ],
        "topics": [
            "Linked Lists — reversal, cycle detection, merge sorted lists",
            "Binary Trees — serialization, path problems, views (left/right/top)",
            "Sorting and Searching — custom comparators, binary search variations",
            "System Design — design OneDrive, design Teams chat system",
            "API Design — RESTful service design, versioning, error handling",
            "Behavioral — growth mindset, collaboration, customer obsession",
        ],
        "tips": [
            "Microsoft values the 'growth mindset' — show curiosity and willingness to learn",
            "The 'As Appropriate' (AA) round with the hiring manager is a strong hire/no-hire signal",
            "Whiteboard coding is still common — practice writing code on a board or paper",
            "For system design, they appreciate when you consider Azure cloud services",
            "Past project deep-dives are common — be ready to discuss a project in detail",
        ],
        "sample_questions": [
            "Design a simplified version of Microsoft Teams messaging system",
            "Implement a function to detect a cycle in a linked list and return the starting node",
            "Given a binary tree, return the zigzag level-order traversal",
            "How would you design a real-time collaborative document editor like Word Online?",
            "Merge k sorted linked lists into one sorted list",
            "Design an efficient notification system for a large-scale application",
            "Tell me about a project where you had to learn a new technology quickly",
        ],
    },
    "adobe": {
        "title": "Adobe SDE Interview Preparation Guide",
        "overview": (
            "Adobe's interview process emphasizes both strong CS fundamentals and creative "
            "problem solving. For engineering roles, expect a mix of coding, design, and "
            "behavioral rounds. Adobe values candidates who can bridge engineering and design."
        ),
        "process": [
            "Online Test: MCQs on CS fundamentals + 2 coding problems (90 minutes)",
            "Technical Phone Screen: 1-2 coding problems with explanation",
            "On-site Round 1: Data Structures and Algorithms",
            "On-site Round 2: Problem Solving and Code Quality",
            "On-site Round 3: System Design or Domain-Specific (e.g., PDF rendering, media)",
            "On-site Round 4: Hiring Manager — behavioral + culture fit",
        ],
        "topics": [
            "Matrix Problems — rotation, spiral traversal, search in 2D matrix",
            "Recursion and Backtracking — N-Queens, Sudoku solver, permutations",
            "Bit Manipulation — power of two, counting bits, XOR tricks",
            "Design Patterns — Observer, Factory, Strategy (Adobe loves these)",
            "System Design — design a PDF rendering pipeline, design Creative Cloud sync",
            "Behavioral — creativity, dealing with ambiguity, cross-functional collaboration",
        ],
        "tips": [
            "Adobe values code quality — clean variable names, proper error handling",
            "Design patterns come up frequently — know at least Observer, Factory, Strategy",
            "If applying to the Document Cloud team, understand PDF structure basics",
            "Creative problem solving is rewarded — don't just give textbook solutions",
            "The hiring manager round carries significant weight at Adobe",
        ],
        "sample_questions": [
            "Rotate a matrix 90 degrees clockwise in-place",
            "Design a file synchronization system for Adobe Creative Cloud",
            "Implement a basic regex matcher supporting '.' and '*'",
            "Solve the N-Queens problem and return all valid configurations",
            "How would you design an image processing pipeline that handles multiple formats?",
            "Given a string, find the longest palindromic substring",
            "Tell me about a time you improved a process or product significantly",
        ],
    },
    "oracle": {
        "title": "Oracle SDE Interview Preparation Guide",
        "overview": (
            "Oracle's interview process is thorough and tends to focus on strong fundamentals "
            "in data structures, algorithms, and database concepts. For backend roles, "
            "expect questions on Java, SQL, and distributed systems."
        ),
        "process": [
            "Online Assessment: 2-3 coding problems + MCQs on DBMS, OS, networking",
            "Technical Phone Screen: coding + conceptual questions",
            "On-site Round 1: Data Structures and Algorithms (often Java-heavy)",
            "On-site Round 2: Database concepts — SQL queries, normalization, indexing",
            "On-site Round 3: System Design or OS/Networking concepts",
            "On-site Round 4: Hiring Manager — behavioral + team fit",
        ],
        "topics": [
            "Java Fundamentals — collections framework, multithreading, JVM internals",
            "SQL and DBMS — complex joins, window functions, query optimization",
            "Data Structures — heap, hash map internals, balanced BSTs",
            "Operating Systems — process scheduling, memory management, deadlocks",
            "System Design — design a database connection pool, design a job scheduler",
            "Networking — TCP/UDP, HTTP lifecycle, load balancing concepts",
        ],
        "tips": [
            "Oracle interviews lean heavily toward Java — know generics, streams, and concurrency",
            "Database questions are more in-depth than at other companies — practice SQL",
            "OS and networking concepts appear more frequently than at FAANG companies",
            "For system design, demonstrate understanding of ACID properties and CAP theorem",
            "Be prepared to write SQL queries on the spot during interviews",
        ],
        "sample_questions": [
            "Write a SQL query to find the second highest salary in each department",
            "Implement a thread-safe singleton pattern in Java",
            "Design a database connection pooling system",
            "Explain the differences between HashMap and ConcurrentHashMap in Java",
            "Given a stream of integers, find the running median efficiently",
            "How would you design a distributed job scheduling system?",
            "What is the difference between optimistic and pessimistic locking?",
        ],
    },
}

# ── Private PDF content ───────────────────────────────────────────────────

PRIVATE_PDF_CONTENT = """COMPANY HIRING UPDATES - 2025 (INTERNAL / CONFIDENTIAL)
Prepared by: Interview Intelligence Team
Last Updated: January 2025

This document contains updated hiring signals and interview format changes
for major tech companies. This information supplements and in some cases
OVERRIDES publicly available interview preparation advice.

==========================================================================
AMAZON - 2025 INTERVIEW UPDATES
==========================================================================

Key Changes:
- Leadership Principles now account for approximately 40% of the overall
  evaluation score, up from the previous informal weighting of ~25%.
- The "Bar Raiser 2.0" format has been introduced: candidates now do a
  live architecture whiteboarding session WITH the Bar Raiser present,
  combining system design evaluation with LP assessment.
- Online Assessment has added a new "Work Style Assessment" section that
  uses situational judgment scenarios.
- Amazon has REDUCED the emphasis on pure LeetCode-style hard problems.
  The focus has shifted toward medium-difficulty problems that test
  practical coding ability and clean code practices.
- New "Day 1 Thinking" behavioral dimension added to LP evaluation.

Insider Signal:
  Teams in AWS and Alexa are now prioritizing candidates who demonstrate
  experience with large-scale distributed systems over pure algorithmic
  problem-solving skills. The coding bar remains high, but the
  differentiator is now system thinking and ownership mindset.

==========================================================================
GOOGLE - 2025 INTERVIEW UPDATES
==========================================================================

Key Changes:
- The on-site interview loop has been REDUCED from 5 rounds to 3 rounds
  for most SDE positions (L3-L5). This is a major change from the
  traditional 5-round format.
- A new "ML Systems Design" round has been added for all L4+ candidates,
  regardless of whether the role is ML-specific. Candidates are expected
  to understand basic ML pipeline design.
- "Googleyness and Leadership" has been renamed to "Collaboration and
  Leadership" and now includes AI ethics scenario questions.
- Coding interviews now allow candidates to choose their programming
  language AND use a basic IDE (no longer restricted to Google Docs).
- The Hiring Committee process has been streamlined - decisions are now
  made within 2 weeks instead of the previous 4-6 week timeline.

Insider Signal:
  Google is actively looking for candidates who can articulate the
  societal impact of the technology they build. Expect at least one
  question about responsible AI development in the behavioral round.

==========================================================================
MICROSOFT - 2025 INTERVIEW UPDATES
==========================================================================

Key Changes:
- The new "Explore" interview track has been launched for candidates with
  non-traditional backgrounds (bootcamp graduates, career changers). This
  track replaces the standard whiteboard round with a take-home project.
- Coding rounds now use VS Code Online (vscode.dev) instead of physical
  whiteboards. Candidates can use their preferred language with full IDE
  features including autocomplete and debugging.
- A new 30-minute "Growth Mindset" behavioral round has been added. This
  round explicitly evaluates learning agility, resilience, and the ability
  to receive and act on feedback.
- For senior roles (L63+), a "technical leadership" round replaces one of
  the coding rounds, focusing on mentorship, technical strategy, and
  cross-team influence.

Insider Signal:
  Microsoft is placing significant emphasis on AI/Copilot integration
  skills. Candidates who can demonstrate experience building with or on
  top of LLM-based tools have a distinct advantage, especially for teams
  working on Microsoft 365 Copilot and Azure AI services.

==========================================================================
ADOBE - 2025 INTERVIEW UPDATES
==========================================================================

Key Changes:
- ALL SDE roles now include a mandatory Generative AI assessment round.
  Candidates are given a real-world scenario (e.g., "How would you add an
  AI-powered feature to Photoshop?") and must design a solution end-to-end.
- Design rounds now test the "Figma-to-Code" workflow - candidates are
  shown a Figma mockup and asked to implement a functional prototype.
- The online coding test has been replaced with a 2-hour take-home challenge
  that emphasizes code quality, testing, and documentation over speed.
- Adobe has added a "Portfolio Review" round for candidates applying to
  Creative Cloud teams, where candidates walk through a past project.
- Cross-functional collaboration questions now make up 30% of the behavioral
  round, up from approximately 15% previously.

Insider Signal:
  Adobe is aggressively hiring for its Firefly AI team. Candidates with
  experience in diffusion models, prompt engineering, or multimodal AI
  are fast-tracked through the process. Even for non-AI roles, showing
  awareness of generative AI applications in creative tools is beneficial.

==========================================================================
ORACLE - 2025 INTERVIEW UPDATES
==========================================================================

Key Changes:
- The Oracle Cloud Infrastructure (OCI) team is on a major hiring surge
  and has a separate, accelerated interview track (3 rounds instead of 4).
- Interviews now include a HANDS-ON cloud deployment exercise where
  candidates deploy a simple service to OCI and explain their choices.
- The emphasis on Java-specific questions has been REDUCED. While Java
  knowledge is still valued, Oracle now accepts solutions in Python, Go,
  or Rust for coding rounds.
- A new "Cloud Architecture" round has replaced the traditional OS/Networking
  round for cloud-focused positions.
- Database questions remain important but now include NoSQL concepts
  (document stores, key-value stores) alongside traditional RDBMS topics.

Insider Signal:
  Oracle is competing aggressively with AWS and Azure for cloud market
  share. Candidates who can articulate why a customer would choose OCI
  over competitors, or who have hands-on experience with any major cloud
  platform, have a significant advantage. The company is also investing
  heavily in AI infrastructure, so ML systems knowledge is becoming
  increasingly relevant even for non-ML roles.
"""



# ── Helpers ─────────────────────────────────────────────────────────────

def fetch_repo_links(company: str) -> list[str]:
    """
    Fetches the raw markdown file for a company from the GitHub repo
    and extracts the link titles (interview experience titles).

    Returns a list of link title strings.
    """
    url = f"{REPO_BASE}/{company}/{company}.md"
    print(f"  Fetching: {url}")

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            content = response.read().decode("utf-8")
    except Exception as e:
        print(f"  WARNING: Could not fetch {url}: {e}")
        print(f"  Using built-in content for {company}")
        return []

    # Extract link titles from markdown links: [Title](url)
    titles = re.findall(r"\[([^\]]+)\]", content)

    # Filter out very short or non-interview titles
    titles = [t for t in titles if len(t) > 10 and "interview" in t.lower()]
    return titles[:15]  # Keep top 15 for manageable size


def generate_company_markdown(company: str) -> str:
    """
    Generates a structured markdown interview prep file for a company.
    Combines fetched link titles with curated preparation content.
    """
    info = COMPANY_CONTENT[company]
    link_titles = fetch_repo_links(company)

    lines = [
        f"# {info['title']}",
        "",
        f"## Company Overview",
        "",
        info["overview"],
        "",
        "## Interview Process",
        "",
    ]

    for i, step in enumerate(info["process"], 1):
        lines.append(f"{i}. {step}")
    lines.append("")

    lines.append("## Key Topics to Prepare")
    lines.append("")
    for topic in info["topics"]:
        lines.append(f"- {topic}")
    lines.append("")

    lines.append("## Preparation Tips")
    lines.append("")
    for tip in info["tips"]:
        lines.append(f"- {tip}")
    lines.append("")

    lines.append("## Sample Interview Questions")
    lines.append("")
    for q in info["sample_questions"]:
        lines.append(f"- {q}")
    lines.append("")

    # Add fetched link titles as additional resources
    if link_titles:
        lines.append("## Real Interview Experiences (from public sources)")
        lines.append("")
        lines.append(
            "The following interview experiences have been reported by candidates:"
        )
        lines.append("")
        for title in link_titles:
            lines.append(f"- {title}")
        lines.append("")

    return "\n".join(lines)


def generate_private_pdf(output_path: str) -> None:
    """
    Generates the private PDF containing 2025 hiring updates.
    Uses fpdf2 for lightweight PDF creation.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    def safe_text(text: str) -> str:
        """Replace any non-Latin1 characters with ASCII equivalents."""
        replacements = {
            "\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00a0": " ",
        }
        for char, repl in replacements.items():
            text = text.replace(char, repl)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def write_line(text: str, style: str = "", size: int = 10, height: int = 6):
        """Write a single line of text to the PDF."""
        pdf.set_font("Helvetica", style, size)
        pdf.set_x(15)  # Always reset x to left margin
        pdf.multi_cell(w=180, h=height, text=safe_text(text))

    # Process the content line by line
    lines = PRIVATE_PDF_CONTENT.strip().split("\n")
    pdf.add_page()

    for line in lines:
        stripped = line.strip()

        # Skip separator lines
        if stripped and all(c == "=" for c in stripped):
            continue

        # Empty line
        if not stripped:
            pdf.ln(3)
            continue

        # Title line
        if "COMPANY HIRING UPDATES" in stripped:
            write_line(stripped, style="B", size=14, height=10)
            continue

        # Company section headers — start a new page
        if stripped.endswith("INTERVIEW UPDATES"):
            pdf.add_page()
            write_line(stripped, style="B", size=13, height=10)
            continue

        # Sub-headers
        if stripped.startswith(("Key Changes:", "Insider Signal:")):
            pdf.ln(2)
            write_line(stripped, style="B", size=11, height=8)
            continue

        # Regular text / bullet points
        write_line(stripped)

    pdf.output(output_path)




# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("INTERVIEW DATASET PREPARATION")
    print("  (repo-inspired sources + curated enrichment)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Generate company markdown files
    print("\n[Step 1] Building company interview prep files...")
    print("  Fetching link titles from GitHub repo, then enriching")
    print("  with curated interview prep content.\n")
    for company in COMPANIES:
        print(f"Processing: {company}")
        content = generate_company_markdown(company)
        filepath = os.path.join(OUTPUT_DIR, f"{company}.md")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Saved: {filepath} ({len(content)} chars)\n")


    # Step 2: Generate the private PDF
    print("[Step 2] Generating private PDF (company_hiring_updates_2025.pdf)...\n")
    pdf_path = os.path.join(OUTPUT_DIR, "company_hiring_updates_2025.pdf")
    generate_private_pdf(pdf_path)
    print(f"  Saved: {pdf_path}")

    # Summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created in '{OUTPUT_DIR}':")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        filepath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(filepath)
        print(f"  - {f} ({size:,} bytes)")
    print()


if __name__ == "__main__":
    main()
