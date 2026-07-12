"""
demo.py
-------
Interactive CLI demo for the Arabic Legal RAG system.
Runs questions through the best-performing configuration: Hybrid + Article chunks.

Usage:
    cd C:/Work/nlp
    python demo.py

    # Or pass a question directly:
    python demo.py --question "ما هي مدة إشعار إنهاء عقد العمل؟"
"""

import sys
import time
import argparse
import textwrap
from pathlib import Path

# ── ANSI colours ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
MAGENTA= "\033[95m"
RED    = "\033[91m"
WHITE  = "\033[97m"
GRAY   = "\033[90m"

def c(text, color, bold=False):
    return f"{BOLD if bold else ''}{color}{text}{RESET}"

def rule(char="─", width=70, color=CYAN):
    return c(char * width, color)

def header_rule(color=CYAN):
    return c("═" * 70, color)

# ── Preset demo questions ─────────────────────────────────────────────────────
SYSTEMS = [
    {
        "key":   "hybrid_article",
        "label": "Hybrid + Article",
        "desc":  "Best answer quality  (EM=0.3794, BERTScore=0.7131)",
        "color": CYAN,
    },
    {
        "key":   "bm25_article",
        "label": "BM25  + Article",
        "desc":  "Best retrieval       (Recall@5=0.9375, MRR=0.8456)",
        "color": MAGENTA,
    },
    {
        "key":   "both",
        "label": "Both side by side",
        "desc":  "Run both systems and compare answers (2× slower)",
        "color": YELLOW,
    },
]

DEMO_QUESTIONS = [
    {
        "label": "Annual Leave",
        "question": "كم عدد أيام الإجازة السنوية التي يستحقها العامل؟",
        "translation": "How many days of annual leave is a worker entitled to?",
    },
    {
        "label": "Notice Period",
        "question": "ما هي مدة الإشعار المطلوبة عند إنهاء عقد العمل؟",
        "translation": "What is the required notice period when terminating an employment contract?",
    },
    {
        "label": "Minimum Wage",
        "question": "هل يحدد قانون العمل الأردني حداً أدنى للأجور؟",
        "translation": "Does Jordanian labor law specify a minimum wage?",
    },
    {
        "label": "Maternity Leave",
        "question": "ما هي مدة إجازة الأمومة للمرأة العاملة؟",
        "translation": "What is the duration of maternity leave for working women?",
    },
    {
        "label": "Overtime Pay",
        "question": "كيف تحسب أجر العمل الإضافي في قانون العمل الأردني؟",
        "translation": "How is overtime pay calculated under Jordanian labor law?",
    },
    {
        "label": "Termination Compensation",
        "question": "ما هي مكافأة نهاية الخدمة التي يستحقها العامل عند انتهاء عقده؟",
        "translation": "What end-of-service compensation is a worker entitled to when their contract ends?",
    },
    {
        "label": "Working Hours",
        "question": "ما هو الحد الأقصى لساعات العمل اليومية والأسبوعية؟",
        "translation": "What is the maximum number of daily and weekly working hours?",
    },
    {
        "label": "Custom question",
        "question": None,
        "translation": "Type your own Arabic question",
    },
]


def print_banner():
    print()
    print(header_rule(CYAN))
    print(c("  ⚖  Arabic Legal RAG — Jordanian Labor Law Demo", CYAN, bold=True))
    print(c("     Hybrid Retrieval (BM25 + Dense) · Article-level Chunks", GRAY))
    print(header_rule(CYAN))
    print()


def print_system_menu():
    print(c("  Select a system:", WHITE, bold=True))
    print()
    for i, s in enumerate(SYSTEMS, 1):
        label = c(f"  [{i}]", s["color"], bold=True)
        name  = c(s["label"], WHITE, bold=True)
        desc  = c(f"       {s['desc']}", GRAY)
        print(f"{label}  {name}")
        print(desc)
        print()


def get_system_choice():
    while True:
        try:
            choice = input(c("  Enter number (1–3): ", CYAN, bold=True)).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(SYSTEMS):
                return SYSTEMS[idx]["key"]
            print(c("  Please enter a number between 1 and 3.", RED))
        except (ValueError, KeyboardInterrupt):
            print(c("  Please enter a number between 1 and 3.", RED))


def print_question_menu():
    print(c("  Select a demo question:", WHITE, bold=True))
    print()
    for i, q in enumerate(DEMO_QUESTIONS, 1):
        label = c(f"  [{i}]", YELLOW, bold=True)
        name  = c(q["label"], WHITE, bold=True)
        trans = c(f"  {q['translation']}", GRAY)
        print(f"{label}  {name}")
        print(trans)
        print()


def get_question_choice():
    while True:
        try:
            choice = input(c("  Enter number (1–8): ", CYAN, bold=True)).strip()
            idx = int(choice) - 1
            if 0 <= idx < len(DEMO_QUESTIONS):
                return idx
            print(c("  Please enter a number between 1 and 8.", RED))
        except (ValueError, KeyboardInterrupt):
            print(c("  Please enter a number between 1 and 8.", RED))


def wrap_arabic(text, width=64):
    """Wrap long text for terminal display."""
    return textwrap.fill(text, width=width, break_long_words=False)


# Cache built systems so we don't rebuild on every question
_built_systems = {}

def get_system(key):
    global _built_systems
    if not _built_systems:
        sys.path.insert(0, str(Path(__file__).parent / "systems"))
        from pipeline import build_all_systems
        all_sys = build_all_systems(chunk_strategies=["article"], top_k=5)
        _built_systems = {s.name: s for s in all_sys}
    return _built_systems[key]


def print_result(result, accent_color):
    """Print retrieved passages and answer for one system result."""
    if result.retrieved:
        for p in result.retrieved:
            rank  = p.get("rank", "?")
            cid   = p.get("chunk_id", "")
            score = p.get("score", 0)
            text  = p.get("text", "")[:300]

            print(f"  {c(f'Rank {rank}', accent_color, bold=True)}  "
                  f"{c(cid, BLUE)}  "
                  f"{c(f'score: {score:.4f}', GRAY)}")
            for line in wrap_arabic(text).split("\n"):
                print(f"    {c(line, WHITE)}")
            print()
    else:
        print(c("  (no passages retrieved)", GRAY))

    print(rule())
    print()
    print(c("  ANSWER", GREEN, bold=True))
    print()
    for line in wrap_arabic(result.answer, width=66).split("\n"):
        print(f"  {c(line, WHITE, bold=True)}")
    print()
    print(c(f"  ⏱  Latency: {result.latency_s:.1f}s", GRAY))
    print()


def run_demo(question: str, system_key: str = "hybrid_article"):
    # ── Initialise ────────────────────────────────────────────────────────────
    print()
    print(rule())
    print(c("  Initialising …", GRAY))
    get_system("hybrid_article")   # warms up both (build_all_systems builds all)
    print(c("  ✓ Systems ready", GREEN, bold=True))
    print(rule())

    # ── Show the question ─────────────────────────────────────────────────────
    print()
    print(c("  QUESTION", YELLOW, bold=True))
    print()
    print(f"  {c(question, WHITE, bold=True)}")
    print()

    # ── Run selected system(s) ────────────────────────────────────────────────
    if system_key == "both":
        for sys_info in SYSTEMS[:2]:   # hybrid then BM25
            key   = sys_info["key"]
            label = sys_info["label"]
            color = sys_info["color"]

            print(rule(char="─", color=color))
            print(c(f"  {label}", color, bold=True))
            print(rule(char="·", color=GRAY))
            print(c("  Retrieving …", GRAY))

            result = get_system(key).run(question)

            print()
            print(c("  RETRIEVED PASSAGES", YELLOW, bold=True))
            print()
            print_result(result, color)
            print(header_rule(color))
            print()

    else:
        sys_info = next(s for s in SYSTEMS if s["key"] == system_key)
        color    = sys_info["color"]

        print(rule(char="·", color=GRAY))
        print(c(f"  Running {sys_info['label']} …", color))

        result = get_system(system_key).run(question)

        print()
        print(c("  RETRIEVED PASSAGES", YELLOW, bold=True))
        print()
        print_result(result, color)
        print(header_rule(color))
        print()


def pick_question():
    """Show question menu and return the chosen question string."""
    print_question_menu()
    idx   = get_question_choice()
    entry = DEMO_QUESTIONS[idx]
    if entry["question"] is None:
        print()
        question = input(c("  Type your Arabic question: ", CYAN, bold=True)).strip()
        if not question:
            return None
    else:
        question = entry["question"]
        print()
        print(c(f"  Selected: {entry['label']}", GREEN, bold=True))
    return question


def main():
    parser = argparse.ArgumentParser(description="Arabic Legal RAG demo")
    parser.add_argument("--question", default=None,
                        help="Arabic question to ask (skips the question menu)")
    parser.add_argument("--system", default=None,
                        choices=["hybrid_article", "bm25_article", "both"],
                        help="System to run (skips the system menu)")
    args = parser.parse_args()

    print_banner()

    # ── Choose system ─────────────────────────────────────────────────────────
    if args.system:
        system_key = args.system
    else:
        print_system_menu()
        system_key = get_system_choice()
        print()

    # ── Choose question ───────────────────────────────────────────────────────
    if args.question:
        question = args.question
    else:
        question = pick_question()
        if not question:
            print(c("  No question entered. Exiting.", RED))
            return

    run_demo(question, system_key)

    # ── Ask to continue ───────────────────────────────────────────────────────
    while True:
        print(c("  Run another question? (y/n): ", CYAN, bold=True), end="")
        again = input().strip().lower()
        if again != "y":
            break

        print()
        print_system_menu()
        system_key = get_system_choice()
        print()

        question = pick_question()
        if not question:
            break

        run_demo(question, system_key)

    print(c("  Thank you for watching the demo.", CYAN, bold=True))
    print()


if __name__ == "__main__":
    main()
