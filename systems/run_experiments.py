"""
run_experiments.py
------------------
Runs all experimental systems over the evaluation QA set and saves results.

Usage:
    cd C:/Work/nlp/systems
    python run_experiments.py

Options:
    --systems   baseline bm25_article bm25_fixed dense_article dense_fixed hybrid_article hybrid_fixed
                (space-separated subset; default: all)
    --limit     N   only run the first N questions (useful for a quick sanity check)
    --top-k     K   passages to retrieve (default: from config.py)

Output:
    results/predictions.jsonl   — one record per (system, question)
    results/retrieved.jsonl     — retrieved passages per (system, question)
"""

import sys
import json
import argparse
import csv
import ctypes
import platform
import subprocess
from pathlib import Path
from datetime import datetime

def _get_brightness() -> int | None:
    """Read the current monitor brightness (0-100) via WMI. Returns None on failure."""
    try:
        result = subprocess.run(
            ["powershell", "-Command",
             "(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness).CurrentBrightness"],
            capture_output=True, text=True, timeout=10
        )
        return int(result.stdout.strip())
    except Exception:
        return None

def _set_brightness(level: int) -> None:
    """Set monitor brightness (0-100) via WMI."""
    try:
        subprocess.run(
            ["powershell", "-Command",
             f"(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods)"
             f".WmiSetBrightness(1,{level})"],
            capture_output=True, timeout=10
        )
    except Exception:
        pass   # silently skip if brightness control isn't supported

_original_brightness: int | None = None
_DIM_BRIGHTNESS = 10   # percent — change this if you want it brighter/dimmer

def prevent_sleep():
    """
    Keep the system awake but let the screensaver and display dimming run normally.

    - ES_SYSTEM_REQUIRED  → CPU stays on, experiments keep running
    - ES_DISPLAY_REQUIRED is intentionally NOT set → screensaver activates
      on its normal schedule, display can dim
    - Brightness is lowered to _DIM_BRIGHTNESS% immediately so the screen
      isn't glaring while you walk away
    """
    global _original_brightness
    if platform.system() != "Windows":
        return

    ES_CONTINUOUS      = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    )
    print("[run] Sleep prevention active — system will stay awake.")
    print("[run] Screensaver and display dimming are still enabled as normal.")

    _original_brightness = _get_brightness()
    _set_brightness(_DIM_BRIGHTNESS)
    if _original_brightness is not None:
        print(f"[run] Brightness lowered: {_original_brightness}% → {_DIM_BRIGHTNESS}% "
              f"(restored automatically when done)")

def allow_sleep():
    """Restore normal sleep behaviour and original brightness after the run."""
    if platform.system() != "Windows":
        return

    ES_CONTINUOUS = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("[run] Sleep prevention lifted.")

    if _original_brightness is not None:
        _set_brightness(_original_brightness)
        print(f"[run] Brightness restored to {_original_brightness}%")

# Make sure the systems/ directory is on the path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import build_all_systems, BaselineSystem, RAGSystem, RunResult
from config   import QA_CSV_PATH, RESULTS_DIR, TOP_K, CHUNK_STRATEGY


# ── Load evaluation questions ─────────────────────────────────────────────────

def load_qa_pairs(csv_path: Path, limit: int = None) -> list[dict]:
    """Load QA pairs from the evaluation CSV, skipping unfilled rows."""
    rows = []
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            q = row.get("question_ar", "").strip()
            a = row.get("reference_answer_ar", "").strip()
            if not q:
                continue
            rows.append({
                "question_id"        : row["question_id"],
                "question_ar"        : q,
                "reference_answer_ar": a,
                "source_law"         : row.get("source_law", ""),
                "article_number"     : row.get("article_number", ""),
                "topic_category"     : row.get("topic_category", ""),
                "difficulty"         : row.get("difficulty", ""),
            })
    if limit:
        rows = rows[:limit]
    return rows


# ── Save helpers ──────────────────────────────────────────────────────────────

def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main experiment loop ──────────────────────────────────────────────────────

def run(args) -> None:
    prevent_sleep()
    # Determine chunk strategies to use
    if CHUNK_STRATEGY == "both":
        strategies = ["article", "fixed"]
    else:
        strategies = [CHUNK_STRATEGY]

    # Build all systems
    print("[run] Building systems …")
    all_systems = build_all_systems(chunk_strategies=strategies, top_k=args.top_k)

    # Filter to requested systems
    if args.systems:
        all_systems = [s for s in all_systems if s.name in args.systems]
        if not all_systems:
            print(f"[run] No systems matched --systems filter: {args.systems}")
            return

    print(f"[run] Systems to evaluate: {[s.name for s in all_systems]}")

    # Load evaluation questions
    qa_pairs = load_qa_pairs(QA_CSV_PATH, limit=args.limit)
    print(f"[run] Loaded {len(qa_pairs)} QA pairs from {QA_CSV_PATH}")
    if not qa_pairs:
        print("[run] No filled questions found. Fill in question_ar in qa_pairs.csv first.")
        return

    # Output files
    run_id     = datetime.now().strftime("%Y%m%d_%H%M%S")
    pred_path  = RESULTS_DIR / f"predictions_{run_id}.jsonl"
    retr_path  = RESULTS_DIR / f"retrieved_{run_id}.jsonl"
    print(f"[run] Saving predictions → {pred_path}")

    total = len(all_systems) * len(qa_pairs)
    done  = 0

    for system in all_systems:
        print(f"\n{'='*60}")
        print(f"[run] System: {system.name}  ({len(qa_pairs)} questions)")
        print(f"{'='*60}")

        for qa in qa_pairs:
            try:
                result: RunResult = system.run(qa["question_ar"])
            except Exception as e:
                done += 1
                pct = 100 * done / total
                print(
                    f"  [{pct:5.1f}%] {system.name} | {qa['question_id']} | "
                    f"[ERROR: {type(e).__name__}: {str(e)[:80]}]"
                )
                pred_record = {
                    "run_id"          : run_id,
                    "system_name"     : system.name,
                    "chunk_strategy"  : getattr(system, "chunk_strategy", "none"),
                    "question_id"     : qa["question_id"],
                    "question_ar"     : qa["question_ar"],
                    "reference_answer": qa["reference_answer_ar"],
                    "predicted_answer": f"[ERROR: {type(e).__name__}]",
                    "article_number"  : qa["article_number"],
                    "topic_category"  : qa["topic_category"],
                    "difficulty"      : qa["difficulty"],
                    "latency_s"       : -1,
                }
                append_jsonl(pred_path, pred_record)
                continue

            done += 1

            # Prediction record
            pred_record = {
                "run_id"          : run_id,
                "system_name"     : result.system_name,
                "chunk_strategy"  : result.chunk_strategy,
                "question_id"     : qa["question_id"],
                "question_ar"     : qa["question_ar"],
                "reference_answer": qa["reference_answer_ar"],
                "predicted_answer": result.answer,
                "article_number"  : qa["article_number"],
                "topic_category"  : qa["topic_category"],
                "difficulty"      : qa["difficulty"],
                "latency_s"       : round(result.latency_s, 3),
            }
            append_jsonl(pred_path, pred_record)

            # Retrieved passages record (RAG systems only)
            if result.retrieved:
                retr_record = {
                    "run_id"        : run_id,
                    "system_name"   : result.system_name,
                    "chunk_strategy": result.chunk_strategy,
                    "question_id"   : qa["question_id"],
                    "question_ar"   : qa["question_ar"],
                    "article_number": qa["article_number"],
                    "passages"      : [
                        {
                            "rank"      : p["rank"],
                            "chunk_id"  : p["chunk_id"],
                            "score"     : p.get("score", 0),
                            "text_preview": p["text"][:150],
                        }
                        for p in result.retrieved
                    ],
                }
                append_jsonl(retr_path, retr_record)

            pct = 100 * done / total
            print(
                f"  [{pct:5.1f}%] {system.name} | {qa['question_id']} | "
                f"{result.latency_s:.1f}s | ans: {result.answer[:60]}…"
            )

    print(f"\n[run] Done. {done} predictions saved to {pred_path}")
    print(f"[run] Run evaluate.py next to compute metrics.")
    allow_sleep()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run all RAG systems over the evaluation QA set."
    )
    parser.add_argument(
        "--systems", nargs="+", default=None,
        help="Subset of systems to run (default: all). "
             "Options: baseline bm25_article bm25_fixed dense_article dense_fixed "
             "hybrid_article hybrid_fixed",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N questions (useful for testing).",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K,
        help=f"Passages to retrieve (default: {TOP_K}).",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
