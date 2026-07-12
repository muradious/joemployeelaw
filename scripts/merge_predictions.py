"""
merge_predictions.py
--------------------
Merges two or more predictions_*.jsonl files into a single file.
Use this when you ran experiments in multiple batches (e.g. re-ran one
system after fixing a bug) and need a single file to pass to evaluate.py.

Deduplication rule:
  For each (system_name, question_id) pair, keep the entry from the
  LAST file listed that does NOT have a [ERROR:...] predicted_answer.
  If all entries for a pair are errors, keep the last one.

Usage:
    cd C:/Work/nlp
    python scripts/merge_predictions.py \\
        results/predictions_20260515_120000.jsonl \\
        results/predictions_20260515_180000.jsonl \\
        --output results/predictions_merged.jsonl

    # Auto-merge ALL predictions_*.jsonl files in results/:
    python scripts/merge_predictions.py --all

    # Same but also merge retrieved_*.jsonl files:
    python scripts/merge_predictions.py --all --merge-retrieved
"""

import json
import argparse
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def is_error(record: dict) -> bool:
    answer = record.get("predicted_answer", "")
    return str(answer).startswith("[ERROR:")


def merge_files(paths: list[Path]) -> list[dict]:
    """
    Merge records from multiple files.
    Key: (system_name, question_id)
    Priority: non-error > error; later file > earlier file within same tier.
    """
    # best[key] = (is_error, record)
    best: dict[tuple, tuple] = {}

    for path in paths:
        print(f"  Loading {path.name}  …", end="  ")
        records = load_jsonl(path)
        print(f"{len(records)} records")

        for rec in records:
            key = (rec.get("system_name", ""), rec.get("question_id", ""))
            err = is_error(rec)

            if key not in best:
                best[key] = (err, rec)
            else:
                prev_err, _ = best[key]
                # Replace if: current is good and previous was error,
                # OR both same tier (always take the newer/later file)
                if (not err) or prev_err:
                    best[key] = (err, rec)

    merged = [rec for _, rec in best.values()]

    # Sort by system_name then question_id for readability
    merged.sort(key=lambda r: (r.get("system_name", ""), r.get("question_id", "")))
    return merged


def merge_retrieved_files(paths: list[Path]) -> list[dict]:
    """
    Merge retrieved_*.jsonl files.
    Key: (system_name, question_id) — keep last seen entry.
    """
    best: dict[tuple, dict] = {}

    for path in paths:
        print(f"  Loading {path.name}  …", end="  ")
        records = load_jsonl(path)
        print(f"{len(records)} records")
        for rec in records:
            key = (rec.get("system_name", ""), rec.get("question_id", ""))
            best[key] = rec   # later file wins

    merged = list(best.values())
    merged.sort(key=lambda r: (r.get("system_name", ""), r.get("question_id", "")))
    return merged


def summarise(records: list[dict]) -> None:
    from collections import Counter
    counts  = Counter(r.get("system_name", "?") for r in records)
    errors  = Counter(
        r.get("system_name", "?") for r in records if is_error(r)
    )
    print(f"\n  {'System':<28} {'Total':>7} {'Errors':>8}")
    print(f"  {'-'*44}")
    for sys in sorted(counts):
        print(f"  {sys:<28} {counts[sys]:>7} {errors.get(sys, 0):>8}")
    print(f"  {'-'*44}")
    print(f"  {'TOTAL':<28} {sum(counts.values()):>7} {sum(errors.values()):>8}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple predictions_*.jsonl files into one."
    )
    parser.add_argument(
        "files", nargs="*",
        help="Two or more predictions_*.jsonl files to merge (in priority order, "
             "last file wins for non-error entries)."
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Merge ALL predictions_*.jsonl files found in results/ "
             "(sorted by filename = chronological order)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for merged file "
             "(default: results/predictions_merged.jsonl)."
    )
    parser.add_argument(
        "--merge-retrieved", action="store_true",
        help="Also merge matching retrieved_*.jsonl files into "
             "results/retrieved_merged.jsonl."
    )
    args = parser.parse_args()

    # ── Collect prediction files ──────────────────────────────────────────────
    if args.all:
        pred_paths = sorted(RESULTS_DIR.glob("predictions_*.jsonl"))
        # Exclude any previously merged file so we don't double-count
        pred_paths = [p for p in pred_paths if "merged" not in p.name]
        if not pred_paths:
            print(f"[merge] No predictions_*.jsonl files found in {RESULTS_DIR}")
            return
    elif args.files:
        pred_paths = [Path(f) for f in args.files]
    else:
        parser.print_help()
        return

    print(f"\n[merge] Merging {len(pred_paths)} predictions file(s):")
    merged = merge_files(pred_paths)

    out_path = Path(args.output) if args.output else RESULTS_DIR / "predictions_merged.jsonl"
    write_jsonl(out_path, merged)
    print(f"\n[merge] {len(merged)} records written → {out_path}")
    summarise(merged)

    # ── Optionally merge retrieved files ─────────────────────────────────────
    if args.merge_retrieved:
        if args.all:
            retr_paths = sorted(RESULTS_DIR.glob("retrieved_*.jsonl"))
            retr_paths = [p for p in retr_paths if "merged" not in p.name]
        else:
            # Derive retrieved paths from prediction paths by name substitution
            retr_paths = []
            for p in pred_paths:
                r = p.parent / p.name.replace("predictions_", "retrieved_")
                if r.exists():
                    retr_paths.append(r)
                else:
                    print(f"  [warn] No matching retrieved file for {p.name}")

        if retr_paths:
            print(f"\n[merge] Merging {len(retr_paths)} retrieved file(s):")
            merged_retr = merge_retrieved_files(retr_paths)
            retr_out = RESULTS_DIR / "retrieved_merged.jsonl"
            write_jsonl(retr_out, merged_retr)
            print(f"[merge] {len(merged_retr)} records written → {retr_out}")
        else:
            print("[merge] No retrieved files found to merge.")

    print("\n[merge] Done. Pass the merged file to evaluate.py:")
    print(f"  python evaluate.py --predictions {out_path.name} "
          f"--retrieved retrieved_merged.jsonl")


if __name__ == "__main__":
    main()
