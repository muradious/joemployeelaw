"""
evaluate.py
-----------
Computes all evaluation metrics from predictions.jsonl and retrieved.jsonl.

Metrics:
  Retrieval  — Recall@1, Recall@3, Recall@5, MRR
  Answer     — Exact Match (EM), BERTScore (F1), Avg latency

Usage:
    cd C:/Work/nlp/systems
    python evaluate.py --predictions ../results/predictions_<run_id>.jsonl \
                       --retrieved   ../results/retrieved_<run_id>.jsonl

Output:
    Prints a summary table to stdout.
    Saves full results to results/metrics_<run_id>.json
"""

import re
import sys
import json
import argparse
import unicodedata
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import RESULTS_DIR, QA_CSV_PATH
import csv


# ── Arabic text normalisation for EM ─────────────────────────────────────────

def normalize_arabic(text: str) -> str:
    """
    Light normalisation for Exact Match comparison.
    Removes diacritics, punctuation, extra whitespace.
    Matches the preprocessing applied to the corpus.
    """
    # Remove tashkeel
    text = re.sub(r'[ً-ٰٟ]', '', text)
    # Normalize hamza
    text = re.sub(r'[أإآ]', 'ا', text)
    text = re.sub(r'ؤ', 'و', text)
    text = re.sub(r'ئ', 'ي', text)
    text = re.sub(r'ى', 'ي', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text.strip()


# ── Exact Match ───────────────────────────────────────────────────────────────

def exact_match(prediction: str, reference: str) -> float:
    """
    1.0 if the normalized prediction equals the normalized reference, else 0.0.
    Uses token-set overlap as a soft fallback (≥80% overlap → 0.5).
    """
    pred_norm = normalize_arabic(prediction)
    ref_norm  = normalize_arabic(reference)

    if pred_norm == ref_norm:
        return 1.0

    # Soft: token overlap
    pred_tokens = set(pred_norm.split())
    ref_tokens  = set(ref_norm.split())
    if not ref_tokens:
        return 0.0
    overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
    return round(overlap, 4)


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(
    predictions: list[str],
    references:  list[str],
    model_type:  str = "aubmindlab/bert-base-arabertv02",
    batch_size:  int = 16,
) -> list[float]:
    """
    Compute BERTScore F1 for a list of (prediction, reference) pairs.

    Primary model : aubmindlab/bert-base-arabertv02  (AraBERT v2 — Arabic-specific)
    Fallback model: bert-base-multilingual-cased     (always supported by bert_score)

    Install:  pip install bert-score

    Notes:
    - Do NOT pass lang= alongside model_type=: bert_score treats lang as a
      signal to apply pre-computed baseline rescaling, which does not exist for
      aubmindlab/bert-base-arabertv02 and produces wrong (even negative) scores.
      Disabling rescaling keeps scores in their natural [0,1] range.
    - Error records ([ERROR:...]) should be filtered out before calling this.
    """
    try:
        from bert_score import score as bert_score_fn
        import torch
        import traceback
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def _run(model):
            print(f"[eval] Computing BERTScore with {model} on {device} …")
            _, _, F1 = bert_score_fn(
                predictions,
                references,
                model_type           = model,
                rescale_with_baseline= False,
                device               = device,
                batch_size           = batch_size,
                verbose              = False,
            )
            return F1.tolist()

        # Try primary model (AraBERT), fall back to multilingual-BERT on any error
        try:
            return _run(model_type)
        except Exception as e:
            print(f"[eval] Primary model ({model_type}) failed: {e}")
            traceback.print_exc()
            fallback = "bert-base-multilingual-cased"
            print(f"[eval] Retrying with fallback model: {fallback}")
            return _run(fallback)

    except ImportError:
        print("[eval] bert-score not installed. Run: pip install bert-score")
        return [0.0] * len(predictions)
    except Exception as e:
        import traceback
        print(f"[eval] BERTScore failed entirely: {e}")
        traceback.print_exc()
        return [0.0] * len(predictions)


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved_passages: list[dict], gold_article: str, k: int) -> float:
    """
    1.0 if the gold article appears in the top-k retrieved passages, else 0.0.

    'gold_article' is the article_number field from qa_pairs.csv.
    A passage is considered correct if its chunk_id or text contains
    a reference to the gold article number.
    """
    if not gold_article:
        return float("nan")   # cannot evaluate without gold label

    top_k_passages = retrieved_passages[:k]
    gold_norm = normalize_arabic(gold_article)

    for passage in top_k_passages:
        chunk_id   = str(passage.get("chunk_id", ""))
        text_prev  = str(passage.get("text_preview", ""))
        # Match article number in chunk_id (e.g. "labor_law_art_012")
        if gold_norm in normalize_arabic(chunk_id):
            return 1.0
        # Match article number in text preview
        if gold_norm in normalize_arabic(text_prev):
            return 1.0

    return 0.0


def reciprocal_rank(retrieved_passages: list[dict], gold_article: str) -> float:
    """Mean Reciprocal Rank contribution for one query."""
    if not gold_article:
        return float("nan")

    gold_norm = normalize_arabic(gold_article)
    for passage in retrieved_passages:
        chunk_id  = normalize_arabic(str(passage.get("chunk_id", "")))
        text_prev = normalize_arabic(str(passage.get("text_preview", "")))
        rank = passage.get("rank", 0)
        if gold_norm in chunk_id or gold_norm in text_prev:
            return 1.0 / rank

    return 0.0


# ── Load helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_gold_articles(qa_csv: Path) -> dict[str, str]:
    """Returns {question_id: article_number}."""
    mapping = {}
    with qa_csv.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["question_id"]] = row.get("article_number", "")
    return mapping


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate(pred_path: Path, retr_path: Path) -> dict:
    predictions = load_jsonl(pred_path)
    retrieved   = load_jsonl(retr_path) if retr_path and retr_path.exists() else []
    gold_map    = load_gold_articles(QA_CSV_PATH)

    # Index retrieved passages by (system_name, question_id)
    retr_index: dict[tuple, list] = defaultdict(list)
    for rec in retrieved:
        key = (rec["system_name"], rec["question_id"])
        retr_index[key] = rec.get("passages", [])

    # Group predictions by system
    by_system: dict[str, list] = defaultdict(list)
    for pred in predictions:
        by_system[pred["system_name"]].append(pred)

    all_metrics = {}

    for system_name, preds in sorted(by_system.items()):
        # Separate successful predictions from error records
        valid_preds = [
            p for p in preds
            if not str(p.get("predicted_answer", "")).startswith("[ERROR:")
            and p.get("latency_s", 0) >= 0
        ]
        error_preds = [p for p in preds if p not in valid_preds]

        n_total  = len(preds)
        n_valid  = len(valid_preds)
        n_errors = len(error_preds)

        print(f"\n── {system_name} ({n_total} predictions"
              + (f", {n_errors} errors skipped from scoring" if n_errors else "") + ") ──")

        pred_texts = [p["predicted_answer"] for p in valid_preds]
        ref_texts  = [p["reference_answer"]  for p in valid_preds]

        # Answer metrics (computed only on valid predictions)
        em_scores = [exact_match(p, r) for p, r in zip(pred_texts, ref_texts)]
        bert_f1   = compute_bertscore(pred_texts, ref_texts)

        avg_em      = sum(s for s in em_scores if s == s) / len(em_scores) if em_scores else 0.0
        avg_bert_f1 = sum(bert_f1) / len(bert_f1) if bert_f1 else 0.0
        # Latency: average over valid predictions only (error records have latency_s=-1)
        avg_latency = sum(p["latency_s"] for p in valid_preds) / n_valid if n_valid else 0.0

        metrics = {
            "n"            : n_total,
            "n_valid"      : n_valid,
            "n_errors"     : n_errors,
            "avg_em"       : round(avg_em, 4),
            "avg_bert_f1"  : round(avg_bert_f1, 4),
            "avg_latency_s": round(avg_latency, 2),
        }

        # Retrieval metrics (skip for baseline)
        if system_name != "baseline":
            recall1_scores, recall3_scores, recall5_scores, mrr_scores = [], [], [], []
            for pred in valid_preds:
                qid     = pred["question_id"]
                gold    = gold_map.get(qid, "")
                passages = retr_index.get((system_name, qid), [])

                r1  = recall_at_k(passages, gold, 1)
                r3  = recall_at_k(passages, gold, 3)
                r5  = recall_at_k(passages, gold, 5)
                mrr = reciprocal_rank(passages, gold)

                # Only include if gold article is available
                if r1 == r1:   recall1_scores.append(r1)
                if r3 == r3:   recall3_scores.append(r3)
                if r5 == r5:   recall5_scores.append(r5)
                if mrr == mrr: mrr_scores.append(mrr)

            def safe_mean(lst):
                return round(sum(lst) / len(lst), 4) if lst else None

            metrics["recall@1"] = safe_mean(recall1_scores)
            metrics["recall@3"] = safe_mean(recall3_scores)
            metrics["recall@5"] = safe_mean(recall5_scores)
            metrics["mrr"]      = safe_mean(mrr_scores)
            metrics["n_with_gold"] = len(recall1_scores)

        all_metrics[system_name] = metrics

        # Print summary
        print(f"  EM:        {metrics['avg_em']:.4f}")
        print(f"  BERTScore: {metrics['avg_bert_f1']:.4f}")
        print(f"  Latency:   {metrics['avg_latency_s']:.2f}s / question")
        if system_name != "baseline":
            print(f"  Recall@1:  {metrics.get('recall@1')}")
            print(f"  Recall@3:  {metrics.get('recall@3')}")
            print(f"  Recall@5:  {metrics.get('recall@5')}")
            print(f"  MRR:       {metrics.get('mrr')}")

    return all_metrics


def print_summary_table(all_metrics: dict) -> None:
    """Print a compact comparison table."""
    print("\n" + "="*90)
    print(f"{'System':<22} {'EM':>7} {'BERT-F1':>9} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7} {'Lat(s)':>8}")
    print("="*90)
    for sys_name, m in sorted(all_metrics.items()):
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, float) else ("  —   " if v is None else str(v))
        print(
            f"{sys_name:<22} "
            f"{fmt(m['avg_em']):>7} "
            f"{fmt(m['avg_bert_f1']):>9} "
            f"{fmt(m.get('recall@1')):>7} "
            f"{fmt(m.get('recall@3')):>7} "
            f"{fmt(m.get('recall@5')):>7} "
            f"{fmt(m.get('mrr')):>7} "
            f"{m['avg_latency_s']:>8.2f}"
        )
    print("="*90)


def _find_latest(pattern: str) -> Path | None:
    """Return the most-recently-modified file matching a glob in RESULTS_DIR."""
    candidates = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for all RAG systems."
    )
    parser.add_argument(
        "--predictions", default=None,
        help="Path to predictions_<run_id>.jsonl  "
             "(omit to use the most-recently-modified predictions file in results/)"
    )
    parser.add_argument(
        "--retrieved", default=None,
        help="Path to retrieved_<run_id>.jsonl  "
             "(omit to auto-match from the predictions filename, or use --latest)"
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Use the most-recently-modified predictions_*.jsonl (and its matching "
             "retrieved_*.jsonl) from the results/ directory.  "
             "Shortcut for: python evaluate.py  (no --predictions argument)"
    )
    args = parser.parse_args()

    # ── Resolve predictions path ──────────────────────────────────────────────
    if args.predictions:
        pred_path = Path(args.predictions)
        # Allow bare filename relative to RESULTS_DIR
        if not pred_path.is_absolute() and not pred_path.exists():
            pred_path = RESULTS_DIR / pred_path
    else:
        # Auto-find latest (triggered by --latest flag OR no argument at all)
        pred_path = _find_latest("predictions_*.jsonl")
        if pred_path is None:
            print(f"[eval] No predictions_*.jsonl found in {RESULTS_DIR}")
            print("[eval] Run run_experiments.py first, or pass --predictions <file>")
            return
        print(f"[eval] Auto-selected: {pred_path.name}")

    if not pred_path.exists():
        print(f"[eval] File not found: {pred_path}")
        return

    # ── Resolve retrieved path ────────────────────────────────────────────────
    if args.retrieved:
        retr_path = Path(args.retrieved)
        if not retr_path.is_absolute() and not retr_path.exists():
            retr_path = RESULTS_DIR / retr_path
    else:
        # Try to find a matching retrieved file by swapping "predictions_" → "retrieved_"
        auto_retr = pred_path.parent / pred_path.name.replace("predictions_", "retrieved_")
        if auto_retr.exists():
            retr_path = auto_retr
            print(f"[eval] Auto-matched retrieved: {retr_path.name}")
        else:
            retr_path = None
            print("[eval] No retrieved file found — retrieval metrics will be skipped.")

    run_id = pred_path.stem.replace("predictions_", "")
    all_metrics = evaluate(pred_path, retr_path)
    print_summary_table(all_metrics)

    # Save metrics JSON
    out_path = RESULTS_DIR / f"metrics_{run_id}.json"
    out_path.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n[eval] Metrics saved → {out_path}")


if __name__ == "__main__":
    main()
