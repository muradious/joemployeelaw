"""
validate.py
-----------
Quick data validation for all pipeline outputs.

Checks:
  1. Article chunks  – no empty texts, reasonable word counts
  2. Fixed chunks    – correct overlap, no empty chunks
  3. Evaluation CSV  – correct columns, no duplicate question IDs,
                       all difficulty values valid

Usage:
    python scripts/validate.py \
        --chunks-dir data/chunks/ \
        --eval       evaluation/qa_pairs.csv

Exits with code 0 on success, 1 if any check fails.
"""

import sys
import csv
import json
import argparse
from pathlib import Path


PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
WARN = '\033[93m⚠\033[0m'

errors = []


def check(condition: bool, message: str, fatal: bool = True) -> bool:
    if condition:
        print(f'  {PASS} {message}')
        return True
    else:
        symbol = FAIL if fatal else WARN
        print(f'  {symbol} {message}')
        if fatal:
            errors.append(message)
        return False


# ---------------------------------------------------------------------------
# 1. Validate article chunks
# ---------------------------------------------------------------------------

def validate_article_chunks(path: Path) -> None:
    print(f'\n[Article Chunks]  {path}')
    if not path.exists():
        print(f'  {FAIL} File not found: {path}')
        errors.append(f'Missing: {path}')
        return

    chunks = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    check(len(chunks) > 0, f'{len(chunks)} chunks loaded')

    empty_texts = [c['chunk_id'] for c in chunks if not c.get('text', '').strip()]
    check(len(empty_texts) == 0,
          f'No empty article texts' if not empty_texts else
          f'{len(empty_texts)} empty texts: {empty_texts[:5]}')

    short = [c for c in chunks if c.get('word_count', 0) < 5]
    check(len(short) == 0,
          f'No suspiciously short articles (< 5 words)',
          fatal=False)
    if short:
        print(f'    (short articles: {[c["chunk_id"] for c in short[:5]]})')

    ids = [c['chunk_id'] for c in chunks]
    check(len(ids) == len(set(ids)), 'All chunk_ids are unique')

    required_fields = {'chunk_id', 'strategy', 'law_name', 'article_id',
                       'article_number', 'text', 'word_count'}
    for c in chunks:
        missing = required_fields - set(c.keys())
        if missing:
            check(False, f'Chunk {c["chunk_id"]} missing fields: {missing}')
            break
    else:
        check(True, 'All required fields present')


# ---------------------------------------------------------------------------
# 2. Validate fixed chunks
# ---------------------------------------------------------------------------

def validate_fixed_chunks(path: Path, window: int = 256, stride: int = 50) -> None:
    print(f'\n[Fixed Chunks]  {path}')
    if not path.exists():
        print(f'  {FAIL} File not found: {path}')
        errors.append(f'Missing: {path}')
        return

    chunks = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    check(len(chunks) > 0, f'{len(chunks)} chunks loaded')

    empty = [c['chunk_id'] for c in chunks if not c.get('text', '').strip()]
    check(len(empty) == 0, f'No empty chunks')

    # Check that all but the last chunk have exactly `window` words
    interior = [c for c in chunks[:-1] if c.get('word_count', 0) != window]
    check(len(interior) == 0,
          f'Interior chunks have correct window size ({window} tokens)',
          fatal=False)

    # Check overlap: consecutive token_start values should differ by stride
    if len(chunks) > 1:
        bad_stride = []
        for i in range(len(chunks) - 1):
            expected_stride = chunks[i + 1]['token_start'] - chunks[i]['token_start']
            if expected_stride != stride:
                bad_stride.append((chunks[i]['chunk_id'], expected_stride))
        check(len(bad_stride) == 0,
              f'Stride is consistently {stride} tokens between chunks' if not bad_stride else
              f'Stride mismatches found: {bad_stride[:3]}')

    ids = [c['chunk_id'] for c in chunks]
    check(len(ids) == len(set(ids)), 'All chunk_ids are unique')


# ---------------------------------------------------------------------------
# 3. Validate evaluation CSV
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = {
    'question_id', 'question_ar', 'reference_answer_ar',
    'source_law', 'article_number', 'topic_category', 'difficulty'
}
VALID_DIFFICULTIES = {'Easy', 'Medium', 'Hard'}
EXPECTED_ROWS = 80


def validate_eval_csv(path: Path) -> None:
    print(f'\n[Evaluation CSV]  {path}')
    if not path.exists():
        print(f'  {FAIL} File not found: {path}')
        errors.append(f'Missing: {path}')
        return

    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        cols = set(reader.fieldnames or [])

    missing_cols = REQUIRED_COLUMNS - cols
    check(len(missing_cols) == 0,
          f'All required columns present' if not missing_cols else
          f'Missing columns: {missing_cols}')

    check(len(rows) == EXPECTED_ROWS,
          f'Exactly {EXPECTED_ROWS} rows (found {len(rows)})',
          fatal=False)

    ids = [r.get('question_id', '') for r in rows]
    check(len(ids) == len(set(ids)), 'All question_ids are unique')

    bad_diff = [r['question_id'] for r in rows
                if r.get('difficulty', '') not in VALID_DIFFICULTIES]
    check(len(bad_diff) == 0,
          f'All difficulty values are valid ({VALID_DIFFICULTIES})' if not bad_diff else
          f'Invalid difficulty in rows: {bad_diff[:5]}')

    filled_q = sum(1 for r in rows if r.get('question_ar', '').strip())
    filled_a = sum(1 for r in rows if r.get('reference_answer_ar', '').strip())
    print(f'  {WARN} Questions filled: {filled_q}/{len(rows)}'
          f'  |  Answers filled: {filled_a}/{len(rows)}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate pipeline outputs for the NLP RAG project.'
    )
    parser.add_argument('--chunks-dir', default='data/chunks',
                        help='Directory containing article_chunks.jsonl and fixed_chunks.jsonl.')
    parser.add_argument('--eval', default='evaluation/qa_pairs.csv',
                        help='Path to the evaluation CSV.')
    parser.add_argument('--window', type=int, default=256)
    parser.add_argument('--stride', type=int, default=50)
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)

    validate_article_chunks(chunks_dir / 'article_chunks.jsonl')
    validate_fixed_chunks(chunks_dir / 'fixed_chunks.jsonl',
                          window=args.window, stride=args.stride)
    validate_eval_csv(Path(args.eval))

    print()
    if errors:
        print(f'\033[91m[validate] FAILED — {len(errors)} error(s):\033[0m')
        for e in errors:
            print(f'  • {e}')
        sys.exit(1)
    else:
        print('\033[92m[validate] All checks passed.\033[0m')
        sys.exit(0)


if __name__ == '__main__':
    main()
