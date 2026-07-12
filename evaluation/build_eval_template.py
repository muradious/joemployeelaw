"""
build_eval_template.py
----------------------
Generates the empty 80-question evaluation set template as a CSV.

The template pre-fills:
  - question_id      : Q001 … Q080
  - source_law       : assigned law based on topic distribution
  - topic_category   : one of 14 categories drawn from Jordanian Labor Law
  - difficulty       : Easy / Medium / Hard   (balanced distribution)

Usage:
    python evaluation/build_eval_template.py \
        --out evaluation/qa_pairs.csv

The resulting CSV is intended to be opened in Excel / LibreOffice Calc.
"""

import csv
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Topic distribution (14 categories, 80 questions total)
# ---------------------------------------------------------------------------
# Each entry: (topic_category, source_law, count)
# Labor Law (No. 8 / 1996) covers ~60 questions
# Social Security Law (No. 1 / 2014) covers ~20 questions

TOPIC_DISTRIBUTION = [
    # ── Jordanian Labor Law ─────────────────────────────────────────────────
    ("Employment Contracts",           "Jordanian Labor Law No. 8 of 1996",       8),
    ("Wages and Compensation",         "Jordanian Labor Law No. 8 of 1996",       8),
    ("Working Hours and Overtime",     "Jordanian Labor Law No. 8 of 1996",       6),
    ("Annual and Sick Leave",          "Jordanian Labor Law No. 8 of 1996",       7),
    ("Maternity and Family Leave",     "Jordanian Labor Law No. 8 of 1996",       5),
    ("Termination and Severance",      "Jordanian Labor Law No. 8 of 1996",       8),
    ("Occupational Safety and Health", "Jordanian Labor Law No. 8 of 1996",       6),
    ("Child and Female Labour",        "Jordanian Labor Law No. 8 of 1996",       5),
    ("Labor Disputes and Arbitration", "Jordanian Labor Law No. 8 of 1996",       4),
    ("Collective Bargaining / Unions", "Jordanian Labor Law No. 8 of 1996",       3),
    # ── Social Security Law ─────────────────────────────────────────────────
    ("Social Security Contributions",  "Social Security Law No. 1 of 2014",       6),
    ("Retirement and Pension",         "Social Security Law No. 1 of 2014",       5),
    ("Work Injury Compensation",       "Social Security Law No. 1 of 2014",       5),
    ("Disability and Survivor Benefits","Social Security Law No. 1 of 2014",      4),
]

assert sum(t[2] for t in TOPIC_DISTRIBUTION) == 80, \
    "Topic distribution must sum to exactly 80 questions."


# ---------------------------------------------------------------------------
# Difficulty distribution within each topic
# (Easy ≈ 40%, Medium ≈ 40%, Hard ≈ 20%)
# ---------------------------------------------------------------------------

def assign_difficulty(count: int) -> list[str]:
    """Return a balanced list of difficulty labels for `count` questions."""
    easy   = round(count * 0.40)
    hard   = round(count * 0.20)
    medium = count - easy - hard
    return (['Easy'] * easy) + (['Medium'] * medium) + (['Hard'] * hard)


# ---------------------------------------------------------------------------
# CSV columns
# ---------------------------------------------------------------------------

COLUMNS = [
    'question_id',        # Q001 … Q080
    'question_ar',        # Arabic question (fill in manually)
    'reference_answer_ar',# Gold-standard Arabic answer (fill in manually)
    'source_law',         # Which law
    'article_number',     # e.g. "12" or "المادة 12"  (fill in manually)
    'topic_category',     # Pre-filled
    'difficulty',         # Easy / Medium / Hard
    'notes',              # Optional free-text notes
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_template(out_path: Path) -> None:
    rows = []
    qid = 1

    for (topic, law, count) in TOPIC_DISTRIBUTION:
        difficulties = assign_difficulty(count)
        for diff in difficulties:
            rows.append({
                'question_id'        : f'Q{qid:03d}',
                'question_ar'        : '',
                'reference_answer_ar': '',
                'source_law'         : law,
                'article_number'     : '',
                'topic_category'     : topic,
                'difficulty'         : diff,
                'notes'              : '',
            })
            qid += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8-sig') as f:
        # utf-8-sig adds BOM so Excel opens Arabic correctly on Windows
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[eval] Wrote {len(rows)}-row evaluation template → {out_path}')
    print('[eval] Open in Excel or LibreOffice Calc, then fill in:')
    print('         • question_ar          (Arabic question text)')
    print('         • reference_answer_ar  (gold-standard answer)')
    print('         • article_number       (source article, e.g. "المادة 12")')
    print('         • notes                (optional)')


def main():
    parser = argparse.ArgumentParser(
        description='Generate the 80-question evaluation set CSV template.'
    )
    parser.add_argument('--out', default='evaluation/qa_pairs.csv',
                        help='Output CSV path (default: evaluation/qa_pairs.csv).')
    args = parser.parse_args()
    build_template(Path(args.out))


if __name__ == '__main__':
    main()
