"""
preprocess.py
-------------
Arabic legal text preprocessing for the Jordanian Labor Law RAG project.

Usage:
    python scripts/preprocess.py --input data/raw/labor_law.txt \
                                  --output data/processed/labor_law_clean.txt \
                                  --law-name "Jordanian Labor Law No. 8 of 1996"

The script applies:
  1. Hamza normalization  (أ إ آ → ا,  ؤ → و,  ئ → ي)
  2. Tashkeel (diacritics) removal
  3. Tatweel (kashida) removal
  4. Whitespace normalization
  5. Light legal terminology standardization

It then dumps a JSON file of structured articles to data/processed/.
"""

import re
import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# 1.  Arabic normalisation helpers
# ---------------------------------------------------------------------------

# Tashkeel: short vowels + shadda + sukun + superscript alef
TASHKEEL = re.compile(
    r'[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670]'
)

TATWEEL = re.compile(r'\u0640+')   # ـ  kashida / tatweel

# Hamza variants on seat → bare alef
HAMZA_ALEF = re.compile(r'[\u0622\u0623\u0625]')   # آ أ إ → ا
HAMZA_WAW  = re.compile(r'\u0624')                  # ؤ → و
HAMZA_YA   = re.compile(r'\u0626')                  # ئ → ي

# Alef maqsura → ya (common in legal texts)
ALEF_MAQSURA = re.compile(r'\u0649')               # ى → ي

# Ta marbuta normalisation (optional – off by default to keep morphology)
TA_MARBUTA = re.compile(r'\u0629\b')               # ة at word boundary → ه


def normalize_hamza(text: str) -> str:
    """Normalize all hamza forms to their base letters."""
    text = HAMZA_ALEF.sub('\u0627', text)   # → ا
    text = HAMZA_WAW.sub('\u0648', text)    # → و
    text = HAMZA_YA.sub('\u064A', text)     # → ي
    return text


def remove_tashkeel(text: str) -> str:
    """Strip all Arabic diacritical marks."""
    return TASHKEEL.sub('', text)


def remove_tatweel(text: str) -> str:
    """Remove kashida / tatweel characters."""
    return TATWEEL.sub('', text)


def normalize_alef_maqsura(text: str) -> str:
    """Normalize ى to ي for consistent tokenisation."""
    return ALEF_MAQSURA.sub('\u064A', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/tabs, strip leading/trailing whitespace per line."""
    lines = text.splitlines()
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    # Collapse runs of blank lines to a single blank line
    cleaned, prev_blank = [], False
    for line in lines:
        is_blank = line == ''
        if is_blank and prev_blank:
            continue
        cleaned.append(line)
        prev_blank = is_blank
    return '\n'.join(cleaned)


def preprocess(text: str,
               normalize_hamza_flag: bool = True,
               remove_tashkeel_flag: bool = True,
               remove_tatweel_flag: bool = True,
               normalize_alef_flag: bool = True) -> str:
    """
    Apply the full normalisation pipeline to raw Arabic legal text.

    Parameters
    ----------
    text : str
        Raw input text (UTF-8).
    normalize_hamza_flag : bool
        Whether to normalize hamza variants. Default True.
    remove_tashkeel_flag : bool
        Whether to strip diacritics. Default True.
    remove_tatweel_flag : bool
        Whether to strip kashida. Default True.
    normalize_alef_flag : bool
        Whether to map ى → ي. Default True.

    Returns
    -------
    str
        Cleaned text.
    """
    if normalize_hamza_flag:
        text = normalize_hamza(text)
    if remove_tashkeel_flag:
        text = remove_tashkeel(text)
    if remove_tatweel_flag:
        text = remove_tatweel(text)
    if normalize_alef_flag:
        text = normalize_alef_maqsura(text)
    text = normalize_whitespace(text)
    return text


# ---------------------------------------------------------------------------
# 2.  Article parsing
# ---------------------------------------------------------------------------

# Matches Arabic article HEADERS only — not in-body citations of other articles.
#
# Handles formats like:
#   "المادة 1   يسمى هذا القانون..."   (number + body on same line)
#   "المادة (1)\nيسمى..."              (parenthesised number, body on next line)
#   "المادة الأولى\n..."               (Arabic ordinal)
#
# Excludes in-body citations such as:
#   "وفقاً للمادة 15 من هذا القانون"   → followed by  من
#   "المادة 7 أعلاه"                   → followed by  أعلاه
#   "المادة 3 إلى 5"                   → followed by  إلى  (article range)
#   "المادة 12 السابقة"               → followed by  السابقة
#
# Strategy: negative lookahead — after matching المادة + number, reject the match
# if the next non-whitespace token is a known citation qualifier.
# Article body text starts with a verb or noun, never these qualifiers.

_CITATION_QUALIFIERS = '|'.join([
    r'من',          # from        — المادة 15 من هذا القانون
    r'اعلاه',   # أعلاه above  (normalized: أ→ا)
    r'ادناه',   # أدناه below  (normalized: أ→ا)
    r'الي',     # إلى   to/range  (normalized: إ→ا, ى→ي)
    r'و\b',             # و     and    — list: المادة 3 و4
    r'او',          # أو    or      (normalized: أ→ا)
    r'المشار',   # المشار  referred to
    r'السابق',   # السابق  previous
    r'التالي',   # التالي  following
    r'المذكور',   # المذكور  mentioned
    r'الانف',    # الآنف aforementioned (normalized: آ→ا)
    r'الفقرة',   # الفقرة  paragraph ref
    r'البند',    # البند   clause ref
    r'في',      # في     in  — المادة في حالة... (mid-sentence ref)
])

# Two-part defence against false positives:
#   1. Negative lookbehind (?<![\u0600-\u06FF]) — skips المادة that is directly
#      preceded by an Arabic letter (e.g. والمادة, فالمادة).
#   2. Negative lookahead — skips المادة X when followed by a citation qualifier
#      (e.g. "المادة 15 من", "المادة 7 اعلاه").
ARTICLE_PATTERN = re.compile(
    r'(?<![\u0600-\u06FF])'              # NOT directly preceded by Arabic letter
    r'(المادة\s+'                       # keyword + whitespace
    r'(?:'
    r'\([0-9\u0660-\u0669\u06F0-\u06F9]+\)'  # (1) parenthesised digit
    r'|[0-9\u0660-\u0669\u06F0-\u06F9]+'       # plain / Arabic-indic digit
    r'|[\u0600-\u06FF]+'                          # Arabic ordinal word (e.g. الاولي)
    r')'
    r')'
    r'(?!\s*(?:' + _CITATION_QUALIFIERS + r'))'    # NOT followed by citation qualifier
)


def parse_articles(text: str, law_name: str = '') -> list[dict]:
    """
    Split preprocessed text into individual articles.

    Each article dict contains:
      - article_id   : int  (sequential, 1-indexed)
      - article_header : str  (the raw المادة X line)
      - article_number : str  (extracted number/label from header)
      - law_name     : str
      - text         : str  (full article text including header)

    Returns an empty list if no article boundaries are found
    (caller should fall back to fixed-token chunking).
    """
    matches = list(ARTICLE_PATTERN.finditer(text))

    # Filter out false-positive matches.
    # A valid article number must be either:
    #   (a) purely numeric  — digits only (Western or Arabic-indic), with
    #       optional parentheses, e.g. "1", "15", "(3)", "١٢"
    #   (b) a proper Arabic ordinal starting with ال (the definite article),
    #       e.g. "الأولى", "الثانية"  (after normalisation: "الاولي", "الثانية")
    # Anything else (verbs, nouns, prepositions, etc.) is mid-sentence noise.
    _VALID_NUMBER = re.compile(
        r'^[\(\)0-9٠-٩۰-۹]+$'   # (a) numeric
        r'|^ال'                     # (b) definite-article ordinal
    )

    matches = [
        m for m in matches
        if _VALID_NUMBER.match(
            re.sub(r'^المادة\s*', '', m.group(1)).strip()
        )
    ]

    if not matches:
        return []

    articles = []
    for idx, match in enumerate(matches):
        start = match.start()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body  = text[start:end].strip()

        header = match.group(1).strip()
        # Extract the article number/label (everything after المادة)
        number_raw = re.sub(r'^المادة\s*', '', header).strip()

        articles.append({
            'article_id'     : idx + 1,
            'article_header' : header,
            'article_number' : number_raw,
            'law_name'       : law_name,
            'text'           : body,
            'char_count'     : len(body),
        })

    return articles


def extract_preamble(text: str) -> str:
    """Return text that appears before the first article (preamble / definitions)."""
    match = ARTICLE_PATTERN.search(text)
    if match:
        return text[:match.start()].strip()
    return text.strip()


# ---------------------------------------------------------------------------
# 3.  CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Arabic legal text for the NLP RAG project.'
    )
    parser.add_argument('--input',  required=True,
                        help='Path to raw .txt file (UTF-8).')
    parser.add_argument('--output', required=True,
                        help='Path for the output cleaned .txt file.')
    parser.add_argument('--law-name', default='',
                        help='Human-readable name for this law (stored in JSON).')
    parser.add_argument('--json-out', default=None,
                        help='Path for the structured articles JSON output. '
                             'Defaults to same dir as --output.')
    parser.add_argument('--no-hamza',    action='store_true',
                        help='Skip hamza normalization.')
    parser.add_argument('--no-tashkeel', action='store_true',
                        help='Skip tashkeel removal.')
    parser.add_argument('--no-tatweel',  action='store_true',
                        help='Skip tatweel removal.')
    parser.add_argument('--no-alef',     action='store_true',
                        help='Skip ى → ي normalization.')
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    raw_text = input_path.read_text(encoding='utf-8')
    print(f'[preprocess] Read {len(raw_text):,} characters from {input_path}')

    cleaned = preprocess(
        raw_text,
        normalize_hamza_flag = not args.no_hamza,
        remove_tashkeel_flag = not args.no_tashkeel,
        remove_tatweel_flag  = not args.no_tatweel,
        normalize_alef_flag  = not args.no_alef,
    )

    output_path.write_text(cleaned, encoding='utf-8')
    print(f'[preprocess] Wrote cleaned text → {output_path}')

    # Parse and save articles JSON
    articles = parse_articles(cleaned, law_name=args.law_name)
    preamble = extract_preamble(cleaned)

    json_path = Path(args.json_out) if args.json_out else \
                output_path.parent / (output_path.stem + '_articles.json')

    output_data = {
        'law_name' : args.law_name,
        'source'   : str(input_path),
        'preamble' : preamble,
        'total_articles': len(articles),
        'articles' : articles,
    }

    json_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(f'[preprocess] Found {len(articles)} articles → {json_path}')
    if not articles:
        print('[preprocess] WARNING: No article boundaries detected. '
              'Check that the text contains "المادة X" headers.')


if __name__ == '__main__':
    main()
