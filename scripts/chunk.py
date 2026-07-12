"""
chunk.py
--------
Produce both chunking strategies required by the project:

  Strategy A – Article-level chunks
      Each legal article is a single chunk, preserving the natural
      structural boundary of the law.

  Strategy B – Fixed-token chunks (256 tokens, 50-token stride)
      A sliding window is applied over the full document.
      Tokenisation uses whitespace splitting as a proxy for word tokens,
      which is appropriate for Arabic before a sub-word tokeniser is applied
      during embedding.  When AraBERT is used for encoding, its own
      tokeniser will handle sub-word splitting at inference time.

Usage:
    python scripts/chunk.py \
        --articles  data/processed/labor_law_articles.json \
        --clean-txt data/processed/labor_law_clean.txt \
        --out-dir   data/chunks/ \
        --law-name  "Jordanian Labor Law No. 8 of 1996"

Outputs (in --out-dir):
    article_chunks.jsonl   – one JSON object per line, strategy A
    fixed_chunks.jsonl     – one JSON object per line, strategy B
"""

import re
import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_articles(articles_json_path: str) -> tuple[str, list[dict]]:
    """Load the structured articles JSON produced by preprocess.py."""
    data = json.loads(Path(articles_json_path).read_text(encoding='utf-8'))
    return data.get('law_name', ''), data.get('articles', [])


def load_clean_text(clean_txt_path: str) -> str:
    return Path(clean_txt_path).read_text(encoding='utf-8')


def word_tokenize(text: str) -> list[str]:
    """
    Whitespace + punctuation tokeniser.

    Splits on spaces and keeps Arabic punctuation attached to tokens
    (AraBERT's sub-word tokeniser will handle further splitting).
    Returns a list of token strings.
    """
    return re.findall(r'\S+', text)


# ---------------------------------------------------------------------------
# Strategy A – Article-level chunks
# ---------------------------------------------------------------------------

def make_article_chunks(articles: list[dict], law_name: str) -> list[dict]:
    """
    Convert each parsed article into a chunk dict.

    Chunk schema:
        chunk_id       : str   unique identifier  e.g. "labor_law_art_001"
        strategy       : str   "article"
        law_name       : str
        article_id     : int
        article_number : str   e.g. "1" or "الأولى"
        article_header : str   e.g. "المادة 1"
        text           : str   full article text
        word_count     : int
        char_count     : int
    """
    chunks = []
    law_slug = re.sub(r'\W+', '_', law_name.lower())[:30].strip('_')

    for art in articles:
        text = art['text'].strip()
        words = word_tokenize(text)
        chunk_id = f"{law_slug}_art_{art['article_id']:03d}"

        chunks.append({
            'chunk_id'       : chunk_id,
            'strategy'       : 'article',
            'law_name'       : law_name,
            'article_id'     : art['article_id'],
            'article_number' : art['article_number'],
            'article_header' : art['article_header'],
            'text'           : text,
            'word_count'     : len(words),
            'char_count'     : len(text),
        })

    return chunks


# ---------------------------------------------------------------------------
# Strategy B – Fixed-token sliding window
# ---------------------------------------------------------------------------

def make_fixed_chunks(text: str,
                      law_name: str,
                      window: int = 256,
                      stride: int = 50) -> list[dict]:
    """
    Sliding-window chunking over the full document.

    Parameters
    ----------
    text   : str   Preprocessed document text.
    law_name : str
    window : int   Number of whitespace tokens per chunk (default 256).
    stride : int   Step size in tokens between consecutive chunks (default 50).
                   Overlap = window - stride = 206 tokens.

    Chunk schema:
        chunk_id     : str   e.g. "labor_law_fixed_0001"
        strategy     : str   "fixed"
        law_name     : str
        chunk_index  : int   0-indexed position
        token_start  : int   start token index in full document
        token_end    : int   end token index (exclusive)
        text         : str   the chunk text
        word_count   : int
        char_count   : int
    """
    tokens = word_tokenize(text)
    total_tokens = len(tokens)
    law_slug = re.sub(r'\W+', '_', law_name.lower())[:30].strip('_')

    chunks = []
    chunk_index = 0
    start = 0

    while start < total_tokens:
        end = min(start + window, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = ' '.join(chunk_tokens)

        chunks.append({
            'chunk_id'    : f"{law_slug}_fixed_{chunk_index:04d}",
            'strategy'    : 'fixed',
            'law_name'    : law_name,
            'chunk_index' : chunk_index,
            'token_start' : start,
            'token_end'   : end,
            'text'        : chunk_text,
            'word_count'  : len(chunk_tokens),
            'char_count'  : len(chunk_text),
        })

        # If we've reached the end, stop
        if end == total_tokens:
            break

        start += stride
        chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_jsonl(chunks: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


def print_stats(strategy: str, chunks: list[dict]) -> None:
    word_counts = [c['word_count'] for c in chunks]
    if not word_counts:
        print(f'[chunk:{strategy}] No chunks produced.')
        return
    avg  = sum(word_counts) / len(word_counts)
    minw = min(word_counts)
    maxw = max(word_counts)
    print(f'[chunk:{strategy}] {len(chunks)} chunks | '
          f'words: min={minw}, avg={avg:.0f}, max={maxw}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate article-level and fixed-token chunks for the NLP RAG project.'
    )
    parser.add_argument('--articles',  required=True,
                        help='Path to *_articles.json produced by preprocess.py.')
    parser.add_argument('--clean-txt', required=True,
                        help='Path to cleaned .txt file produced by preprocess.py.')
    parser.add_argument('--out-dir',   default='data/chunks',
                        help='Directory for output JSONL files.')
    parser.add_argument('--law-name',  default='',
                        help='Human-readable law name (overrides value in JSON if set).')
    parser.add_argument('--window',    type=int, default=256,
                        help='Token window size for fixed chunking (default 256).')
    parser.add_argument('--stride',    type=int, default=50,
                        help='Token stride for fixed chunking (default 50).')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    # ── Load inputs ──────────────────────────────────────────────────────────
    json_law_name, articles = load_articles(args.articles)
    law_name = args.law_name or json_law_name
    clean_text = load_clean_text(args.clean_txt)

    print(f'[chunk] Law: "{law_name}"')
    print(f'[chunk] Loaded {len(articles)} articles from {args.articles}')
    print(f'[chunk] Clean text: {len(word_tokenize(clean_text)):,} tokens')

    # ── Strategy A ───────────────────────────────────────────────────────────
    art_chunks = make_article_chunks(articles, law_name)
    art_path   = out_dir / 'article_chunks.jsonl'
    write_jsonl(art_chunks, art_path)
    print_stats('article', art_chunks)
    print(f'[chunk:article] Wrote → {art_path}')

    # ── Strategy B ───────────────────────────────────────────────────────────
    fix_chunks = make_fixed_chunks(clean_text, law_name,
                                   window=args.window, stride=args.stride)
    fix_path   = out_dir / 'fixed_chunks.jsonl'
    write_jsonl(fix_chunks, fix_path)
    print_stats('fixed', fix_chunks)
    print(f'[chunk:fixed]   Wrote → {fix_path}')


if __name__ == '__main__':
    main()
