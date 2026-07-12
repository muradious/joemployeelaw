"""
retriever_bm25.py
-----------------
System 3 — Sparse retrieval using BM25 (rank_bm25 library).

BM25 is a strong baseline for legal retrieval because exact legal terminology
(e.g. "إشعار الفصل", "مكافأة نهاية الخدمة") is highly predictive of relevance.

Install:  pip install rank-bm25
"""

import re
import json
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from config import ARTICLE_CHUNKS_PATH, FIXED_CHUNKS_PATH, INDEX_CACHE_DIR, TOP_K


# ── Arabic tokeniser ─────────────────────────────────────────────────────────

def arabic_tokenize(text: str) -> list[str]:
    """
    Whitespace tokeniser for preprocessed Arabic text.

    Since preprocess.py already handles normalisation (hamza, tashkeel, etc.),
    simple whitespace splitting is sufficient here.  Punctuation is stripped
    so that "العمل،" and "العمل" are treated as the same token.
    """
    # Remove Arabic/Latin punctuation
    text = re.sub(r'[،؛؟!.\-،()«»""\']+', ' ', text)
    tokens = text.split()
    # Filter very short tokens (single chars, stray digits)
    return [t for t in tokens if len(t) > 1]


# ── BM25 Retriever ───────────────────────────────────────────────────────────

class BM25Retriever:
    """
    Loads chunks from a JSONL file, builds a BM25Okapi index, and retrieves
    the top-k most relevant chunks for a given Arabic query.

    Parameters
    ----------
    chunk_strategy : "article" | "fixed"
        Which chunking strategy's JSONL to load.
    top_k : int
        Number of passages to return per query.
    """

    def __init__(self, chunk_strategy: str = "article", top_k: int = TOP_K):
        if chunk_strategy not in ("article", "fixed"):
            raise ValueError("chunk_strategy must be 'article' or 'fixed'")

        self.chunk_strategy = chunk_strategy
        self.top_k = top_k
        self.chunks: list[dict] = []
        self.bm25: Optional[BM25Okapi] = None

        self._load_or_build()

    # ── index build / cache ───────────────────────────────────────────────────

    def _chunks_path(self) -> Path:
        return ARTICLE_CHUNKS_PATH if self.chunk_strategy == "article" \
               else FIXED_CHUNKS_PATH

    def _cache_path(self) -> Path:
        return INDEX_CACHE_DIR / f"bm25_{self.chunk_strategy}.pkl"

    def _load_or_build(self) -> None:
        """Load BM25 index from cache, or build + save it if absent."""
        chunks_path = self._chunks_path()
        cache_path  = self._cache_path()

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunks_path}\n"
                "Run scripts/chunk.py first."
            )

        # Load chunks
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Use cache if it exists and is newer than the chunk file
        if cache_path.exists() and \
           cache_path.stat().st_mtime >= chunks_path.stat().st_mtime:
            print(f"[bm25:{self.chunk_strategy}] Loading index from cache …")
            with cache_path.open("rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self._build_index(cache_path)

    def _build_index(self, cache_path: Path) -> None:
        print(
            f"[bm25:{self.chunk_strategy}] Building BM25 index "
            f"over {len(self.chunks)} chunks …"
        )
        tokenised = [arabic_tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenised)

        with cache_path.open("wb") as f:
            pickle.dump(self.bm25, f)
        print(f"[bm25:{self.chunk_strategy}] Index saved → {cache_path}")

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve the top-k most relevant chunks for *query*.

        Returns
        -------
        list of dicts, each containing:
            chunk_id, text, score, rank  (and all original chunk fields)
        """
        k = top_k or self.top_k
        query_tokens = arabic_tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Rank by score descending, take top-k
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        results = []
        for rank, idx in enumerate(ranked_indices):
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["rank"]  = rank + 1
            results.append(chunk)

        return results

    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None
                             ) -> tuple[list[dict], list[float]]:
        """Same as retrieve() but also returns raw scores for RRF fusion."""
        results = self.retrieve(query, top_k=top_k or len(self.chunks))
        scores  = [r["score"] for r in results]
        return results, scores

    def rebuild_index(self) -> None:
        """Force a full index rebuild (e.g. after adding new chunks)."""
        self._build_index(self._cache_path())


if __name__ == "__main__":
    # Quick smoke test
    for strategy in ("article", "fixed"):
        print(f"\n── BM25 {strategy} ──")
        try:
            ret = BM25Retriever(chunk_strategy=strategy)
            results = ret.retrieve("ما هي مدة إشعار الفصل من العمل؟")
            for r in results:
                print(f"  rank={r['rank']}  score={r['score']:.3f}  "
                      f"id={r['chunk_id']}  text={r['text'][:60]}…")
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
