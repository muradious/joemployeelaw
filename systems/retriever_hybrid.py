"""
retriever_hybrid.py
-------------------
System 5 — Hybrid retrieval via Reciprocal Rank Fusion (RRF).

RRF fuses the ranked lists from BM25 (sparse) and dense retrieval using:

    RRF_score(d) = Σ  1 / (k + rank_i(d))
                  retrievers

where k is a smoothing constant (default 60, shown to work well across tasks).
RRF is parameter-free beyond k, robust to score-scale differences between
retrievers, and consistently outperforms individual retrievers.

Reference: Cormack, Clarke & Buettcher (SIGIR 2009).
"""

from typing import Optional

from retriever_bm25  import BM25Retriever
from retriever_dense import DenseRetriever
from config import INDEX_CACHE_DIR, TOP_K, RRF_K

import json
from pathlib import Path


class HybridRetriever:
    """
    Combines BM25 and dense retrieval using Reciprocal Rank Fusion.

    Parameters
    ----------
    chunk_strategy : "article" | "fixed"
    top_k          : final passages to return after fusion
    rrf_k          : RRF smoothing constant (default 60)
    """

    def __init__(
        self,
        chunk_strategy: str = "article",
        top_k: int = TOP_K,
        rrf_k: int = RRF_K,
    ):
        self.chunk_strategy = chunk_strategy
        self.top_k = top_k
        self.rrf_k = rrf_k

        print(f"[hybrid:{chunk_strategy}] Initialising BM25 retriever …")
        self.bm25  = BM25Retriever(chunk_strategy=chunk_strategy,
                                    top_k=None)   # type: ignore  (we use all ranks)

        print(f"[hybrid:{chunk_strategy}] Initialising dense retriever …")
        self.dense = DenseRetriever(chunk_strategy=chunk_strategy,
                                     top_k=None)  # type: ignore

        # Build a lookup from chunk_id → chunk dict for the final results
        self._chunk_lookup: dict[str, dict] = {
            c["chunk_id"]: c for c in self.dense.chunks
        }

    # ── RRF core ──────────────────────────────────────────────────────────────

    @staticmethod
    def _rrf_scores(
        ranked_lists: list[list[str]],
        k: int = 60,
    ) -> dict[str, float]:
        """
        Compute RRF score for each document across all ranked lists.

        Parameters
        ----------
        ranked_lists : list of lists of chunk_ids, each sorted best-first
        k            : RRF smoothing constant

        Returns
        -------
        dict mapping chunk_id → RRF score (higher is better)
        """
        scores: dict[str, float] = {}
        for ranked in ranked_lists:
            for rank_0based, chunk_id in enumerate(ranked):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + \
                                   1.0 / (k + rank_0based + 1)
        return scores

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k chunks via RRF fusion of BM25 + dense ranked lists.

        Returns
        -------
        list of dicts with: chunk_id, text, rrf_score, bm25_rank,
        dense_rank, rank, and all original chunk metadata.
        """
        k = top_k or self.top_k
        n = len(self._chunk_lookup)   # use full ranked lists for RRF

        # Get full ranked lists from each retriever
        bm25_results  = self.bm25.retrieve(query, top_k=n)
        dense_results = self.dense.retrieve(query, top_k=n)

        bm25_ranked  = [r["chunk_id"] for r in bm25_results]
        dense_ranked = [r["chunk_id"] for r in dense_results]

        # Build rank lookup for metadata
        bm25_rank_map  = {r["chunk_id"]: r["rank"] for r in bm25_results}
        dense_rank_map = {r["chunk_id"] for r in dense_results}
        dense_rank_map = {r["chunk_id"]: r["rank"] for r in dense_results}

        # Compute RRF scores
        rrf = self._rrf_scores([bm25_ranked, dense_ranked], k=self.rrf_k)

        # Sort by RRF score, take top-k
        top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:k]

        results = []
        for final_rank, chunk_id in enumerate(top_ids, start=1):
            chunk = dict(self._chunk_lookup.get(chunk_id, {"chunk_id": chunk_id,
                                                             "text": ""}))
            chunk["rrf_score"]  = round(rrf[chunk_id], 6)
            chunk["bm25_rank"]  = bm25_rank_map.get(chunk_id, n + 1)
            chunk["dense_rank"] = dense_rank_map.get(chunk_id, n + 1)
            chunk["rank"]       = final_rank
            chunk["score"]      = chunk["rrf_score"]   # uniform interface
            results.append(chunk)

        return results


if __name__ == "__main__":
    for strategy in ("article", "fixed"):
        print(f"\n── Hybrid {strategy} ──")
        try:
            ret = HybridRetriever(chunk_strategy=strategy)
            results = ret.retrieve("ما هي مدة إشعار الفصل من العمل؟")
            for r in results:
                print(
                    f"  rank={r['rank']}  rrf={r['rrf_score']:.5f}  "
                    f"bm25_r={r['bm25_rank']}  dense_r={r['dense_rank']}  "
                    f"text={r['text'][:55]}…"
                )
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
