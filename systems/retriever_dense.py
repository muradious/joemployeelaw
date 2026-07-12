"""
retriever_dense.py
------------------
System 4 — Dense retrieval using multilingual-E5 embeddings + FAISS.

Encodes all chunks into dense vectors using a sentence-transformers model,
stores them in a FAISS flat-IP index, and retrieves by cosine similarity
at query time.  The index is cached to disk so it only needs to be built once.

Install:
    pip install sentence-transformers faiss-cpu
    # GPU FAISS (optional, faster on your RTX 4060):
    # pip install faiss-gpu
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import faiss
from sentence_transformers import SentenceTransformer

from config import (
    ARTICLE_CHUNKS_PATH,
    FIXED_CHUNKS_PATH,
    INDEX_CACHE_DIR,
    TOP_K,
    EMBEDDING_MODEL,
    EMBEDDING_PREFIX_QUERY,
    EMBEDDING_PREFIX_PASSAGE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
)


class DenseRetriever:
    """
    Dense retriever: encodes passages with a sentence-transformer model,
    indexes them with FAISS (cosine similarity via inner-product on L2-normalised
    vectors), and retrieves the top-k nearest neighbours at query time.

    Parameters
    ----------
    chunk_strategy  : "article" | "fixed"
    top_k           : passages returned per query
    embedding_model : HuggingFace model name (overrides config default)
    device          : "cuda" | "cpu"
    """

    def __init__(
        self,
        chunk_strategy: str = "article",
        top_k: int = TOP_K,
        embedding_model: str = EMBEDDING_MODEL,
        device: str = EMBEDDING_DEVICE,
    ):
        if chunk_strategy not in ("article", "fixed"):
            raise ValueError("chunk_strategy must be 'article' or 'fixed'")

        self.chunk_strategy  = chunk_strategy
        self.top_k           = top_k
        self.embedding_model = embedding_model
        self.device          = device

        self.chunks: list[dict] = []
        self.index:  Optional[faiss.Index] = None

        print(f"[dense:{chunk_strategy}] Loading embedding model: {embedding_model} …")
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.dim = self.encoder.get_sentence_embedding_dimension()

        self._load_or_build()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _chunks_path(self) -> Path:
        return ARTICLE_CHUNKS_PATH if self.chunk_strategy == "article" \
               else FIXED_CHUNKS_PATH

    def _cache_prefix(self) -> Path:
        model_slug = self.embedding_model.replace("/", "_")
        return INDEX_CACHE_DIR / f"dense_{self.chunk_strategy}_{model_slug}"

    def _encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Encode texts, optionally prepending a query/passage prefix."""
        if prefix:
            texts = [prefix + t for t in texts]
        vecs = self.encoder.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2 normalise → inner-product = cosine
        )
        return vecs.astype(np.float32)

    # ── index build / cache ───────────────────────────────────────────────────

    def _load_or_build(self) -> None:
        chunks_path  = self._chunks_path()
        faiss_path   = Path(str(self._cache_prefix()) + ".faiss")
        chunks_cache = Path(str(self._cache_prefix()) + "_chunks.pkl")

        if not chunks_path.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {chunks_path}\n"
                "Run scripts/chunk.py first."
            )

        # Load raw chunks
        self.chunks = [
            json.loads(line)
            for line in chunks_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        # Use cache if it is newer than the source chunks
        if faiss_path.exists() and chunks_cache.exists() and \
           faiss_path.stat().st_mtime >= chunks_path.stat().st_mtime:
            print(f"[dense:{self.chunk_strategy}] Loading FAISS index from cache …")
            self.index = faiss.read_index(str(faiss_path))
            with chunks_cache.open("rb") as f:
                self.chunks = pickle.load(f)
        else:
            self._build_index(faiss_path, chunks_cache)

    def _build_index(self, faiss_path: Path, chunks_cache: Path) -> None:
        print(
            f"[dense:{self.chunk_strategy}] Encoding {len(self.chunks)} chunks …"
        )
        texts = [c["text"] for c in self.chunks]
        vecs  = self._encode(texts, prefix=EMBEDDING_PREFIX_PASSAGE)

        # Flat inner-product index (exact search; fine for <10k chunks)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

        faiss.write_index(self.index, str(faiss_path))
        with chunks_cache.open("wb") as f:
            pickle.dump(self.chunks, f)

        print(
            f"[dense:{self.chunk_strategy}] FAISS index saved "
            f"({self.index.ntotal} vectors) → {faiss_path}"
        )

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """
        Retrieve top-k chunks most similar to *query*.

        Returns
        -------
        list of dicts with fields: chunk_id, text, score (cosine sim), rank,
        plus all original chunk metadata.
        """
        k = top_k or self.top_k
        q_vec = self._encode([query], prefix=EMBEDDING_PREFIX_QUERY)

        scores, indices = self.index.search(q_vec, k)
        scores  = scores[0].tolist()
        indices = indices[0].tolist()

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores)):
            if idx == -1:        # FAISS returns -1 for padding
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["rank"]  = rank + 1
            results.append(chunk)

        return results

    def retrieve_ranked_ids(self, query: str, n: Optional[int] = None
                            ) -> list[str]:
        """
        Return chunk_ids in ranked order (for RRF fusion in retriever_hybrid.py).
        n defaults to all chunks so the hybrid retriever has full ranked lists.
        """
        n = n or len(self.chunks)
        results = self.retrieve(query, top_k=n)
        return [r["chunk_id"] for r in results]

    def rebuild_index(self) -> None:
        """Force a full index rebuild."""
        p = self._cache_prefix()
        self._build_index(
            Path(str(p) + ".faiss"),
            Path(str(p) + "_chunks.pkl"),
        )


if __name__ == "__main__":
    for strategy in ("article", "fixed"):
        print(f"\n── Dense {strategy} ──")
        try:
            ret = DenseRetriever(chunk_strategy=strategy)
            results = ret.retrieve("ما هي مدة إشعار الفصل من العمل؟")
            for r in results:
                print(f"  rank={r['rank']}  score={r['score']:.4f}  "
                      f"id={r['chunk_id']}  text={r['text'][:60]}…")
        except FileNotFoundError as e:
            print(f"  Skipped: {e}")
