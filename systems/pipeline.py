"""
pipeline.py
-----------
Unified interface for all four experimental systems.

Each system is a (retriever, llm) pair with a common .run(question) method.
The four systems are:

  System 1 — Baseline   : no retriever, LLM answers from memory
  System 2 — BM25-RAG   : sparse BM25 retriever + LLM
  System 3 — Dense-RAG  : AraBERT/E5 + FAISS retriever + LLM
  System 4 — Hybrid-RAG : RRF(BM25 + Dense) retriever + LLM

Each system is instantiated for each chunking strategy (article / fixed),
giving up to 7 configurations in total (baseline has no chunking strategy).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from llm              import OllamaLLM
from retriever_bm25   import BM25Retriever
from retriever_dense  import DenseRetriever
from retriever_hybrid import HybridRetriever
from config           import TOP_K


@dataclass
class RunResult:
    """Holds everything produced for one question by one system."""
    system_name    : str
    chunk_strategy : str          # "none" for baseline
    question       : str
    answer         : str
    retrieved      : list[dict]   # empty for baseline
    latency_s      : float        # wall-clock seconds


class BaselineSystem:
    """System 1 — LLM only, no retrieval."""

    name           = "baseline"
    chunk_strategy = "none"

    def __init__(self, llm: Optional[OllamaLLM] = None):
        self.llm = llm or OllamaLLM()

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()
        answer = self.llm.answer_baseline(question)
        return RunResult(
            system_name    = self.name,
            chunk_strategy = self.chunk_strategy,
            question       = question,
            answer         = answer,
            retrieved      = [],
            latency_s      = time.perf_counter() - t0,
        )


class RAGSystem:
    """
    Generic RAG system: retrieve passages → inject into LLM prompt.

    Parameters
    ----------
    retriever      : BM25Retriever | DenseRetriever | HybridRetriever
    system_name    : human-readable label for results tables
    top_k          : passages to retrieve (overrides retriever default if set)
    llm            : shared OllamaLLM instance (creates a new one if not given)
    """

    def __init__(
        self,
        retriever,
        system_name: str,
        top_k: int = TOP_K,
        llm: Optional[OllamaLLM] = None,
    ):
        self.retriever      = retriever
        self.name           = system_name
        self.chunk_strategy = retriever.chunk_strategy
        self.top_k          = top_k
        self.llm            = llm or OllamaLLM()

    def run(self, question: str) -> RunResult:
        t0 = time.perf_counter()

        passages = self.retriever.retrieve(question, top_k=self.top_k)
        texts    = [p["text"] for p in passages]
        answer   = self.llm.answer_with_context(question, texts)

        return RunResult(
            system_name    = self.name,
            chunk_strategy = self.chunk_strategy,
            question       = question,
            answer         = answer,
            retrieved      = passages,
            latency_s      = time.perf_counter() - t0,
        )


def build_all_systems(
    chunk_strategies: list[str] = ("article", "fixed"),
    top_k: int = TOP_K,
) -> list[BaselineSystem | RAGSystem]:
    """
    Instantiate every system that will be evaluated.

    Returns
    -------
    list of system objects, each with a .run(question) method.

    System list:
        baseline
        bm25_article,  bm25_fixed
        dense_article, dense_fixed
        hybrid_article, hybrid_fixed
    """
    llm = OllamaLLM()   # one shared LLM instance for all systems

    systems: list[BaselineSystem | RAGSystem] = [BaselineSystem(llm=llm)]

    for strategy in chunk_strategies:
        # BM25
        systems.append(RAGSystem(
            retriever   = BM25Retriever(chunk_strategy=strategy, top_k=top_k),
            system_name = f"bm25_{strategy}",
            top_k       = top_k,
            llm         = llm,
        ))
        # Dense
        systems.append(RAGSystem(
            retriever   = DenseRetriever(chunk_strategy=strategy, top_k=top_k),
            system_name = f"dense_{strategy}",
            top_k       = top_k,
            llm         = llm,
        ))
        # Hybrid
        systems.append(RAGSystem(
            retriever   = HybridRetriever(chunk_strategy=strategy, top_k=top_k),
            system_name = f"hybrid_{strategy}",
            top_k       = top_k,
            llm         = llm,
        ))

    return systems
