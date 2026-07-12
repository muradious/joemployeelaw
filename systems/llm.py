"""
llm.py
------
Thin wrapper around the Ollama HTTP API.

Ollama must be running locally:  ollama serve
The model must be pulled first:  ollama pull jais

If Jais is not yet available on Ollama Hub, substitute with any Arabic-capable
model (e.g. `ollama pull aya:8b` or `ollama pull qwen2.5:7b`) and update
LLM_MODEL in config.py.
"""

import json
import requests
from typing import Optional
from config import (
    OLLAMA_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT_S,
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_BASELINE,
)


class OllamaLLM:
    """
    Minimal Ollama client using the /api/chat endpoint.

    Parameters
    ----------
    model       : Ollama model tag (e.g. "jais", "aya:8b")
    temperature : Sampling temperature (0.0 = deterministic)
    max_tokens  : Maximum new tokens to generate
    base_url    : Ollama server URL
    """

    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.url         = f"{base_url.rstrip('/')}/api/chat"

    def _call(self, messages: list[dict]) -> str:
        """Send messages to Ollama and return the assistant reply."""
        payload = {
            "model"   : self.model,
            "messages": messages,
            "stream"  : False,
            "think"   : False,     # disable Qwen3 thinking mode — answers directly without <think> chain
            "options" : {
                "temperature" : self.temperature,
                "num_predict" : self.max_tokens,
            },
        }
        try:
            resp = requests.post(self.url, json=payload, timeout=LLM_TIMEOUT_S)
            resp.raise_for_status()
            data = resp.json()
            raw = data["message"]["content"].strip()
            # Strip any residual <think>...</think> block Qwen3 may still emit
            import re
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            return raw
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach Ollama. Make sure it is running: `ollama serve`"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}\n{resp.text}")

    def answer_baseline(self, question: str) -> str:
        """
        System 1 — LLM-only baseline: answer with no retrieved context.
        """
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT_BASELINE},
            {"role": "user",    "content": question},
        ]
        return self._call(messages)

    def answer_with_context(self, question: str, passages: list[str]) -> str:
        """
        Systems 2-4 — RAG answer: inject retrieved passages as context.

        Parameters
        ----------
        question : Arabic question string
        passages : List of retrieved passage texts (ordered best-first)
        """
        context_block = "\n\n".join(
            f"[نص {i+1}]\n{p}" for i, p in enumerate(passages)
        )
        user_content = (
            f"النصوص القانونية المسترجعة:\n\n{context_block}\n\n"
            f"السؤال: {question}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_RAG},
            {"role": "user",   "content": user_content},
        ]
        return self._call(messages)

    def is_available(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        try:
            resp = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags", timeout=5
            )
            tags = [m["name"] for m in resp.json().get("models", [])]
            return any(self.model in t for t in tags)
        except Exception:
            return False


def check_ollama():
    """Print a friendly status message about Ollama availability."""
    llm = OllamaLLM()
    if llm.is_available():
        print(f"[llm] ✓ Ollama is running — model '{LLM_MODEL}' is ready.")
    else:
        print(
            f"[llm] ✗ Model '{LLM_MODEL}' not found in Ollama.\n"
            f"      Run:  ollama pull {LLM_MODEL}\n"
            f"      Or update LLM_MODEL in config.py to a model you have."
        )


if __name__ == "__main__":
    check_ollama()
