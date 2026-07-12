"""
config.py
---------
Single source of truth for all experiment settings.
Edit this file before running experiments and do not hardcode paths elsewhere.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent   # C:/Work/nlp
DATA_DIR      = ROOT / "data"
CHUNKS_DIR    = DATA_DIR / "chunks"
RESULTS_DIR   = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Chunk files produced by chunk.py
ARTICLE_CHUNKS_PATH = CHUNKS_DIR / "article_chunks.jsonl"
FIXED_CHUNKS_PATH   = CHUNKS_DIR / "fixed_chunks.jsonl"

# Evaluation QA pairs
QA_CSV_PATH = ROOT / "evaluation" / "qa_pairs.csv"

# FAISS index cache (created on first run, reloaded after)
INDEX_CACHE_DIR = ROOT / "index_cache"
INDEX_CACHE_DIR.mkdir(exist_ok=True)

# ── LLM settings (Ollama) ────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"   # default Ollama endpoint
LLM_MODEL       = "qwen3.6"                  # ollama model name  (run: ollama pull qwen3.6)
LLM_TEMPERATURE = 0.1                        # low for factual legal Q&A
LLM_MAX_TOKENS  = 2048                       # must be high enough for Qwen3 thinking chain + answer
LLM_TIMEOUT_S   = 600                        # 10 min per question — Qwen3 thinking can be slow

# System prompt — tells the LLM to answer ONLY from retrieved context
SYSTEM_PROMPT_RAG = (
    "أنت مساعد قانوني متخصص في قانون العمل الأردني. "
    "أجب على السؤال بناءً فقط على النصوص القانونية المقدمة. "
    "إذا لم تجد الإجابة في النصوص، قل: 'لا تتوفر معلومات كافية في النصوص المقدمة'. "
    "كن دقيقاً ومختصراً."
)

# System prompt for the baseline (no retrieval)
SYSTEM_PROMPT_BASELINE = (
    "أنت مساعد قانوني متخصص في قانون العمل الأردني. "
    "أجب على السؤال بدقة واختصار بناءً على معرفتك بالقانون الأردني."
)

# ── Retrieval settings ───────────────────────────────────────────────────────
TOP_K = 5            # number of passages retrieved per query

# Chunking strategy to use for retrieval experiments
# Options: "article" | "fixed" | "both"
# "both" runs every retrieval system twice (once per strategy) — doubles runtime
CHUNK_STRATEGY = "both"

# ── Dense retrieval (AraBERT / sentence-transformers) ────────────────────────
# Any sentence-transformers compatible model that handles Arabic.
# Recommended options (all available on HuggingFace, ~200–400 MB each):
#   "intfloat/multilingual-e5-base"           ← best quality, needs "query: " prefix
#   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#   "aubmindlab/bert-base-arabertv02"          ← Arabic-specific
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
EMBEDDING_PREFIX_QUERY = "query: "      # required prefix for multilingual-e5
EMBEDDING_PREFIX_PASSAGE = "passage: "  # required prefix for multilingual-e5
# Set both to "" if using a model that does not require prefixes (e.g. AraBERT)

EMBEDDING_BATCH_SIZE = 32   # reduce to 16 if you run out of GPU memory
EMBEDDING_DEVICE     = "cuda"   # "cuda" or "cpu"

# ── Hybrid RRF settings ──────────────────────────────────────────────────────
RRF_K = 60   # RRF constant — standard value; higher → smoother fusion
