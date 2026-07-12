"""
preflight.py
------------
Pre-flight check for the NLP RAG project.
Run this before run_experiments.py to catch any missing pieces.

Usage (from C:/Work/nlp):
    python preflight.py

Checks:
    1. Python version
    2. Required Python packages  (rank-bm25, sentence-transformers, faiss, bert-score)
    3. Ollama installed + running
    4. LLM model available in Ollama
    5. Chunk files exist and are non-empty
    6. Evaluation CSV has enough filled questions
    7. GPU availability (optional but recommended)
"""

import sys
import json
import subprocess
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
INFO  = "[INFO]"

failures = []
warnings = []

def ok(msg):
    print(f"  {PASS}  {msg}")

def fail(msg, fix=None):
    print(f"  {FAIL}  {msg}")
    if fix:
        print(f"          FIX: {fix}")
    failures.append(msg)

def warn(msg, fix=None):
    print(f"  {WARN}  {msg}")
    if fix:
        print(f"          FIX: {fix}")
    warnings.append(msg)

def info(msg):
    print(f"  {INFO}  {msg}")

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ── 1. Python version ─────────────────────────────────────────────────────────

section("1. Python Version")
v = sys.version_info
if v.major == 3 and v.minor >= 9:
    ok(f"Python {v.major}.{v.minor}.{v.micro}")
else:
    fail(
        f"Python {v.major}.{v.minor}.{v.micro} — need 3.9+",
        fix="Download Python 3.11+ from python.org"
    )


# ── 2. Python packages ────────────────────────────────────────────────────────

section("2. Python Packages")

REQUIRED_PACKAGES = {
    "rank_bm25"           : ("rank-bm25",            "pip install rank-bm25"),
    "sentence_transformers": ("sentence-transformers", "pip install sentence-transformers"),
    "faiss"               : ("faiss-cpu",             "pip install faiss-cpu"),
    "bert_score"          : ("bert-score",            "pip install bert-score"),
    "requests"            : ("requests",              "pip install requests"),
    "numpy"               : ("numpy",                 "pip install numpy"),
    "torch"               : ("torch",                 "pip install torch --index-url https://download.pytorch.org/whl/cu121"),
}

for import_name, (pkg_name, install_cmd) in REQUIRED_PACKAGES.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, "__version__", "?")
        ok(f"{pkg_name} {version}")
    except ImportError:
        fail(f"{pkg_name} not installed", fix=install_cmd)


# ── 3. Ollama installed ───────────────────────────────────────────────────────

section("3. Ollama Installation")

ollama_installed = False
try:
    result = subprocess.run(
        ["ollama", "--version"],
        capture_output=True, text=True, timeout=10
    )
    if result.returncode == 0:
        ok(f"Ollama installed: {result.stdout.strip()}")
        ollama_installed = True
    else:
        fail(
            "Ollama command failed",
            fix="Download from https://ollama.com and run the installer"
        )
except FileNotFoundError:
    fail(
        "Ollama not found on PATH",
        fix="Download from https://ollama.com  —  then restart your terminal"
    )
except Exception as e:
    fail(f"Ollama check error: {e}")


# ── 4. Ollama service + model ─────────────────────────────────────────────────

section("4. Ollama Service & Model")

import requests as req

OLLAMA_URL  = "http://localhost:11434"

# Read model name from config if possible
try:
    sys.path.insert(0, str(Path(__file__).parent / "systems"))
    from config import LLM_MODEL
except Exception:
    LLM_MODEL = "jais"

ollama_running = False
try:
    resp = req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    resp.raise_for_status()
    models = resp.json().get("models", [])
    model_names = [m["name"] for m in models]
    ok(f"Ollama is running at {OLLAMA_URL}")
    ollama_running = True

    if any(LLM_MODEL in name for name in model_names):
        ok(f"Model '{LLM_MODEL}' is available")
    else:
        warn(
            f"Model '{LLM_MODEL}' not found in Ollama",
            fix=f"Run:  ollama pull {LLM_MODEL}"
        )
        if model_names:
            info(f"Available models: {', '.join(model_names)}")
            info(f"To use a different model, edit systems/config.py → LLM_MODEL")

except req.exceptions.ConnectionError:
    fail(
        "Ollama service is not running",
        fix="Open a terminal and run:  ollama serve"
    )
except Exception as e:
    fail(f"Ollama service check failed: {e}")

# Quick generation test if Ollama is running
if ollama_running:
    try:
        test_resp = req.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": "قل مرحبا فقط"}],
                "stream": False,
                "options": {"num_predict": 10},
            },
            timeout=300,  # first call loads model into VRAM — can take 2-3 min
        )
        if test_resp.status_code == 200:
            reply = test_resp.json()["message"]["content"].strip()
            ok(f"LLM test generation succeeded: '{reply[:50]}'")
        else:
            warn(f"LLM test returned HTTP {test_resp.status_code}: {test_resp.text[:100]}")
    except Exception as e:
        warn(f"LLM test generation failed: {e}")


# ── 5. Data files ─────────────────────────────────────────────────────────────

section("5. Data Files")

ROOT       = Path(__file__).parent
CHUNKS_DIR = ROOT / "data" / "chunks"
ARTICLE_F  = CHUNKS_DIR / "article_chunks.jsonl"
FIXED_F    = CHUNKS_DIR / "fixed_chunks.jsonl"

for path, label, min_lines in [
    (ARTICLE_F, "article_chunks.jsonl", 10),
    (FIXED_F,   "fixed_chunks.jsonl",   10),
]:
    if not path.exists():
        fail(
            f"{label} missing",
            fix="Run:  python scripts/preprocess.py  then  python scripts/chunk.py"
        )
    else:
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if len(lines) >= min_lines:
            ok(f"{label}  ({len(lines)} chunks)")
        else:
            warn(f"{label} exists but only {len(lines)} chunks — re-run chunking?")


# ── 6. Evaluation CSV ─────────────────────────────────────────────────────────

section("6. Evaluation QA Set")

import csv

QA_CSV = ROOT / "evaluation" / "qa_pairs.csv"
if not QA_CSV.exists():
    fail("evaluation/qa_pairs.csv not found")
else:
    with QA_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    total   = len(rows)
    filled  = sum(1 for r in rows if r.get("question_ar", "").strip())
    answered = sum(1 for r in rows if r.get("reference_answer_ar", "").strip())

    if filled == total:
        ok(f"All {total} questions filled in")
    elif filled >= 60:
        warn(f"{filled}/{total} questions filled — {total - filled} still empty")
    else:
        fail(
            f"Only {filled}/{total} questions filled",
            fix="Fill in more questions in evaluation/qa_pairs.csv before running"
        )

    if answered < filled:
        warn(f"{answered}/{filled} questions have reference answers")


# ── 7. GPU ────────────────────────────────────────────────────────────────────

section("7. GPU (optional)")

try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        ok(f"CUDA GPU found: {gpu}  ({mem:.1f} GB VRAM)")
    else:
        warn(
            "No CUDA GPU detected — embeddings will run on CPU (slower)",
            fix="Edit systems/config.py → set EMBEDDING_DEVICE = 'cpu' to suppress this"
        )
except ImportError:
    warn("PyTorch not installed — cannot check GPU")


# ── Summary ───────────────────────────────────────────────────────────────────

section("Summary")

if not failures and not warnings:
    print("\n  Everything looks good — you're ready to run experiments!\n")
    print("  Next step:")
    print("    cd systems")
    print("    python run_experiments.py --limit 3   # quick sanity check first")
elif not failures:
    print(f"\n  {len(warnings)} warning(s) — experiments will likely work but review above.")
else:
    print(f"\n  {len(failures)} failure(s) must be fixed before running experiments:")
    for i, f in enumerate(failures, 1):
        print(f"    {i}. {f}")

if not ollama_installed or not ollama_running:
    print("""
  OLLAMA SETUP (if not yet installed):
  ─────────────────────────────────────
  1. Download:   https://ollama.com  (click Download for Windows)
  2. Install:    run the installer
  3. Start:      open a terminal and type:  ollama serve
  4. Pull model: in another terminal:       ollama pull qwen2.5:7b
     (Jais may not be on Ollama Hub yet — qwen2.5:7b is a strong Arabic alternative)
  5. Update:     set LLM_MODEL = "qwen2.5:7b" in systems/config.py
  6. Re-run:     python preflight.py
""")
