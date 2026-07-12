@echo off
echo ============================================================
echo  NLP Project - Installing Python packages (Python 3.13 fix)
echo ============================================================
echo.

echo [1/3] Installing PyTorch for Python 3.13 + RTX 4060...
echo        Trying CUDA 12.8 wheels (latest, supports Python 3.13)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if %errorlevel% neq 0 (
    echo        cu128 failed, trying CUDA 12.4...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)
if %errorlevel% neq 0 (
    echo        cu124 failed, trying PyPI directly...
    pip install torch torchvision torchaudio
)
if %errorlevel% neq 0 (
    echo [FAIL] Could not install PyTorch. See note at the bottom.
    goto packages
)

:packages
echo.
echo [2/3] Installing retrieval and evaluation packages...
pip install rank-bm25 sentence-transformers faiss-cpu bert-score numpy
if %errorlevel% neq 0 (
    echo [FAIL] One or more packages failed. Check errors above.
    pause
    exit /b 1
)

echo.
echo [3/3] Verifying installs...
python -c "import torch; print('  torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available()); print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" 2>nul || echo   [FAIL] torch not working
python -c "import sentence_transformers; print('  sentence-transformers:', sentence_transformers.__version__)" 2>nul || echo   [FAIL] sentence-transformers
python -c "import rank_bm25; print('  rank-bm25: OK')" 2>nul || echo   [FAIL] rank-bm25
python -c "import faiss; print('  faiss: OK')" 2>nul || echo   [FAIL] faiss
python -c "import bert_score; print('  bert-score: OK')" 2>nul || echo   [FAIL] bert-score
python -c "import numpy; print('  numpy:', numpy.__version__)" 2>nul || echo   [FAIL] numpy

echo.
echo ============================================================
echo  Next steps:
echo  1. Install Ollama: https://ollama.com (Download for Windows)
echo  2. Open a terminal and run:  ollama serve
echo  3. Open another terminal:    ollama pull qwen2.5:7b
echo  4. Run:  python preflight.py
echo ============================================================
echo.
echo  NOTE: If torch shows "CUDA available: False" after install,
echo  run this command manually and re-run this script:
echo  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
echo.
pause
