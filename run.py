"""
run.py
Entry point — starts the FastAPI server with uvicorn.
Usage:  python run.py
"""
import sys
from pathlib import Path

# Ensure src/ and api/ are importable regardless of working directory
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "api"))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(ROOT / "api"), str(ROOT / "src")],
        log_level="info",
    )
