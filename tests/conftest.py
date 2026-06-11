"""
conftest.py — pytest configuration for miniRAG tests.

Ensures src/minirag is at the front of sys.path so tests import the
new package version, not the legacy flat files in the repo root.
"""

import sys
from pathlib import Path

# src/minirag must come BEFORE repo root so the new pydantic-settings
# Config is used instead of the old streamlit-coupled one.
_SRC = Path(__file__).resolve().parent.parent / "src" / "minirag"
_ROOT = Path(__file__).resolve().parent.parent

# Insert src/minirag at position 0 so it wins over repo root
if str(_SRC) in sys.path:
    sys.path.remove(str(_SRC))
sys.path.insert(0, str(_SRC))

# Keep repo root for any remaining flat-file imports (embeddings, llm, etc.)
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
