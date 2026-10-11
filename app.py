"""Streamlit Cloud entry point for Athena RAG."""

import sys
from pathlib import Path

# Ensure 'src' directory is in Python path for Streamlit Cloud deployment
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from athena.ui import main

main()
