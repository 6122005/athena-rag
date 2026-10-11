"""Streamlit Cloud entry point for Athena RAG."""

import sys
from pathlib import Path

# Fix SQLite3 on Streamlit Cloud Linux containers if pysqlite3 is available
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

# Ensure 'src' directory is in Python path for Streamlit Cloud deployment
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from athena.ui import main

main()

