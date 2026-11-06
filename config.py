"""
config.py  --  ALL SETTINGS LIVE HERE
Change a number here and every other file picks it up.
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"                 # where your 4 source files live
INDEX_DIR = ROOT / "index"               # where the searchable index is saved
VISUALS_FILE = ROOT / "visuals.json"     # text descriptions of charts/images

# --- Extraction settings (step1) ---
# Every sample file ends with its own "Taxonomy included in this file" table.
# That is metadata, not content, so we stop reading when we reach it.
STOP_HEADING = "taxonomy included in this file"
SKIP_PREFIXES = ("note:",)               # boilerplate lines to ignore

# --- Chunking settings (step2) ---
MAX_CHUNK_CHARS = 800                    # split very long text blocks

# --- Embedding settings (embedder.py) ---
EMBED_BACKEND = "auto"                   # "auto" | "sbert" | "tfidf"
SBERT_MODEL = "all-MiniLM-L6-v2"         # small, fast, free, runs locally

# --- Retrieval settings (step4) ---
TOP_K = 4                                # how many chunks to fetch per question

# --- Answer generation settings (step5) ---
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
MAX_ANSWER_TOKENS = 500
