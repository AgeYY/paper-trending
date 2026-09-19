"""Local storage configuration; no external checkout or GPU dependency."""
import os

DATA_DIR = os.environ.get("PAPER_ATLAS_DATA_DIR", "./data")
