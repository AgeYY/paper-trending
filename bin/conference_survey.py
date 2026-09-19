#!/usr/bin/env python
"""Thin entrypoint for the reusable conference survey module."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_atlas.conference_survey import main

if __name__ == "__main__":
    raise SystemExit(main())
