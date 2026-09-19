#!/usr/bin/env python
"""Thin entrypoint for the interactive OpenReview diagnostic."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper_atlas.openreview_auth_check import main

if __name__ == "__main__":
    raise SystemExit(main())
