"""Create a fresh, audited public source tree without copying Git history or secrets."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
import shutil

ROOT_FILES = ("README.md", "pyproject.toml", "LICENSE", "SECURITY.md", ".gitignore")
DOCS = ("conference_survey.md", "conference_web.md", "DATA.md", "STATIC_DEMO.md")
SUFFIXES = {".py", ".json", ".html", ".css", ".js", ".mjs", ".cjs"}
# Report file names and categories only, never matching secret text.
PATTERNS = {
    "api_token": re.compile(r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-)?[A-Za-z0-9_-]{32,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})(?![A-Za-z0-9_-])"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "workstation_path": re.compile(r"/(?:home|Users)/[A-Za-z][^\s\"<>]*"),
}


def audit(root):
    root = Path(root)
    issues = []
    for file in root.rglob("*"):
        relative = file.relative_to(root)
        if ".git" in relative.parts or not file.is_file():
            continue
        if file.is_symlink():
            issues.append((str(relative), "symlink"))
            continue
        if file.name.endswith((".env", ".sqlite3", ".log", ".swp", ".pem", ".key")) or file.name.startswith(".env"):
            issues.append((str(relative), "private_file_type"))
        raw = file.read_bytes()
        if file.name.endswith(".gz"):
            raw = gzip.decompress(raw)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            issues.append((str(relative), "unexpected_binary"))
            continue
        for category, pattern in PATTERNS.items():
            if pattern.search(text):
                issues.append((str(relative), category))
    if issues:
        raise ValueError("Public export audit failed: " + json.dumps(issues))
    return {"files": sum(f.is_file() for f in root.rglob("*") if ".git" not in f.relative_to(root).parts), "status": "passed"}


def prepare(source, site, destination):
    source, site, destination = map(Path, (source, site, destination))
    if destination.exists():
        raise ValueError("Destination already exists; use a fresh public checkout.")
    files = [(source / name, Path(name)) for name in ROOT_FILES]
    files += [(source / "docs" / name, Path("docs") / name) for name in DOCS]
    for folder in ("paper_atlas", "tests", "bin"):
        files += [(f, f.relative_to(source)) for f in (source / folder).rglob("*")
                  if f.is_file() and f.suffix in SUFFIXES and "__pycache__" not in f.parts]
    files += [(f, Path("site") / f.relative_to(site)) for f in site.rglob("*") if f.is_file()]
    workflow = source / "public_workflows" / "pages.yml"
    if not workflow.exists():
        workflow = source / ".github" / "workflows" / "pages.yml"
    files.append((workflow, Path(".github/workflows/pages.yml")))
    # Resolve all selected files before writing; no symlink traversal into secrets.
    for src, _ in files:
        if src.is_symlink() or any(p.is_symlink() for p in src.parents):
            raise ValueError("Symlink in release input.")
    destination.mkdir(parents=True)
    for src, relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    return audit(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--site", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args(argv)
    if args.audit:
        result = audit(args.audit)
    elif args.site and args.destination:
        result = prepare(args.source, args.site, args.destination)
    else:
        parser.error("Use --audit DIR or --site DIR --destination NEW_DIR.")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
