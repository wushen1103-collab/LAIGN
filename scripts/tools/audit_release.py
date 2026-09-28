#!/usr/bin/env python3
"""Fail when a release contains local paths, personal metadata or large files."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SELF = Path(__file__).resolve()
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
MAX_FILE_BYTES = 50 * 1024 * 1024
PATTERNS = {
    "Unix home path": re.compile(r"/(?:home|Users|root)/"),
    "Windows absolute path": re.compile(r"(?i)\b[A-Z]:[\\/]"),
    "private-network host": re.compile(r"\b10\.110\.3\.71\b|\b7006\b"),
    "personal email": re.compile(r"(?i)\b[\w.+-]+@hunnu\.edu\.cn\b"),
    "personal name": re.compile(
        r"(?i)\b(?:shenkuang|shankuang|pingping|lianming)\b|wushen1103"
    ),
    "internal experiment label": re.compile(r"(?<![A-Za-z0-9])[Hh][0-9]{2}(?![A-Za-z0-9])"),
}


def main() -> int:
    failures: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == SELF or SKIP_PARTS.intersection(path.parts):
            continue
        relative = path.relative_to(ROOT)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            failures.append(f"large file ({size} bytes): {relative}")
        if path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {relative}:{line}")
    if failures:
        print("Release audit failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
