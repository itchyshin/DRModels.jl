#!/usr/bin/env python3
"""Reject internal work-tracking language from public documentation sources.

This is intentionally a narrow mechanical guard.  It does not decide whether a
scientific claim is useful or earned; Pat and Rose review that separately.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_ROOT = REPO_ROOT / "docs" / "src"
PATTERNS = {
    "internal_tracking": re.compile(
        r"(?:\b(?:PR|issue)\s*#\d+|\bArc\s+\d+\b|"
        r"\b(?:implementation|development|work|active)\s+lane\b|\bworktree\b|\bdev-log/|"
        r"\b(?:agent|persona)\s+(?:review|approved|approval|handoff)\b|"
        r"\b(?:Rose|Pat)\s+(?:reviewed|approved)\b|"
        r"\bfixture(?:-backed|\s+evidence)\b|\bcapability\s+ledger\b|"
        r"\b(?:catch-up\s+)?scoreboard\b|\boptimizer-health\b)",
        flags=re.IGNORECASE,
    ),
}
PUBLIC_SUFFIXES = {".md", ".qmd", ".rmd"}


def landing_contract_findings(root: Path) -> list[str]:
    """Return missing essentials from a reader's first DRModels page.

    This is intentionally a small floor, not a style scorer: it makes sure the
    landing page says what the model class is in ordinary words, establishes
    that Julia can be used directly, and sends a new reader to a runnable path.
    """
    landing = root / "index.md"
    if not landing.is_file():
        return ["index.md is missing"]
    text = landing.read_text(encoding="utf-8")
    checks = {
        "a plain definition of distributional regression": r"\bdistributional regression\b.*\b(?:average|mean)\b.*\b(?:variability|spread|variation)\b",
        "a standalone Julia identity": r"\bstandalone Julia\b",
        "a link to the runnable getting-started route": r"\]\((?:/)?getting-started(?:\.md)?\)",
    }
    return [label for label, pattern in checks.items() if not re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)]


def navigation_files(root: Path) -> list[Path]:
    """Return literal Markdown routes declared in the production navigation.

    This deliberately reads only string-literal Markdown paths: it never
    evaluates ``docs/make.jl``.  A non-literal, missing, or escaping route is
    a configuration error, not a reason to scan a smaller set of pages.
    """
    make = root.parent / "make.jl"
    if not make.is_file():
        raise ValueError(f"production navigation is absent: {make}")
    paths = re.findall(r'"([^"\\]+\.md)"', make.read_text(encoding="utf-8"))
    if not paths:
        raise ValueError(f"production navigation names no Markdown pages: {make}")
    selected: list[Path] = []
    for raw in paths:
        candidate = Path(raw)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"production navigation has an unsafe route: {raw}")
        page = root / candidate
        if not page.is_file():
            raise ValueError(f"production navigation names a missing page: {raw}")
        selected.append(page)
    return sorted(set(selected))


def public_files(root: Path, *, public_only: bool) -> list[Path]:
    if public_only:
        return navigation_files(root)
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in PUBLIC_SUFFIXES
    )


def findings(root: Path, *, public_only: bool) -> list[str]:
    problems: list[str] = []
    for path in public_files(root, public_only=public_only):
        lines = path.read_text(encoding="utf-8").splitlines()
        # Keep newline boundaries for useful source locations while allowing a
        # wrapped phrase such as "capability\nledger" to be recognised.
        text = "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                excerpt = " ".join(lines[number - 1:number + 1]).strip()
                problems.append(f"{path.relative_to(root)}:{number}: {kind}: {excerpt}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs-root", type=Path, default=DEFAULT_DOCS_ROOT)
    parser.add_argument("--public-only", action="store_true", help="scan only literal Markdown routes in docs/make.jl; source-text check only")
    parser.add_argument("--landing-contract", action="store_true", help="require a plain purpose, standalone Julia identity, and runnable first route on index.md")
    args = parser.parse_args()
    root = args.docs_root.resolve()
    if not root.is_dir():
        parser.error(f"documentation root is absent: {root}")
    try:
        files = public_files(root, public_only=args.public_only)
    except ValueError as error:
        parser.error(str(error))
    if not files:
        parser.error(f"no documentation sources found under: {root}")
    if args.landing_contract:
        missing = landing_contract_findings(root)
        if missing:
            print("LANDING CONTRACT FAILED")
            print("Missing: " + "; ".join(missing))
            return 1
        print("LANDING CONTRACT PASSED")
    problems = findings(root, public_only=args.public_only)
    if problems:
        print("READER SURFACE AUDIT FAILED")
        print("\n".join(problems))
        return 1
    print(f"READER SURFACE AUDIT PASSED files={len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
