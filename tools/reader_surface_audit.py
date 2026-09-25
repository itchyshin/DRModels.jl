#!/usr/bin/env python3
"""Reject internal work-tracking language from public documentation sources.

This is intentionally a narrow mechanical guard.  It does not decide whether a
scientific claim is useful or earned; Pat and Rose review that separately.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_ROOT = REPO_ROOT / "docs" / "src"
PATTERNS = {
    "internal_handoff": re.compile(r"\bHANDOVER(?:\.|%2e)md\b", flags=re.IGNORECASE),
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

# Deliberately exact phrases, restricted to beginner routes. Words such as
# "cell", "validation", or "gate" alone also have scientific meanings.
BEGINNER_ROUTES = {
    "index.md", "getting-started.md", "capabilities.md", "coming-from-r.md",
    "model-guides/model-map.md", "model-guides/model-workflow.md",
}
BEGINNER_PROCESS = re.compile(
    r"\b(?:validation|evidence|parity)\s+receipt\b|"
    r"\badmitted\s+routes?\b|\bcertified\s+cells?\b|\bmerge\s+gate\b",
    flags=re.IGNORECASE,
)

# These are explicit editorial anchors, not inferred measures of readability.
# An intentional heading change should update its contract and tests together.
# (question heading, section containing a next-step link)
FLOW_CONTRACTS = {
    "index.md": ("What is distributional regression?", "Choose your analysis"),
    "model-guides/model-map.md": ("What can I fit today?", "Which page next"),
    "getting-started.md": (None, "Where to go next"),
}


def reader_flow_findings(root: Path, files: list[Path]) -> list[str]:
    """Check retained headings/order and next-page links on three entry routes.

    Recognise ATX headings outside fenced code, and ordinary inline Markdown
    links. This cannot judge the prose, execute examples, inspect generated
    docstrings, or prove that readers understand a model.
    """
    problems: list[str] = []
    published = {path.resolve() for path in files}
    for path in files:
        route = path.relative_to(root).as_posix()
        if route not in FLOW_CONTRACTS:
            continue
        question, next_heading = FLOW_CONTRACTS[route]
        lines = path.read_text(encoding="utf-8").splitlines()
        visible = list(lines)
        headings: list[tuple[int, int, str]] = []
        fence = None
        first_example = None
        for number, line in enumerate(lines):
            marker = re.match(r"^\s{0,3}(`{3,}|~{3,})(.*)$", line)
            if fence is not None:
                visible[number] = ""
                if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                    fence = None
                continue
            if marker:
                fence = marker[1]
                visible[number] = ""
                if first_example is None and re.match(r"(?:julia|@example|@repl)\b", marker[2].strip()):
                    first_example = number
                continue
            heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", line)
            if heading:
                headings.append((number, len(heading[1]), heading[2]))
        by_title = {title: (number, level) for number, level, title in headings}
        if question:
            if question not in by_title:
                problems.append(f"{route}:1: reader_flow: missing question heading '{question}'")
            elif first_example is not None and by_title[question][0] > first_example:
                problems.append(f"{route}:1: reader_flow: explain the question before the first runnable example")
        if next_heading not in by_title:
            problems.append(f"{route}:1: reader_flow: missing next-step heading '{next_heading}'")
            continue
        start, level = by_title[next_heading]
        end = next((number for number, depth, _ in headings if number > start and depth <= level), len(lines))
        section = "\n".join(visible[start + 1:end])
        linked_page = False
        for target in re.findall(r"\[[^\]]+\]\(([^\s)]+)(?:\s+[^)]*)?\)", section):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path or url.path.startswith("@"):
                continue
            relative = unquote(url.path)
            destination = (root / relative.lstrip("/")) if relative.startswith("/") else (path.parent / relative)
            if destination.resolve() != path.resolve() and destination.resolve() in published:
                linked_page = True
                break
        if not linked_page:
            problems.append(f"{route}:{start + 1}: reader_flow: next step must link to another published local page")
    return problems


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
    files = public_files(root, public_only=public_only)
    for path in files:
        lines = path.read_text(encoding="utf-8").splitlines()
        # Keep newline boundaries for useful source locations while allowing a
        # wrapped phrase such as "capability\nledger" to be recognised.
        text = "\n".join(re.sub(r"[ \t]+", " ", line) for line in lines)
        patterns = dict(PATTERNS)
        if path.relative_to(root).as_posix() in BEGINNER_ROUTES:
            patterns["beginner_process"] = BEGINNER_PROCESS
        for kind, pattern in patterns.items():
            for match in pattern.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                excerpt = " ".join(lines[number - 1:number + 1]).strip()
                problems.append(f"{path.relative_to(root)}:{number}: {kind}: {excerpt}")
    if public_only:
        problems.extend(reader_flow_findings(root, files))
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
