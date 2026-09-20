#!/usr/bin/env python3
"""Fail-closed static audit of a rendered Documenter HTML site.

The audit verifies local rendered paths, fragments, image alt attributes, and
source-page coverage. It reports external targets without fetching them. It
does not assess visual layout, keyboard behavior, screen-reader quality, or a
complete accessibility standard.
"""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ElementTree


CSS_URL = re.compile(
    r"url\(\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s)]+))\s*\)",
    re.IGNORECASE | re.DOTALL,
)
CSS_IMPORT = re.compile(r"""@import\s+(['"])([^'"]+)\1""", re.IGNORECASE)
PRIVATE_SOURCE_PATH = re.compile(r"(?:^|/)developer-notes(?:/|$)")
PRIVATE_SOURCE_LANGUAGE = re.compile(
    r"(?:developer\s+note|docs/dev-log|reviewer\s+contract|\bissue\s*#\d+|\bphase[- ]\d+)",
    re.IGNORECASE,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _external(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme or parsed.netloc or value.startswith("//"))


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.targets: list[tuple[str, str]] = []
        self.ids: set[str] = set()
        self.images_without_alt = 0
        self._title_depth = 0
        self._title_chunks: list[str] = []
        self.title = ""
        self._h1_depth = 0
        self._h1_chunks: list[str] = []
        self.h1: list[str] = []
        self._style_depth = 0
        self.inline_css: list[str] = []
        self.base_hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = dict(attrs_list)
        if attrs.get("id"):
            self.ids.add(attrs["id"])
        if tag in {"a", "area"} and attrs.get("href"):
            self.targets.append(("page", attrs["href"]))
        elif tag == "link" and attrs.get("href"):
            self.targets.append(("asset", attrs["href"]))
        elif tag in {"use", "image"}:
            for name in ("href", "xlink:href"):
                if attrs.get(name):
                    self.targets.append(("asset", attrs[name]))
        elif attrs.get("src"):
            self.targets.append(("asset", attrs["src"]))
        if tag == "img" and "alt" not in attrs:
            self.images_without_alt += 1
        if attrs.get("srcset"):
            for item in attrs["srcset"].split(","):
                candidate = item.strip().split(None, 1)[0] if item.strip() else ""
                if candidate:
                    self.targets.append(("asset", candidate))
        if attrs.get("style"):
            self.inline_css.append(attrs["style"])
        if tag == "base" and attrs.get("href"):
            self.base_hrefs.append(attrs["href"])
        if tag == "title":
            self._title_depth = 1
            self._title_chunks = []
        elif self._title_depth:
            self._title_depth += 1
        if tag == "h1":
            self._h1_depth = 1
            self._h1_chunks = []
        elif self._h1_depth:
            self._h1_depth += 1
        if tag == "style":
            self._style_depth = 1
            self.inline_css.append("")
        elif self._style_depth:
            self._style_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self._title_depth:
            self._title_depth -= 1
            if self._title_depth == 0:
                self.title = "".join(self._title_chunks).strip()
        if self._h1_depth:
            self._h1_depth -= 1
            if self._h1_depth == 0:
                self.h1.append("".join(self._h1_chunks).strip())
        if self._style_depth:
            self._style_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._title_depth:
            self._title_chunks.append(data)
        if self._h1_depth:
            self._h1_chunks.append(data)
        if self._style_depth and self.inline_css:
            self.inline_css[-1] += data


def _target(root: Path, page: Path, value: str, kind: str) -> tuple[str, Path | None, str]:
    """Resolve one local target without allowing escape outside ``root``."""
    if _external(value) or value.startswith(("data:", "javascript:", "mailto:", "tel:")):
        return "external", None, ""
    parsed = urlsplit(value)
    raw_path = unquote(parsed.path)
    fragment = unquote(parsed.fragment)
    base = root if raw_path.startswith("/") else page.parent
    initial = (base / raw_path.lstrip("/")) if raw_path else page
    if not _inside(initial.resolve(strict=False), root):
        return "outside", initial.resolve(strict=False), fragment
    candidates = [initial]
    if kind == "page":
        if initial.suffix == "":
            candidates.extend((Path(str(initial) + ".html"), initial / "index.html"))
        elif initial.is_dir():
            candidates.append(initial / "index.html")
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if not _inside(resolved, root):
            # A synthetic clean-URL fallback can leave the root for a root
            # route (for example, `site-root.html`). The original target was
            # already checked above, so this is not an attempted escape.
            continue
        if resolved.is_file():
            return "file", resolved, fragment
    return "missing", initial, fragment


def _css_targets(text: str) -> list[str]:
    url_targets = [
        next(value for value in match.groups() if value is not None).strip()
        for match in CSS_URL.finditer(text)
        if next(value for value in match.groups() if value is not None).strip()
    ]
    return url_targets + [
        match.group(2).strip() for match in CSS_IMPORT.finditer(text) if match.group(2).strip()
    ]


def _read_html(path: Path) -> _PageParser:
    parser = _PageParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return parser


def _failure(kind: str, page: Path | None = None, target: str | None = None, detail: str | None = None) -> dict[str, str]:
    result = {"kind": kind}
    if page is not None:
        result["page"] = page.as_posix()
    if target is not None:
        result["target"] = target
    if detail is not None:
        result["detail"] = detail
    return result


def _source_pages(source_root: Path, emitted_source_paths: Iterable[str] | None) -> list[Path]:
    """Return the source pages Documenter is configured to emit."""
    if emitted_source_paths is None:
        return sorted(source_root.rglob("*.md"))
    pages: list[Path] = []
    for raw_path in sorted(set(emitted_source_paths)):
        candidate = source_root / raw_path
        if candidate.suffix != ".md" or not _inside(candidate.resolve(strict=False), source_root):
            raise ValueError(f"invalid emitted source path: {raw_path}")
        pages.append(candidate)
    return pages


def audit(
    site_root: Path | str,
    source_root: Path | str,
    emitted_source_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return a JSON-ready report. Any local unresolved reference is a failure."""
    site_root = Path(site_root).resolve()
    source_root = Path(source_root).resolve()
    failures: list[dict[str, str]] = []
    external: set[str] = set()
    parsed_pages: dict[Path, _PageParser] = {}

    if not site_root.is_dir():
        return {"schema_version": 1, "site_root": str(site_root), "source_root": str(source_root),
                "scope": {"external_links": "reported_not_checked"}, "pages": [], "source_pages": [],
                "external_targets": [], "failures": [_failure("missing_site_root", detail=str(site_root))]}
    if not source_root.is_dir():
        return {"schema_version": 1, "site_root": str(site_root), "source_root": str(source_root),
                "scope": {"external_links": "reported_not_checked"}, "pages": [], "source_pages": [],
                "external_targets": [], "failures": [_failure("missing_source_root", detail=str(source_root))]}

    html_pages = sorted(
        path for path in site_root.rglob("*.html")
        if not path.is_symlink() and _inside(path.resolve(), site_root)
    )
    if not html_pages:
        failures.append(_failure("no_rendered_html_pages", detail=str(site_root)))
    audited_pages = set(html_pages)

    for page in html_pages:
        try:
            parsed_pages[page] = _read_html(page)
        except (OSError, UnicodeError, ValueError) as error:
            failures.append(_failure("html_parse_error", page.relative_to(site_root), detail=str(error)))

    source_pages: list[dict[str, Any]] = []
    for source in _source_pages(source_root, emitted_source_paths):
        if not source.is_file():
            failures.append(_failure("missing_emitted_source_page", detail=str(source.relative_to(source_root))))
            continue
        if source.is_symlink():
            failures.append(_failure("symlinked_source_page", source.relative_to(source_root)))
            continue
        relative = source.relative_to(source_root)
        source_text = source.read_text(encoding="utf-8")
        if PRIVATE_SOURCE_PATH.search(relative.as_posix()):
            failures.append(_failure("forbidden_public_source_path", relative))
        if PRIVATE_SOURCE_LANGUAGE.search(source_text):
            failures.append(_failure("forbidden_public_source_language", relative))
        expected = site_root / relative.with_suffix(".html")
        resolved_expected = expected.resolve(strict=False)
        record = {
            "source_path": relative.as_posix(),
            "source_sha256": _sha256(source),
            "rendered_path": relative.with_suffix(".html").as_posix(),
            "rendered_exists": (
                expected.is_file()
                and not expected.is_symlink()
                and _inside(resolved_expected, site_root)
                and resolved_expected in audited_pages
            ),
        }
        source_pages.append(record)
        if not record["rendered_exists"]:
            failures.append(_failure("missing_rendered_source_page", relative, record["rendered_path"]))

    def resolve(referrer: Path, kind: str, value: str) -> None:
        status, target, fragment = _target(site_root, referrer, value, kind)
        rel_referrer = referrer.relative_to(site_root)
        if status == "external":
            external.add(value)
            return
        if status != "file" or target is None:
            failures.append(_failure("outside_site_root" if status == "outside" else f"missing_{kind}", rel_referrer, value))
            return
        if fragment:
            if target.suffix.lower() == ".html":
                target_parser = parsed_pages.get(target)
                if target_parser is None:
                    try:
                        target_parser = _read_html(target)
                        parsed_pages[target] = target_parser
                    except (OSError, UnicodeError, ValueError) as error:
                        failures.append(_failure("html_parse_error", target.relative_to(site_root), detail=str(error)))
                        return
                ids = target_parser.ids
            elif target.suffix.lower() == ".svg":
                try:
                    ids = {
                        element.attrib["id"] for element in ElementTree.parse(target).iter()
                        if "id" in element.attrib
                    }
                except (OSError, UnicodeError, ElementTree.ParseError) as error:
                    failures.append(_failure(
                        "unsupported_svg_fragment", rel_referrer, value, str(error)
                    ))
                    return
            else:
                failures.append(_failure("unsupported_asset_fragment", rel_referrer, value))
                return
            if fragment not in ids:
                failures.append(_failure("missing_fragment", rel_referrer, value))

    inspected_css: set[Path] = set()

    def inspect_css_contents(text: str, referrer: Path) -> None:
        for value in _css_targets(text):
            resolve(referrer, "asset", value)
            status, target, _ = _target(site_root, referrer, value, "asset")
            if status == "file" and target is not None and target.suffix.lower() == ".css":
                inspect_css(target)

    def inspect_css(path: Path) -> None:
        path = path.resolve()
        if path in inspected_css:
            return
        inspected_css.add(path)
        try:
            inspect_css_contents(path.read_text(encoding="utf-8"), path)
        except (OSError, UnicodeError) as error:
            failures.append(_failure("css_read_error", path.relative_to(site_root), detail=str(error)))

    for page, parser in parsed_pages.items():
        rel_page = page.relative_to(site_root)
        for base_href in parser.base_hrefs:
            failures.append(_failure("unsupported_base_href", rel_page, base_href))
        for kind, value in parser.targets:
            resolve(page, kind, value)
        for _ in range(parser.images_without_alt):
            failures.append(_failure("missing_image_alt", rel_page))
        for css in parser.inline_css:
            inspect_css_contents(css, page)
        for kind, value in parser.targets:
            if kind != "asset" or not value.lower().split("?", 1)[0].endswith(".css"):
                continue
            status, css_path, _ = _target(site_root, page, value, "asset")
            if status != "file" or css_path is None:
                continue
            inspect_css(css_path)

    page_records = []
    for page in html_pages:
        parser = parsed_pages.get(page)
        if parser is None:
            continue
        page_records.append({
            "path": page.relative_to(site_root).as_posix(),
            "sha256": _sha256(page),
            "title": parser.title,
            "h1": parser.h1,
            "ids": sorted(parser.ids),
        })
    failures.sort(key=lambda item: (item["kind"], item.get("page", ""), item.get("target", ""), item.get("detail", "")))
    return {
        "schema_version": 1,
        "site_root": str(site_root),
        "source_root": str(source_root),
        "scope": {
            "external_links": "reported_not_checked",
            "visual_layout": "not_checked",
            "accessibility": "limited_to_image_alt_absence",
            "source_coverage": "all_markdown_sources" if emitted_source_paths is None else "explicit_emitted_sources",
        },
        "pages": page_records,
        "source_pages": source_pages,
        "external_targets": sorted(external),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-root", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument(
        "--make",
        type=Path,
        help="Documenter make.jl; requires pagesonly = true and derives emitted source pages from pages =.",
    )
    parser.add_argument("--report", type=Path, help="Write the complete JSON report here")
    args = parser.parse_args()
    emitted_source_paths = None
    if args.make:
        from parity_docs_audit import navigation_entries

        make_text = args.make.read_text(encoding="utf-8")
        if not re.search(r"\bpagesonly\s*=\s*true\b", make_text):
            parser.error("--make requires Documenter pagesonly = true")
        emitted_source_paths = [entry["path"] for entry in navigation_entries(args.make)]
    report = audit(args.site_root, args.source_root, emitted_source_paths)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "RENDERED_DOCS_AUDIT "
        f"pages={len(report['pages'])} source_pages={len(report['source_pages'])} "
        f"external={len(report['external_targets'])} failures={len(report['failures'])}"
    )
    for failure in report["failures"]:
        print("FAIL", json.dumps(failure, sort_keys=True))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
