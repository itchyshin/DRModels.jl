"""Damage-control tests for the rendered Documenter-site auditor."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "parity_rendered_docs_audit.py"
SPEC = importlib.util.spec_from_file_location("rendered_docs_audit", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDITOR)


class RenderedDocsFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source = root / "source"
        self.site = root / "site"
        (self.source / "guide").mkdir(parents=True)
        (self.site / "guide").mkdir(parents=True)
        (self.site / "assets").mkdir()
        (self.source / "index.md").write_text("# Home\n", encoding="utf-8")
        (self.source / "guide" / "topic.md").write_text("# Topic\n", encoding="utf-8")
        (self.site / "assets" / "logo.svg").write_text("<svg/>", encoding="utf-8")
        (self.site / "assets" / "icons.svg").write_text(
            "<svg><symbol id='present'/></svg>", encoding="utf-8"
        )
        (self.site / "assets" / "site.css").write_text(
            "@font-face { src: url(font.woff2); }", encoding="utf-8"
        )
        (self.site / "assets" / "font.woff2").write_bytes(b"font")
        (self.site / "index.html").write_text(
            "<html><head><title>Home title</title>"
            "<link rel='stylesheet' href='/assets/site.css'></head>"
            "<body><h1 id='home'>Home heading</h1>"
            "<a href='/'>Root</a>"
            "<a href='/guide/topic#topic'>Guide</a>"
            "<a href='https://example.invalid/out'>External</a>"
            "<img src='/assets/logo.svg' alt='Logo'>"
            "<svg><use xlink:href='/assets/icons.svg#present'></use></svg>"
            "</body></html>",
            encoding="utf-8",
        )
        (self.site / "guide" / "topic.html").write_text(
            "<html><head><title>Topic title</title></head>"
            "<body><h1 id='topic'>Topic heading</h1></body></html>",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_audit(self):
        return AUDITOR.audit(self.site, self.source)

    def test_complete_site_passes_and_reports_inventory(self) -> None:
        report = self.run_audit()
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            {page["source_path"] for page in report["source_pages"]},
            {"index.md", "guide/topic.md"},
        )
        self.assertEqual(len(report["pages"]), 2)
        home = next(page for page in report["pages"] if page["path"] == "index.html")
        self.assertEqual(home["title"], "Home title")
        self.assertEqual(home["h1"], ["Home heading"])
        self.assertIn("sha256", home)
        self.assertEqual(report["scope"]["external_links"], "reported_not_checked")
        self.assertEqual(report["external_targets"], ["https://example.invalid/out"])

    def test_missing_rendered_source_page_fails_closed(self) -> None:
        (self.source / "legacy.md").write_text("# Legacy\n", encoding="utf-8")
        report = self.run_audit()
        self.assertTrue(any(item["kind"] == "missing_rendered_source_page" for item in report["failures"]))

    def test_explicit_emitted_source_set_excludes_private_source_notes(self) -> None:
        (self.source / "developer-note.md").write_text("# Private note\n", encoding="utf-8")
        report = AUDITOR.audit(
            self.site,
            self.source,
            emitted_source_paths={"index.md", "guide/topic.md"},
        )
        self.assertEqual(report["failures"], [])
        self.assertEqual(
            {page["source_path"] for page in report["source_pages"]},
            {"index.md", "guide/topic.md"},
        )

    def test_public_surface_rejects_developer_notes_and_dev_log_language(self) -> None:
        (self.source / "developer-notes").mkdir()
        (self.source / "developer-notes" / "private.md").write_text("# Private\n", encoding="utf-8")
        (self.source / "index.md").write_text("# Home\n\nSee docs/dev-log/receipt.md\n", encoding="utf-8")
        report = AUDITOR.audit(
            self.site,
            self.source,
            emitted_source_paths={"index.md", "developer-notes/private.md"},
        )
        self.assertTrue(
            {"forbidden_public_source_path", "forbidden_public_source_language"}
            <= {item["kind"] for item in report["failures"]}
        )

    def test_rendered_internal_work_tracking_is_rejected(self) -> None:
        (self.site / "guide" / "topic.html").write_text(
            "<html><head><title>Topic title</title></head><body>"
            "<h1 id='topic'>Topic heading</h1>"
            "<p>This implementation lane closes PR #456.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "forbidden_rendered_public_language"
            and item.get("page") == "guide/topic.html"
            for item in report["failures"]
        ))

    def test_missing_asset_fragment_and_image_alt_are_reported(self) -> None:
        (self.site / "index.html").write_text(
            "<html><head><title>Home title</title></head><body>"
            "<h1>Home heading</h1><a href='/guide/topic#absent'>Bad fragment</a>"
            "<img src='/assets/missing.svg'><script src='/assets/missing.js'></script>"
            "</body></html>",
            encoding="utf-8",
        )
        report = self.run_audit()
        kinds = {item["kind"] for item in report["failures"]}
        self.assertTrue({"missing_fragment", "missing_asset", "missing_image_alt"} <= kinds)

    def test_missing_css_url_asset_is_reported(self) -> None:
        (self.site / "assets" / "site.css").write_text(
            "@font-face { src: url(missing-font.woff2); }", encoding="utf-8"
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_asset" and item.get("target") == "missing-font.woff2"
            for item in report["failures"]
        ))

    def test_source_page_through_outside_directory_symlink_is_not_covered(self) -> None:
        (self.site / "guide" / "topic.html").unlink()
        (self.site / "guide").rmdir()
        with tempfile.TemporaryDirectory() as outside_directory:
            outside = Path(outside_directory)
            (outside / "topic.html").write_text("<h1>Outside</h1>", encoding="utf-8")
            (self.site / "guide").symlink_to(outside, target_is_directory=True)
            report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_rendered_source_page" and item.get("page") == "guide/topic.md"
            for item in report["failures"]
        ))

    def test_svg_href_fragment_is_checked(self) -> None:
        (self.site / "index.html").write_text(
            "<html><head><title>Home title</title></head><body>"
            "<h1>Home heading</h1>"
            "<svg><use href='/assets/icons.svg#missing'></use></svg>"
            "</body></html>",
            encoding="utf-8",
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_fragment" and item.get("target") == "/assets/icons.svg#missing"
            for item in report["failures"]
        ))

    def test_quoted_css_import_is_checked(self) -> None:
        (self.site / "assets" / "site.css").write_text(
            '@import "missing-theme.css";', encoding="utf-8"
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_asset" and item.get("target") == "missing-theme.css"
            for item in report["failures"]
        ))

    def test_nested_css_dependency_is_scanned(self) -> None:
        (self.site / "assets" / "site.css").write_text(
            '@import "nested.css";', encoding="utf-8"
        )
        (self.site / "assets" / "nested.css").write_text(
            "body { background: url(missing.png); }", encoding="utf-8"
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_asset" and item.get("target") == "missing.png"
            for item in report["failures"]
        ))

    def test_css_import_cycle_is_finite_when_assets_exist(self) -> None:
        (self.site / "assets" / "site.css").write_text(
            '@import "nested.css";', encoding="utf-8"
        )
        (self.site / "assets" / "nested.css").write_text(
            '@import "site.css";', encoding="utf-8"
        )
        self.assertEqual(self.run_audit()["failures"], [])

    def test_base_href_is_refused_instead_of_ignored(self) -> None:
        (self.site / "index.html").write_text(
            "<html><head><base href='/guide/'><title>Home title</title></head>"
            "<body><h1>Home heading</h1></body></html>",
            encoding="utf-8",
        )
        report = self.run_audit()
        self.assertTrue(any(item["kind"] == "unsupported_base_href" for item in report["failures"]))

    def test_inline_css_import_scans_nested_stylesheet(self) -> None:
        (self.site / "assets" / "nested.css").write_text(
            "body { background: url(missing-inline.png); }", encoding="utf-8"
        )
        (self.site / "index.html").write_text(
            "<html><head><title>Home title</title>"
            "<style>@import 'assets/nested.css';</style></head>"
            "<body><h1>Home heading</h1></body></html>",
            encoding="utf-8",
        )
        report = self.run_audit()
        self.assertTrue(any(
            item["kind"] == "missing_asset" and item.get("target") == "missing-inline.png"
            for item in report["failures"]
        ))

    def test_inline_css_import_cycle_is_finite_when_assets_exist(self) -> None:
        (self.site / "assets" / "site.css").write_text(
            '@import "nested.css";', encoding="utf-8"
        )
        (self.site / "assets" / "nested.css").write_text(
            '@import "site.css";', encoding="utf-8"
        )
        (self.site / "index.html").write_text(
            "<html><head><title>Home title</title>"
            "<style>@import 'assets/nested.css';</style></head>"
            "<body><h1>Home heading</h1></body></html>",
            encoding="utf-8",
        )
        self.assertEqual(self.run_audit()["failures"], [])


if __name__ == "__main__":
    unittest.main()
