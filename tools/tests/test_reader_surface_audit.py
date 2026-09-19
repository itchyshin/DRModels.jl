"""Tests for the reader-facing provenance-leak scanner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "reader_surface_audit.py"


class ReaderSurfaceAuditTests(unittest.TestCase):
    def invoke(self, root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--docs-root", str(root), *args],
            text=True,
            capture_output=True,
        )

    def test_clean_public_page_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guide.md").write_text(
                "# Fit a model\n\nUse this guide when your response is continuous.\n",
                encoding="utf-8",
            )
            result = self.invoke(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("READER SURFACE AUDIT PASSED", result.stdout)

    def test_internal_work_tracking_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guide.md").write_text(
                "# Fit a model\n\nThis Arc 7 lane closes #123 after PR #456.\n",
                encoding="utf-8",
            )
            result = self.invoke(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("guide.md:3", result.stdout)

    def test_invalid_roots_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "page.md"
            file.write_text("# Model guide\n", encoding="utf-8")
            for bad in (root / "absent", file):
                with self.subTest(root=bad):
                    self.assertEqual(self.invoke(bad).returncode, 2)

    def test_empty_root_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = self.invoke(Path(tmp))
            self.assertEqual(result.returncode, 2)
            self.assertIn("no documentation sources", result.stderr)

    def test_scientific_language_is_allowed(self) -> None:
        cases = (
            "The interval traces an arc of a circle.",
            "Arc length measures curvature.",
            "Compare treatment effects within each lane.",
            "The sequencing lane is a random effect.",
            "Agent-based simulations generate clustered observations.",
            "Fisher information measures local curvature.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text in cases:
                with self.subTest(text=text):
                    (root / "guide.md").write_text(text, encoding="utf-8")
                    result = self.invoke(root)
                    self.assertEqual(result.returncode, 0, result.stdout)

    def test_process_phrases_are_independently_rejected(self) -> None:
        cases = (
            "Arc 7 is complete.",
            "The implementation lane is complete.",
            "The scoreboard tracks each supported family.",
            "This result has fixture evidence.",
            "The capability\nledger is current.",
            "The agent\nreview approved this fit.",
            "The catch-up\nscoreboard is current.",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text in cases:
                with self.subTest(text=text):
                    (root / "guide.md").write_text("# Model\n\n" + text, encoding="utf-8")
                    result = self.invoke(root)
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertIn("guide.md:3:", result.stdout)

    def test_published_scope_includes_developer_notes_only_when_routed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            root = docs / "src"
            (root / "developer-notes").mkdir(parents=True)
            (root / "guide.md").write_text("# Model guide\n", encoding="utf-8")
            note = root / "developer-notes" / "guide.md"
            note.write_text("An agent review approved this.\n", encoding="utf-8")
            (docs / "make.jl").write_text('makedocs(pages = ["guide.md"])\nerror("never execute")\n', encoding="utf-8")
            result = self.invoke(root, "--public-only")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("files=1", result.stdout)
            (docs / "make.jl").write_text('makedocs(pages = ["guide.md", "Development" => ["developer-notes/guide.md"]])\n', encoding="utf-8")
            result = self.invoke(root, "--public-only")
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("developer-notes/guide.md:1:", result.stdout)

    def test_invalid_published_routes_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            docs = Path(tmp)
            root = docs / "src"
            root.mkdir()
            (root / "guide.md").write_text("# Model guide\n", encoding="utf-8")
            cases = ('[]', '["absent.md"]', '["../outside.md"]', 'read("secret")')
            (docs / "outside.md").write_text("# Outside\n", encoding="utf-8")
            for pages in cases:
                with self.subTest(pages=pages):
                    (docs / "make.jl").write_text(f"makedocs(pages = {pages})\n", encoding="utf-8")
                    self.assertEqual(self.invoke(root, "--public-only").returncode, 2)
            (docs / "make.jl").unlink()
            self.assertEqual(self.invoke(root, "--public-only").returncode, 2)

    def test_fixture_and_ledger_language_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "guide.md").write_text(
                "# Fit a model\n\nThis fixture-backed capability ledger is current.\n",
                encoding="utf-8",
            )
            result = self.invoke(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("guide.md:3", result.stdout)


if __name__ == "__main__":
    unittest.main()
