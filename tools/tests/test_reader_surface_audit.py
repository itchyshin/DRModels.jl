"""Tests for the reader-facing provenance-leak scanner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "reader_surface_audit.py"


class ReaderSurfaceAuditTests(unittest.TestCase):
    def invoke(self, root: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--docs-root", str(root)],
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
            self.assertIn("internal_tracking", result.stdout)

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
