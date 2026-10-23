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

    def test_landing_contract_requires_plain_purpose_and_first_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            landing = root / "index.md"
            landing.write_text("# DRModels.jl\n", encoding="utf-8")

            result = self.invoke(root, "--landing-contract")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("LANDING CONTRACT FAILED", result.stdout)

            landing.write_text(
                "# DRModels.jl\n\n"
                "Distributional regression models how predictors change a response's "
                "average and variability.\n\n"
                "A standalone Julia package for scientists.\n\n"
                "Start with the [runnable first model](getting-started.md).\n",
                encoding="utf-8",
            )
            result = self.invoke(root, "--landing-contract")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("LANDING CONTRACT PASSED", result.stdout)

    def test_internal_handoff_destinations_are_rejected(self) -> None:
        cases = (
            "See `HANDOVER.md` for model support.",
            "See [details](https://github.com/org/repo/blob/main/HANDOVER.md).",
            "See [details](../HANDOVER.md#supported-models).",
            "See [details](../HANDOVER%2Emd).",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text in cases:
                with self.subTest(text=text):
                    (root / "guide.md").write_text(text, encoding="utf-8")
                    result = self.invoke(root)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("internal_handoff", result.stdout)

    def test_beginner_process_phrases_are_rejected_but_scientific_terms_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for text in ("Read the validation receipt.", "This is an admitted route.",
                         "Compare the certified cells.", "Wait for the merge gate."):
                with self.subTest(text=text):
                    (root / "getting-started.md").write_text(text, encoding="utf-8")
                    result = self.invoke(root)
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn("beginner_process", result.stdout)
            (root / "getting-started.md").write_text(
                "Count cells in each sample. Validate the model with simulated data. "
                "Study receipt of treatment and a gate in a laboratory maze.", encoding="utf-8")
            self.assertEqual(self.invoke(root).returncode, 0)

    def reader_flow_fixture(self, docs: Path, route: str, text: str) -> Path:
        root = docs / "src"
        (root / "model-guides").mkdir(parents=True)
        (root / route).write_text(text, encoding="utf-8")
        (root / "example.md").write_text("# A worked example\n", encoding="utf-8")
        (docs / "make.jl").write_text(f'makedocs(pages = ["{route}", "example.md"])\n', encoding="utf-8")
        return root

    def test_published_key_routes_keep_question_and_next_step_sections(self) -> None:
        cases = {
            "index.md": "## What is distributional regression?\nDescribe average and spread.\n\n## Choose your analysis\n[Fit a model](example.md)\n",
            "model-guides/model-map.md": "# What can I fit today?\nChoose a model for your response.\n\n## Which page next\n[Fit a model](../example.md)\n",
            "getting-started.md": "# Getting started\n\n## Where to go next\n[Another model](example.md)\n",
        }
        for route, text in cases.items():
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                root = self.reader_flow_fixture(Path(tmp), route, text)
                self.assertEqual(self.invoke(root, "--public-only").returncode, 0)
                # A title elsewhere on the page cannot stand in for a missing
                # route section; the section must be a real Markdown heading.
                (root / route).write_text(text.replace("## ", ""), encoding="utf-8")
                result = self.invoke(root, "--public-only")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("reader_flow", result.stdout)

    def test_next_step_requires_an_existing_local_page_in_its_section(self) -> None:
        for link in ("No next step.", "[Next](missing.md)", "[Next](https://example.org)",
                     "[Next](getting-started.md#top)", "[Next](hidden.md)",
                     "```markdown\n[Next](example.md)\n```"):
            with self.subTest(link=link), tempfile.TemporaryDirectory() as tmp:
                root = self.reader_flow_fixture(Path(tmp), "getting-started.md",
                    "# Getting started\n[Early link](example.md)\n\n## Where to go next\n" + link)
                (root / "hidden.md").write_text("# An unpublished page\n", encoding="utf-8")
                result = self.invoke(root, "--public-only")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("next step", result.stdout)

    def test_question_heading_cannot_be_removed_while_retaining_next_steps(self) -> None:
        for route, next_section, link in (
            ("index.md", "Choose your analysis", "example.md"),
            ("model-guides/model-map.md", "Which page next", "../example.md"),
        ):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as tmp:
                root = self.reader_flow_fixture(Path(tmp), route,
                    f"# Technical details\n\n## {next_section}\n[Next]({link})\n")
                result = self.invoke(root, "--public-only")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("missing question heading", result.stdout)

    def test_question_must_precede_the_first_runnable_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text = "```@example first\nfit = drm(...)\n```\n\n## What is distributional regression?\nMeaning.\n\n## Choose your analysis\n[Fit](example.md)\n"
            root = self.reader_flow_fixture(Path(tmp), "index.md", text)
            result = self.invoke(root, "--public-only")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("before the first runnable example", result.stdout)

    def test_headings_in_code_fences_cannot_satisfy_reader_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.reader_flow_fixture(Path(tmp), "getting-started.md",
                "# Getting started\n\n```markdown\n## Where to go next\n[Next](example.md)\n```\n")
            result = self.invoke(root, "--public-only")
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("reader_flow", result.stdout)


if __name__ == "__main__":
    unittest.main()
