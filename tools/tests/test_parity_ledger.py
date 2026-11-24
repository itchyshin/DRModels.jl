"""Regression tests for cross-language export name matching."""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "parity_ledger.py"
SPEC = importlib.util.spec_from_file_location("parity_ledger", SCRIPT)
LEDGER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEDGER)


class ExportAliasTests(unittest.TestCase):
    def test_beta_family_matches_julia_beta_family_constructor(self):
        self.assertTrue(LEDGER.has_julia_twin("beta_family", ["Beta"]))

    def test_julia_exports_are_read_from_the_requested_git_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "src").mkdir()
            source = repo / "src" / "DRModels.jl"
            source.write_text("module DRModels\nexport CommittedName\n\nend\n")
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
            subprocess.run(["git", "-C", str(repo), "add", "src/DRModels.jl"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
            source.write_text("module DRModels\nexport DirtyWorkingTreeName\n\nend\n")

            self.assertEqual(LEDGER.drmjl_exports(repo, "HEAD"), ["CommittedName"])


if __name__ == "__main__":
    unittest.main()
