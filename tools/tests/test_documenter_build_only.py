"""Regression tests for the pull-request Documenter render path."""

from __future__ import annotations

import unittest
from pathlib import Path


MAKE = Path(__file__).parents[2] / "docs" / "make.jl"


class DocumenterBuildOnlyTests(unittest.TestCase):
    def test_non_deploy_path_keeps_the_vitepress_writer_enabled(self) -> None:
        source = MAKE.read_text(encoding="utf-8")
        self.assertIn("build_vitepress = true", source)
        marker = 'if get(ENV, "DRM_DOCS_DEPLOY", "true") == "true"'
        self.assertIn(marker, source)
        _, separator, non_deploy = source.partition("\nelse\n")
        self.assertTrue(separator)
        self.assertNotIn("DocumenterVitepress.deploydocs", non_deploy)


if __name__ == "__main__":
    unittest.main()
