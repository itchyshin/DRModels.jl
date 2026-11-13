#!/usr/bin/env python3
"""Tests for check_parity_ledger.py on tiny fixture ledgers."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).with_name("check_parity_ledger.py")
HEAD = "map\tcell_id\tfamily\tmodel_shape\tclass\tevidence\treference\tnote\n"


def run(rows):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.tsv"
        p.write_text(HEAD + "".join("\t".join(r) + "\n" for r in rows))
        return subprocess.run([sys.executable, str(CHECKER), str(p)], capture_output=True, text=True)


def row(cell, cls, m="map1"):
    return [m, cell, "gaussian", "x", cls, "ll 0", "", ""]


class CheckParityLedger(unittest.TestCase):
    def test_zero_open_passes(self):
        r = run([row("a", "agree"), row("b", "agree-under-Laplace"), row("c", "signed-by-design", "map2"),
                 row("d", "R-not-converged", "map2")])
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("map1: agree 1, agree-under-Laplace 1", r.stdout)
        self.assertIn("open cells: 0", r.stdout)

    def test_one_open_fails(self):
        r = run([row("a", "agree"), row("b", "open")])
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("open cells: 1", r.stdout)

    def test_repeated_cell_is_malformed(self):
        r = run([row("a", "agree"), row("a", "agree")])
        self.assertEqual(r.returncode, 2)
        self.assertIn("repeated cell", r.stdout)

    def test_same_id_in_two_maps_is_allowed(self):
        self.assertEqual(run([row("a", "agree"), row("a", "agree", "map2")]).returncode, 0)

    def test_bad_class_word_is_malformed(self):
        r = run([row("a", "agrees")])
        self.assertEqual(r.returncode, 2)
        self.assertIn("bad class", r.stdout)


if __name__ == "__main__":
    unittest.main()
