#!/usr/bin/env python3
"""Tests for check_parity_ledger.py on tiny fixture ledgers."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(__file__).with_name("check_parity_ledger.py")
HEAD = "map\tcell_id\tfamily\tmodel_shape\tclass\tevidence\treference\tnote\n"


def run(rows, raw=None, ids=None, head=HEAD):
    """Write a fixture ledger plus id lists (by default the ids of `rows`) and run the check.

    `raw` replaces the table body with literal text; `ids` maps "map1"/"map2" to a list of ids."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "ledger.tsv"
        p.write_text(head + (raw if raw is not None else "".join("\t".join(r) + "\n" for r in rows)))
        for m in ("map1", "map2"):
            want = ids[m] if ids is not None else [r[1] for r in rows if r[0] == m]
            (Path(d) / ("ids-%s.txt" % m)).write_text("".join(i + "\n" for i in want))
        return subprocess.run([sys.executable, str(CHECKER), str(p)], capture_output=True, text=True)


def row(cell, cls, m="map1", ref=""):
    return [m, cell, "gaussian", "x", cls, "ll 0", ref, ""]


class CheckParityLedger(unittest.TestCase):
    def test_zero_open_passes(self):
        r = run([row("a", "agree"), row("b", "agree-under-Laplace"), row("c", "signed-by-design", "map2", "decision"),
                 row("d", "R-not-converged", "map2")])
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertIn("map1: agree 1, agree-under-Laplace 1", r.stdout)
        self.assertIn("open cells: 0", r.stdout)

    def test_one_open_fails(self):
        r = run([row("a", "agree"), row("b", "open", ref="#1")])
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

    def test_empty_ledger_is_malformed(self):
        r = run([], ids={"map1": ["a"], "map2": []})
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("no rows", r.stdout)

    def test_short_row_is_malformed(self):
        r = run([], raw="map1\tx\tg\tm\tsigned-by-design\n", ids={"map1": ["x"], "map2": []})
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("wrong number of columns", r.stdout)

    def test_signed_row_needs_reference(self):
        r = run([row("a", "signed-by-design")])
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("empty reference", r.stdout)

    def test_open_row_needs_reference(self):
        r = run([row("a", "open")])
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("empty reference", r.stdout)

    def test_deleted_row_is_caught_by_id_list(self):
        r = run([row("a", "agree")], ids={"map1": ["a", "b"], "map2": []})
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("in the id list but not in the ledger", r.stdout)

    def test_extra_row_is_caught_by_id_list(self):
        r = run([row("a", "agree"), row("b", "agree")], ids={"map1": ["a"], "map2": []})
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("in the ledger but not in the id list", r.stdout)

    def test_missing_id_list_is_malformed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.tsv"
            p.write_text(HEAD + "\t".join(row("a", "agree")) + "\n")
            r = subprocess.run([sys.executable, str(CHECKER), str(p)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout)
        self.assertIn("missing id list", r.stdout)


if __name__ == "__main__":
    unittest.main()
