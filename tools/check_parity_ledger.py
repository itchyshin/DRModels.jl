#!/usr/bin/env python3
"""Check the R-Julia parity ledger: usage check_parity_ledger.py [LEDGER.tsv].

The ledger (default test/parity/parity-ledger.tsv) has one row per cell of the two
parity gap maps. The check:

1. every row has all required columns and a class from the fixed list;
2. no (map, cell_id) pair appears twice;
3. prints the count per class per map;
4. exits 1 if any row is `open` (parity is done when this count is zero),
   2 if the file is malformed, 0 otherwise.

Classes: agree, agree-under-Laplace, signed-by-design, R-not-converged, open.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

CLASSES = ["agree", "agree-under-Laplace", "signed-by-design", "R-not-converged", "open"]
COLUMNS = ["map", "cell_id", "family", "model_shape", "class", "evidence", "reference", "note"]
DEFAULT = Path(__file__).resolve().parent.parent / "test" / "parity" / "parity-ledger.tsv"


def check(path):
    """Return (errors, counts) where counts[map][class] is an int."""
    errors = []
    counts = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return ["missing column(s): " + ", ".join(missing)], counts
        seen = set()
        for n, row in enumerate(reader, start=2):
            key = (row["map"], row["cell_id"])
            if not row["map"] or not row["cell_id"]:
                errors.append("line %d: empty map or cell_id" % n)
                continue
            if key in seen:
                errors.append("line %d: repeated cell %s %s" % (n, *key))
            seen.add(key)
            if row["class"] not in CLASSES:
                errors.append("line %d: bad class %r for %s" % (n, row["class"], row["cell_id"]))
                continue
            counts.setdefault(row["map"], Counter())[row["class"]] += 1
    return errors, counts


def main(argv):
    path = Path(argv[1]) if len(argv) > 1 else DEFAULT
    errors, counts = check(path)
    for e in errors:
        print("ERROR:", e)
    for m in sorted(counts):
        print(m + ": " + ", ".join("%s %d" % (c, counts[m][c]) for c in CLASSES))
    if errors:
        return 2
    n_open = sum(c["open"] for c in counts.values())
    print("open cells: %d" % n_open)
    return 1 if n_open else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
