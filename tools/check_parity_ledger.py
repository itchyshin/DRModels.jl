#!/usr/bin/env python3
"""Check the R-Julia parity ledger: usage check_parity_ledger.py [LEDGER.tsv].

The ledger (default test/parity/parity-ledger.tsv) has one row per cell of the two
parity gap maps. The check:

1. the ledger has at least one row, and every row has as many columns as the header
   and a class from the fixed list;
2. no (map, cell_id) pair appears twice;
3. every `signed-by-design` and `open` row has a non-empty reference column;
4. the set of cell ids of each map equals the committed id list next to the ledger
   (`ids-map1.txt`, `ids-map2.txt`, one id per line), so deleting or adding a row
   cannot make the check pass;
5. prints the count per class per map;
6. exits 1 if any row is `open` (parity is done when this count is zero),
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


NEED_REFERENCE = ("signed-by-design", "open")


def read_ids(path):
    """Return the set of ids in an id-list file, or None if the file is missing."""
    try:
        with open(path, encoding="utf-8") as fh:
            return {line.rstrip("\n") for line in fh if line.strip()}
    except FileNotFoundError:
        return None


def check(path):
    """Return (errors, counts) where counts[map][class] is an int."""
    errors = []
    counts = {}
    ledger_ids = {}
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t", quoting=csv.QUOTE_NONE)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            return ["missing column(s): " + ", ".join(missing)], counts
        seen = set()
        for n, row in enumerate(reader, start=2):
            if None in row.values() or None in row:
                errors.append("line %d: wrong number of columns (the header has %d)" % (n, len(COLUMNS)))
                continue
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
            if row["class"] in NEED_REFERENCE and not row["reference"].strip():
                errors.append("line %d: %s row %s has an empty reference" % (n, row["class"], row["cell_id"]))
            counts.setdefault(row["map"], Counter())[row["class"]] += 1
            ledger_ids.setdefault(row["map"], set()).add(row["cell_id"])
    if not counts and not errors:
        errors.append("ledger has no rows")
    for m in ("map1", "map2"):
        listed = read_ids(Path(path).parent / ("ids-%s.txt" % m))
        if listed is None:
            errors.append("missing id list ids-%s.txt next to the ledger" % m)
            continue
        have = ledger_ids.get(m, set())
        for cid in sorted(listed - have):
            errors.append("%s: cell %s is in the id list but not in the ledger" % (m, cid))
        for cid in sorted(have - listed):
            errors.append("%s: cell %s is in the ledger but not in the id list" % (m, cid))
    for m in sorted(set(ledger_ids) - {"map1", "map2"}):
        errors.append("unknown map %r" % m)
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
