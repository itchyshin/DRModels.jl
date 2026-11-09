#!/usr/bin/env python3
"""Stratified sample of the wave-10 'both-fit agree' cells. Seeded, deterministic."""
import csv, random, collections, sys
gap, out = sys.argv[1], sys.argv[2]
rows = [r for r in csv.DictReader(open(gap), delimiter="\t") if r["class"] == "both-fit" and r["ll_flag"] == "ok"]
rng = random.Random(20261109)
strata = collections.defaultdict(list)
for r in rows: strata[(r["family"], r["kind"], r["est"])].append(r)
pick = {}
for k in sorted(strata):
    r = rng.choice(strata[k]); pick[r["cell_id"]] = r
# top up to 90: prefer cells whose pattern/q/lab/cm differ from those already picked in the stratum
left = [r for r in rows if r["cell_id"] not in pick]
rng.shuffle(left)
seen = collections.Counter((r["family"], r["kind"], r["est"], r["pat"], r["q"], r["lab"]) for r in pick.values())
for r in sorted(left, key=lambda r: seen[(r["family"], r["kind"], r["est"], r["pat"], r["q"], r["lab"])]):
    if len(pick) >= 90: break
    key = (r["family"], r["kind"], r["est"], r["pat"], r["q"], r["lab"])
    if seen[key] == 0:
        pick[r["cell_id"]] = r; seen[key] += 1
for r in sorted(left, key=lambda r: r["cell_id"]):
    if len(pick) >= 90: break
    if r["cell_id"] not in pick: pick[r["cell_id"]] = r
w = open(out, "w")
w.write("cell_id\n")
for k in sorted(pick): w.write(k + "\n")
c = collections.Counter(r["family"] for r in pick.values()); print(len(pick), dict(c))
print(collections.Counter(r["kind"] for r in pick.values()), collections.Counter(r["est"] for r in pick.values()))
