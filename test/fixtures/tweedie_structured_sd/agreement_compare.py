# Applies the thresholds fixed in docs/dev-log/2026-11-08-tweedie-structured-sd.md.
# python3 agreement_compare.py agree_r.tsv agree_julia.tsv
import csv, sys, math
def load(p):
    return {(r["dataset"], r["model"], int(r["idx"])): r for r in csv.DictReader(open(p), delimiter="\t")}
R, J = load(sys.argv[1]), load(sys.argv[2])
rows = []
for k in sorted(R):
    r, j = R[k], J[k]
    d = abs(float(j["est"]) - float(r["est"]))
    ser = float(r["se"]) if r["se"] not in ("NA", "") else float("nan")
    rows.append((k, d, d / ser if ser == ser and ser > 0 else float("nan"), float(r["est"]), float(j["est"]), j["se"], r["se"]))
def q(xs, p):
    xs = sorted(xs); i = p * (len(xs) - 1); lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)
def summarise(label, sel):
    ds = [x[1] for x in sel]; zs = [x[2] for x in sel if x[2] == x[2]]
    print(f"{label}: n={len(sel)} median|d|={q(ds,.5):.2e} p99|d|={q(ds,.99):.2e} max|d|={max(ds):.2e} "
          f"median d/SE_R={q(zs,.5):.2e} max d/SE_R={max(zs):.2e}")
summarise("ALL", rows)
summarise("excluding phylo_1/sd-z", [x for x in rows if x[0][:2] != ("phylo_1", "sd-z")])
print("non-finite R SE:", sum(1 for x in rows if x[6] in ("NA", "") or not math.isfinite(float(x[6]))))
print("non-finite Julia SE:", sum(1 for x in rows if not math.isfinite(float(x[5]))))
print("Julia not converged:", sum(1 for k in J if J[k]["conv"] != "true"), " R convergence!=0:", sum(1 for k in R if R[k]["conv"] != "0"),
      " R pdHess false:", sum(1 for k in R if R[k]["pdhess"] != "TRUE"))
bd = 0
for k in R:
    for e in (R[k], J[k]):
        v = abs(float(e["est"]))
        i = k[2]
        hi = 6 if (i >= 5) else (15 if i == 3 else 12 if i == 4 else 1e9)
        if k[2] in (1, 2): hi = 1e9
        if v >= hi: bd += 1
print("boundary flags:", bd)
print("per-fit max|d|:")
byfit = {}
for x in rows: byfit.setdefault(x[0][:2], []).append(x[1])
for k, v in byfit.items(): print(" ", k, f"{max(v):.2e}")
print("worst entries:")
for x in sorted(rows, key=lambda x: -x[1])[:8]: print(" ", x[0], f"R={x[3]:.4f} Julia={x[4]:.4f} d={x[1]:.3e} d/SE_R={x[2]:.2f}")
