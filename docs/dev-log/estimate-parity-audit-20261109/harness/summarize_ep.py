#!/usr/bin/env python3
"""Merge run A (r.tsv, j.tsv, per_cell.tsv) with run B (j2) and the profile runs; print markdown tables."""
import sys, csv, collections, math, re
A, runa, runb_j2, prof_r, prof_j, out = sys.argv[1:7]
def f(x):
    try: return float(x)
    except Exception: return float("nan")
def load(path):
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(path, errors="replace"):
        p = l.rstrip("\n").split("\t") + [""] * 6
        D[p[0]][p[1]].append(p[2:6])
    return D
rows = list(csv.DictReader(open(A + "/per_cell.tsv"), delimiter="\t"))
R, J, B = load(runa + "/r.tsv"), load(runa + "/j.tsv"), load(runb_j2)
for r in rows:
    c = r["cell"]
    rm = {k: a for k, a, *_ in R[c]["meta"]}
    r["R_converged"] = rm.get("converged"); r["R_pdHess"] = rm.get("pdHess")
    ic = [x for x in B[c].get("conv", []) if x[0] == "is_converged"]
    r["J_is_converged"] = ic[0][1] if ic else ""
    r["J_conv_status"] = ic[0][2] if ic else ""
    nf = [x for x in B[c].get("conv", []) if x[0] == "vcov_finite"]
    r["J_nonfinite_se"] = nf[0][2] if nf else ""
    # boundary / flat: R not converged, or R says converged but some coefficient SE > 1e3 or an SD coefficient < -8
    big = any(f(a) > 1e3 for k, a, b, _ in R[c].get("coef", []) for a in [b])
    low = any(f(a) < -8 and k.startswith("sd") for k, a, b, _ in R[c].get("coef", []))
    r["boundary"] = "yes" if (r["R_converged"] != "TRUE" or big or low) else ""
    r["defect_flag"] = "J_unconverged_R_ok" if (r["J_is_converged"] == "false" and r["R_converged"] == "TRUE") else ""
    # simulation moments (Julia: run A marginal, else run B sim2 which drops the harness-only ids kwarg)
    sr = {k.split(":", 1)[1]: f(a) for k, a, *_ in R[c].get("sim", []) if k.startswith("marginal:")}
    sj = {k.split(":", 1)[1]: f(a) for k, a, *_ in J[c].get("sim", []) if k.startswith("marginal:")}
    if not sj: sj = {k: f(a) for k, a, *_ in B[c].get("sim2", []) if k != "ERROR"}
    r["sim_both"] = "yes" if sr and sj else ""
    for m in ("grand_mean", "grand_var", "mean_perobs_var", "p_zero"):
        if sr.get(m) is not None and sj.get(m) is not None and sr["grand_var"] > 0:
            if m == "grand_mean": r["sim_rel_" + m] = abs(sj[m] - sr[m]) / math.sqrt(sr["grand_var"])
            elif m == "p_zero": r["sim_abs_" + m] = abs(sj[m] - sr[m])
            else: r["sim_rel_" + m] = abs(sj[m] - sr[m]) / sr[m]
    r["sim_R"] = "ok" if sr else ("err" if R[c].get("simerr:marginal") else "")
    r["sim_J"] = "ok" if sj else ("err" if (J[c].get("simerr:marginal") or B[c].get("sim2")) else "")
    r["sim_J_err_msg"] = next((a[:100] for k, a, *_ in B[c].get("sim2", []) if k == "ERROR"), "")
    r["sim_R_err_msg"] = next((a[:100] for k, a, *_ in R[c].get("simerr:marginal", [])), "")
cols = list(rows[0].keys())
for r in rows:
    for k in r:
        if k not in cols: cols.append(k)
with open(out, "w") as w:
    w.write("\t".join(cols) + "\n")
    for r in rows: w.write("\t".join(str(r.get(k, "")) for k in cols) + "\n")

def mx(v):
    v = [x for x in v if math.isfinite(x)]
    return max(v) if v else float("nan")
def fmt(x):
    if x != x: return "n/a"
    if x == 0: return "0"
    return f"{x:.1e}" if (abs(x) < 1e-3 or abs(x) >= 1e4) else f"{x:.3g}"
def table(keyf, title, subset):
    G = collections.OrderedDict()
    for r in subset: G.setdefault(keyf(r), []).append(r)
    print(f"\n### {title}\n")
    print("| Group | Cells | max abs logLik diff | max abs coef diff | max abs (SE ratio − 1) | max abs SD diff | max abs pred link diff | max abs pred response diff | max abs CI bound diff |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for k in sorted(G):
        v = G[k]
        print(f"| {k} | {len(v)} | {fmt(mx([abs(f(r['d_logLik'])) for r in v]))} | {fmt(mx([f(r['coef_max_absdiff']) for r in v]))} | {fmt(mx([f(r['se_max_ratio']) - 1 for r in v]))} | {fmt(mx([f(r.get('sd_max_absdiff','')) for r in v]))} | {fmt(mx([f(r['pred_link_maxabs']) for r in v]))} | {fmt(mx([f(r['pred_resp_maxabs']) for r in v]))} | {fmt(mx([f(r['ci_max_absdiff']) for r in v]))} |")
clean = [r for r in rows if not r["boundary"] and not r["defect_flag"]]
print(f"cells {len(rows)}; clean {len(clean)}; boundary/flat {sum(1 for r in rows if r['boundary'])}; Julia-unconverged-while-R-ok {sum(1 for r in rows if r['defect_flag'])}")
table(lambda r: r["family"], "By family (clean cells)", clean)
table(lambda r: r["kind"], "By route (clean cells)", clean)
table(lambda r: r["est"], "By estimator (clean cells)", clean)
table(lambda r: "boundary/flat" if r["boundary"] else "J unconverged, R ok", "Excluded cells", [r for r in rows if r["boundary"] or r["defect_flag"]])
# overall quantiles on the clean cells
def q(v, p):
    v = sorted(x for x in v if math.isfinite(x)); return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
print("\nQuantiles on clean cells (median, 90th, max):")
for k in ("coef_max_absdiff", "se_ratio_m1", "sd_max_absdiff", "pred_link_maxabs", "ci_max_absdiff", "d_logLik"):
    v = [abs(f(r.get(k, ""))) if k != "se_ratio_m1" else f(r["se_max_ratio"]) - 1 for r in clean]
    print(f"  {k}: n={sum(1 for x in v if math.isfinite(x))}, median {fmt(q(v,.5))}, p90 {fmt(q(v,.9))}, max {fmt(mx(v))}")
# simulation
sb = [r for r in rows if r["sim_both"]]
print(f"\nSimulation: both marginal draws available in {len(sb)} of {len(rows)} cells")
for k in ("sim_rel_grand_mean", "sim_rel_grand_var", "sim_rel_mean_perobs_var", "sim_abs_p_zero"):
    v = [f(r.get(k, "")) for r in sb]; print(f"  {k}: median {fmt(q(v,.5))}, p90 {fmt(q(v,.9))}, max {fmt(mx(v))}")
print("  coverage R-ok/J-ok:", collections.Counter((r["sim_R"], r["sim_J"]) for r in rows))

# residual / fitted agreement by route (continuous families compared pointwise)
CONT = {"gaussian", "gamma", "beta", "lognormal", "student"}
print("\n### Fitted values and quantile residuals by route (continuous families; R conditional vs Julia as documented)\n")
print("| Route | SD formula | Cells | fitted agree (<1e-6) | quantile residual agree (<1e-6) | median corr where not |")
print("| --- | --- | --- | --- | --- | --- |")
G = collections.OrderedDict()
for r in rows:
    if r["family"] in CONT: G.setdefault((r["kind"], "yes" if r["pat"] != "sd-none" else "no"), []).append(r)
for k in sorted(G):
    v = G[k]; na = [r for r in v if not f(r["resid_q_maxabs"]) < 1e-6]
    cs = sorted(f(r["resid_q_corr"]) for r in na if math.isfinite(f(r["resid_q_corr"])))
    print(f"| {k[0]} | {k[1]} | {len(v)} | {sum(1 for r in v if f(r['fitted_maxabs']) < 1e-6)} | {sum(1 for r in v if f(r['resid_q_maxabs']) < 1e-6)} | {fmt(cs[len(cs)//2]) if cs else '-'} |")
