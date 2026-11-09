#!/usr/bin/env python3
"""Compare the long-TSV dumps of ep_r.R and ep_j.jl, cell by cell.

Usage: python3 -I compare_ep.py <r.tsv> <j.tsv> <outdir>
Writes per_cell.tsv (one row of summary metrics per cell) and detail_*.tsv (row-level differences).
Matching rules (documented in the note):
  coef   rows grouped by distributional parameter (mu, sigma, any sd* block -> "sd"); positional within group.
  par    sigma on the response scale from the sigma coefficient (exp, delta SE); random-effect SD and correlation
         rows matched by order within kind (sd, cor), R rows without a finite SE (per-level structured SDs) dropped.
  ci     fixed rows matched like coef; random-effect rows compared on the SD scale (Julia log scale exponentiated).
"""
import sys, csv, math, collections, re, os

rf, jf, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)


def load(path):
    D = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path, errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) < 3:
            continue
        p += [""] * (6 - len(p))
        D[p[0]][p[1]].append(tuple(p[2:6]))
    return D


def f(x):
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


R, J = load(rf), load(jf)
cells = sorted(set(R) | set(J))
NAN = float("nan")


def grp(name):
    d = name.split(":")[0]
    return "sd" if d.startswith("sd") else d


def by_group(rows, keep):
    g = collections.OrderedDict()
    for k, a, b, c in rows:
        d = grp(k)
        if keep(d):
            g.setdefault(d, []).append((k, f(a), f(b)))
    return g


def vecdiff(a, b):
    """max abs diff, max rel diff (|d| / max(|a|,1e-8 scale)), corr"""
    n = min(len(a), len(b))
    if n == 0:
        return NAN, NAN, NAN
    d = [abs(a[i] - b[i]) for i in range(n)]
    scale = max(max(abs(x) for x in a[:n]), 1e-12)
    ma, mb = sum(a[:n]) / n, sum(b[:n]) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a[:n])); sb = math.sqrt(sum((x - mb) ** 2 for x in b[:n]))
    corr = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (sa * sb) if sa > 0 and sb > 0 else NAN
    return max(d), max(d) / scale, corr


def vec(rows):
    return [f(a) for k, a, *_ in rows]


per = []
det = collections.defaultdict(list)
for c in cells:
    r, j = R.get(c, {}), J.get(c, {})
    fam, kind, lab, q, pat, cm, est = c.split("|")
    row = dict(cell=c, family=fam, kind=kind, q=q, lab=lab, pat=pat, est=est)
    mr = {k: a for k, a, *_ in r.get("meta", [])}
    mj = {k: a for k, a, *_ in j.get("meta", [])}
    row["fit_status"] = "R_err" if "FIT_ERROR" in mr else ("J_err" if "FIT_ERROR" in mj else ("none" if not mr or not mj else "ok"))
    if row["fit_status"] != "ok":
        row["note"] = (mr.get("FIT_ERROR") or mj.get("FIT_ERROR") or "")[:120]
        per.append(row); continue
    for m in ("logLik", "AIC", "BIC", "df", "nobs"):
        row["d_" + m] = f(mj.get(m)) - f(mr.get(m))
    # ---- coefficients
    cr = by_group(r.get("coef", []), lambda d: True)
    cj = by_group(j.get("coef", []), lambda d: d in cr)
    maxd, maxratio, nmis = 0.0, 1.0, 0
    worst = ""
    for d in cr:
        a, b = cr[d], cj.get(d)
        if b is not None and len(b) > len(a):
            row["coef_extra_J_rows"] = row.get("coef_extra_J_rows", 0) + len(b) - len(a); b = b[:len(a)]
        if b is None or len(a) != len(b):
            nmis += 1; det["coef"].append((c, d, "COUNT", len(a), len(b) if b else -1)); continue
        for (k1, e1, s1), (k2, e2, s2) in zip(a, b):
            dd = abs(e1 - e2)
            ratio = max(s1 / s2, s2 / s1) if s1 > 0 and s2 > 0 else NAN
            det["coef"].append((c, d, k1, k2, e1, e2, s1, s2, dd, ratio))
            if dd > maxd: maxd, worst = dd, k1
            if ratio == ratio and ratio > maxratio: maxratio = ratio
            if ratio != ratio: nmis += 0
    row["coef_max_absdiff"] = maxd; row["coef_worst"] = worst; row["se_max_ratio"] = maxratio; row["coef_group_mismatch"] = nmis
    row["n_coef"] = sum(len(v) for v in cr.values())
    row["se_nonfinite_R"] = sum(1 for d in cr for x in cr[d] if not math.isfinite(x[2]))
    row["se_nonfinite_J"] = sum(1 for d in cj for x in cj[d] if not math.isfinite(x[2]))
    # ---- derived sigma (response scale) and random-effect SD / cor rows
    rpar = r.get("par", []); jpar = j.get("par", [])
    # sigma
    rs = [(f(a), f(b)) for k, a, b, c3 in rpar if k.startswith("distributional-scale|sigma")]
    js = cj.get("sigma") or []
    if rs and js:
        e, s = js[0][1], js[0][2]
        row["sigma_resp_diff"] = abs(math.exp(e) - rs[0][0]); row["sigma_se_ratio"] = (lambda a, b: max(a / b, b / a) if a > 0 and b > 0 else NAN)(math.exp(e) * s, rs[0][1])
    rsd = [(k, f(a), f(b)) for k, a, b, c3 in rpar if k.startswith("random-effect-sd|") and not re.search(r":s\d+$", k)]
    rcor = [(k, f(a), f(b)) for k, a, b, c3 in rpar if k.startswith("random-effect-correlation|")]
    jerr = [a for k, a, *_ in jpar if k == "ERROR"]
    jsd = [(k, f(a), f(b)) for k, a, b, c3 in jpar if k.startswith("sd|")]
    jcor = [(k, f(a), f(b)) for k, a, b, c3 in jpar if k.startswith("cor|")]
    row["re_status"] = "J_vc_refused" if jerr else ""
    row["n_R_sd"] = len(rsd); row["n_J_sd"] = len(jsd); row["n_R_cor"] = len(rcor); row["n_J_cor"] = len(jcor)
    for tag, A, B in (("sd", rsd, jsd), ("cor", rcor, jcor)):
        if A and B and len(A) == len(B):
            dmax = max(abs(a[1] - b[1]) for a, b in zip(A, B))
            rat = [max(a[2] / b[2], b[2] / a[2]) for a, b in zip(A, B) if a[2] > 0 and b[2] > 0 and math.isfinite(a[2]) and math.isfinite(b[2])]
            sa, sb = sorted(a[1] for a in A), sorted(b[1] for b in B)
            row[tag + "_max_absdiff"] = dmax
            row[tag + "_sorted_absdiff"] = max(abs(x - y) for x, y in zip(sa, sb))
            row[tag + "_se_max_ratio"] = max(rat) if rat else NAN
            for a, b in zip(A, B): det[tag].append((c, a[0], b[0], a[1], b[1], a[2], b[2]))
        elif A or B:
            row[tag + "_count_mismatch"] = f"R{len(A)}/J{len(B)}"
            for a in A: det[tag].append((c, a[0], "", a[1], NAN, a[2], NAN))
            for b in B: det[tag].append((c, "", b[0], NAN, b[1], NAN, b[2]))
    # ---- confidence intervals (Wald)
    cir = [(k, f(a), f(b)) for k, a, b, c3 in r.get("ci", []) if k.startswith("fixef:") and k.endswith("|link")]
    cij = [(k, f(a), f(b)) for k, a, b, c3 in j.get("ci", [])]
    cirg = by_group([(k.split("fixef:")[1], a, b, "") for k, a, b in [(x[0], x[1], x[2]) for x in cir]], lambda d: True)
    cijg = by_group([(k, a, b, "") for k, a, b in cij], lambda d: d in cirg)
    cimax = 0.0; cin = 0
    for d in cirg:
        a, b = cirg[d], cijg.get(d)
        if b and len(a) == len(b):
            for x, y in zip(a, b):
                cimax = max(cimax, abs(x[1] - y[1]), abs(x[2] - y[2])); cin += 1
                det["ci"].append((c, d, x[0], y[0], x[1], y[1], x[2], y[2]))
    if cin: row["ci_max_absdiff"] = cimax
    row["ci_status"] = "R_err" if any(k == "ERROR" for k, *_ in r.get("ci", [])) else ("J_err" if any(k == "ERROR" for k, *_ in j.get("ci", [])) else "")
    # ---- predictions / fitted / residuals
    for sec in ("pred_resp", "pred_link", "fitted", "resid_q", "resid_r"):
        a, b = r.get(sec, []), j.get(sec, [])
        if any(k == "ERROR" for k, *_ in a): row[sec + "_status"] = "R_err:" + [x[1] for x in a if x[0] == "ERROR"][0][:80]
        if any(k == "ERROR" for k, *_ in b): row[sec + "_status"] = (row.get(sec + "_status", "") + " J_err:" + [x[1] for x in b if x[0] == "ERROR"][0][:80]).strip()
        va, vb = vec([x for x in a if x[0] != "ERROR"]), vec([x for x in b if x[0] != "ERROR"])
        if va and vb and len(va) == len(vb):
            md, rd, cr_ = vecdiff(va, vb)
            row[sec + "_maxabs"] = md; row[sec + "_maxrel"] = rd; row[sec + "_corr"] = cr_
        elif va and vb:
            row[sec + "_status"] = f"len R{len(va)}/J{len(vb)}"
    # J fitted vs R fitted at 8 rows vs R predict at same rows (convention check)
    # ---- simulation moments
    for mode in ("marginal", "zero", "cond"):
        sr = {k.split(":", 1)[1]: f(a) for k, a, *_ in r.get("sim", []) if k.startswith(mode + ":")}
        sj = {k.split(":", 1)[1]: f(a) for k, a, *_ in j.get("sim", []) if k.startswith(mode + ":")}
        for m in ("grand_mean", "grand_var", "mean_perobs_var", "p_zero"):
            if m in sr: row[f"simR_{mode}_{m}"] = sr[m]
            if m in sj: row[f"simJ_{mode}_{m}"] = sj[m]
    row["simerr_R_marginal"] = next((a[:90] for k, a, *_ in r.get("simerr:marginal", [])), "")
    row["simerr_J_marginal"] = next((a[:90] for k, a, *_ in j.get("simerr:marginal", [])), "")
    per.append(row)

cols = []
for r_ in per:
    for k in r_:
        if k not in cols: cols.append(k)
with open(os.path.join(outdir, "per_cell.tsv"), "w") as w:
    w.write("\t".join(cols) + "\n")
    for r_ in per:
        w.write("\t".join("" if r_.get(k) is None else (f"{r_[k]:.6g}" if isinstance(r_[k], float) else str(r_[k])) for k in cols) + "\n")
for name, rows in det.items():
    with open(os.path.join(outdir, f"detail_{name}.tsv"), "w") as w:
        for t in rows:
            w.write("\t".join(f"{x:.8g}" if isinstance(x, float) else str(x) for x in t) + "\n")
print(len(per), "cells;", sum(1 for r_ in per if r_["fit_status"] == "ok"), "compared")
