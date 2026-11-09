#!/usr/bin/env python3
"""Classify the gap-map v2 grid.  Usage: python3 -I classify.py <rawdir with r/ and julia/> <outdir>
Reads raw/<engine>/*.out (S/R lines) and cells.tsv; writes gap_map.tsv and derived tables.

Per-engine outcome: FITS, NONCONVERGED, REFUSES (deliberate abort), NOSYNTAX (Julia has no spelling),
ERRORS (any other exception: a crash), TIMEOUT, PROCESS_DIED, MISSING.
Cell class: both-fit (sub-classed agree / ll-differs / fitted-differs / both-differ), R-only (Julia behind),
Julia-only (Julia ahead), both-refuse, crash-R / crash-Julia / crash-both, nonconverged, inconclusive.
Agreement thresholds are fixed in advance: |logLik diff| <= 1e-4; fitted mean and sd within 1e-4 * (1 + |value|)."""
import os, sys, re, glob, collections, csv, json

here = os.path.dirname(os.path.abspath(__file__))
raw, outdir = sys.argv[1], sys.argv[2]
os.makedirs(outdir, exist_ok=True)
COLS = ["status", "etype", "msg", "msgfull", "ll", "conv", "extra", "secs", "warn", "metrics", "probes"]

def load(eng):
    res = {}
    for f in sorted(glob.glob(os.path.join(raw, eng, "*.out"))):
        for l in open(f, errors="replace"):
            p = l.rstrip("\n").split("\t")
            if p[0] == "R" and len(p) >= 4:
                p = p + [""] * (14 - len(p))
                res[p[2]] = dict(zip(COLS, p[3:14]))
    return res

R, J = load("r"), load("julia")
cells = [dict(zip(["id", "area", "chunk", "probes", "r", "jl", "note"], l.split("\t") + [""] * 7)) for l in open(os.path.join(here, "cells.tsv")).read().splitlines()[1:]]

NUMERIC = re.compile(r"(not positive definite|PosDef|NaN|non-?finite|Cholesky|singular|did not converge|line search|LAPACK|overflow|DomainError|sqrt will only)", re.I)
CRASH = {"MethodError", "BoundsError", "KeyError", "DimensionMismatch", "UndefVarError", "UndefRefError", "TypeError", "InexactError",
         "AssertionError", "PosDefException", "SingularException", "LAPACKException", "DomainError", "OverflowError", "StackOverflowError",
         "DivideError", "OutOfMemoryError", "InvalidStateException", "NaNMath", "ArgumentError:internal"}

def outcome(eng, r):
    s = r["status"]
    if s in ("FITS", "NONCONVERGED", "TIMEOUT", "NOSYNTAX"): return s, ""
    if s == "PROCESS_DIED": return "ERRORS", "process-died"
    if s == "RUNNER_ERROR": return "RUNNER_ERROR", r["etype"]
    et, msg = r["etype"], (r["msgfull"] or r["msg"])
    if eng == "r":
        if et.startswith("abort:"):
            return ("ERRORS", "numeric-abort") if NUMERIC.search(msg) else ("REFUSES", "cli_abort")
        if "could not find function" in msg: return "REFUSES", "no-such-function"
        if "unused argument" in msg: return "REFUSES", "constructor-lacks-argument"
        return "ERRORS", "base-error"
    if et in ("ArgumentError", "ErrorException"):
        return ("ERRORS", "numeric-" + et) if (NUMERIC.search(msg) and "grouping column" not in msg) else ("REFUSES", et)
    return "ERRORS", et

def sig(r):
    m = r["msgfull"] or r["msg"]
    m = re.sub(r"\x1b\[[0-9;]*m", "", m); m = re.sub(r"^ArgumentError:\s*", "", m)
    m = re.split(r" [✖ℹ] ", m)[0]
    m = re.sub(r"`[^`]*`", "`T`", m); m = re.sub(r"\"[^\"]*\"", "\"S\"", m); m = re.sub(r"\d+(\.\d+)*", "N", m)
    return m[:170].strip()

def kv(s):
    d = {}
    for t in s.split(";"):
        if "=" in t:
            k, v = t.split("=", 1); d[k] = v
    return d

def fl(x):
    try: return float(x)
    except Exception: return None

def close(a, b, tol=1e-4):
    return a is not None and b is not None and abs(a - b) <= tol * (1 + max(abs(a), abs(b)))

rows = []
for c in cells:
    row = dict(id=c["id"], area=c["area"], chunk=c["chunk"], note=c["note"])
    info = {}
    for eng, tab in (("r", R), ("j", J)):
        x = tab.get(c["id"])
        if x is None:
            info[eng] = dict(o="MISSING", sub="", x=dict.fromkeys(COLS, ""), m={}, p={})
            continue
        o, sub = outcome("r" if eng == "r" else "julia", x)
        info[eng] = dict(o=o, sub=sub, x=x, m=kv(x["metrics"]), p=kv(x["probes"]))
    ro, jo = info["r"]["o"], info["j"]["o"]
    row["r_outcome"], row["r_sub"], row["j_outcome"], row["j_sub"] = ro, info["r"]["sub"], jo, info["j"]["sub"]
    row["r_ll"], row["j_ll"] = info["r"]["x"]["ll"], info["j"]["x"]["ll"]
    row["r_msg"] = info["r"]["x"]["msg"] if ro != "FITS" else ""
    row["j_msg"] = info["j"]["x"]["msg"] if jo != "FITS" else ""
    row["r_sig"] = sig(info["r"]["x"]) if ro in ("REFUSES", "ERRORS") else ""
    row["j_sig"] = sig(info["j"]["x"]) if jo in ("REFUSES", "ERRORS") else ""
    row["j_at"] = info["j"]["x"]["extra"] if jo == "ERRORS" else ""
    row["r_extra"] = info["r"]["x"]["extra"]; row["r_warn"] = info["r"]["x"]["warn"]; row["j_warn"] = info["j"]["x"]["warn"]
    row["detail"] = ""; row["flags"] = ""
    flags = []
    if ro == "FITS" and "pdHess=FALSE" in row["r_extra"]: flags.append("R:pdHess=FALSE")
    if ro == "FITS" and "is_converged=FALSE" in row["r_extra"]: flags.append("R:is_converged=FALSE")
    crash = "+".join(s for s, o in (("R", ro), ("Julia", jo)) if o == "ERRORS")
    if ro == "FITS" and jo == "FITS":
        row["class"] = "both-fit"
        rm, jm = info["r"]["m"], info["j"]["m"]
        dll = abs(float(row["r_ll"]) - float(row["j_ll"])); row["ll_diff"] = "%.3e" % dll
        ll_ok = dll <= 1e-4
        fit_ok = close(fl(rm.get("fit_mean")), fl(jm.get("fit_mean"))) and close(fl(rm.get("fit_sd")), fl(jm.get("fit_sd")))
        pred_ok = (close(fl(rm.get("pred_mean")), fl(jm.get("pred_mean"))) if ("pred_mean" in rm and "pred_mean" in jm) else None)
        row["sub"] = "agree" if ll_ok and fit_ok else "ll-differs-fitted-agrees" if (not ll_ok and fit_ok) else "ll-agrees-fitted-differs" if (ll_ok and not fit_ok) else "both-differ"
        row["fit_agree"] = str(fit_ok); row["pred_agree"] = str(pred_ok)
        row["npar_r"], row["npar_j"] = rm.get("npar", ""), jm.get("npar", "")
        if "|" not in c["r"].split("family")[0] and rm.get("npar") and jm.get("npar") and rm["npar"] != jm["npar"]: flags.append("npar:R=%s,J=%s" % (rm["npar"], jm["npar"]))
        if ll_ok and rm.get("aic") and jm.get("aic") and not close(fl(rm["aic"]), fl(jm["aic"]), 1e-6): flags.append("AIC-differs-though-ll-agrees")
        if rm.get("nobs") and jm.get("nobs") and rm["nobs"] != jm["nobs"]: flags.append("nobs:R=%s,J=%s" % (rm["nobs"], jm["nobs"]))
        if jm.get("pred_new_consistent") == "false": flags.append("Julia:predict(newdata)!=predict(full)")
        if rm.get("pred_new_consistent") == "false": flags.append("R:predict(newdata)!=predict(full)")
        if pred_ok is False and "(" not in "".join(re.findall(r"\(\s*[01x]\S*\s*\|", c["r"])): flags.append("predict-mean-differs")
        if jm.get("se_finite") == "false": flags.append("Julia:se_nonfinite")
        if rm.get("se_finite") == "false": flags.append("R:se_nonfinite")
        # entry-point gaps: Julia probe fails where R's works
        rp, jp = info["r"]["p"], info["j"]["p"]
        for k in sorted(set(rp) | set(jp)):
            rv, jv = rp.get(k, ""), jp.get(k, "")
            if rv == "ok" and jv.startswith("ERR"): flags.append("entry:%s:Julia-fails" % k)
            if jv == "ok" and rv.startswith("ERR"): flags.append("entry:%s:R-fails" % k)
    elif "ERRORS" in (ro, jo):
        row["class"] = "crash-" + {"R": "R", "Julia": "Julia", "R+Julia": "both"}[crash]
    elif ro in ("TIMEOUT", "MISSING", "RUNNER_ERROR") or jo in ("TIMEOUT", "MISSING", "RUNNER_ERROR"):
        row["class"] = "inconclusive"
    elif "NONCONVERGED" in (ro, jo):
        row["class"] = "nonconverged"
    elif ro == "FITS" and jo in ("REFUSES", "NOSYNTAX"):
        row["class"] = "R-only" if jo == "REFUSES" else "R-only-no-Julia-syntax"
    elif jo == "FITS" and ro == "REFUSES":
        row["class"] = "Julia-only"
    else:
        row["class"] = "both-refuse"
    row.setdefault("sub", ""); row.setdefault("ll_diff", ""); row.setdefault("fit_agree", ""); row.setdefault("pred_agree", "")
    row.setdefault("npar_r", ""); row.setdefault("npar_j", "")
    row["flags"] = ";".join(flags)
    row["r_fit_mean"] = info["r"]["m"].get("fit_mean", ""); row["j_fit_mean"] = info["j"]["m"].get("fit_mean", "")
    row["r_fit_sd"] = info["r"]["m"].get("fit_sd", ""); row["j_fit_sd"] = info["j"]["m"].get("fit_sd", "")
    rows.append(row)

FIELDS = ["id", "area", "class", "sub", "r_outcome", "r_sub", "j_outcome", "j_sub", "r_ll", "j_ll", "ll_diff", "fit_agree", "pred_agree", "npar_r", "npar_j",
          "r_fit_mean", "j_fit_mean", "r_fit_sd", "j_fit_sd", "flags", "r_extra", "r_msg", "j_msg", "r_sig", "j_sig", "j_at", "r_warn", "j_warn", "note", "chunk", "detail"]
def write(name, rs, fields=FIELDS):
    with open(os.path.join(outdir, name), "w") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(fields)
        for r in rs: w.writerow([str(r.get(k, "")).replace("\t", " ") for k in fields])
write("gap_map.tsv", rows)
cnt = collections.Counter((r["class"] + ((":" + r["sub"]) if r["sub"] else "")) for r in rows)
write("crash_cells.tsv", [r for r in rows if r["class"].startswith("crash")])
write("disagreements.tsv", [r for r in rows if r["class"] == "both-fit" and r["sub"] != "agree"])
write("flagged_agreeing.tsv", [r for r in rows if r["class"] == "both-fit" and r["sub"] == "agree" and r["flags"]])
write("julia_behind_cells.tsv", [r for r in rows if r["class"] in ("R-only", "R-only-no-Julia-syntax")])
write("julia_only_cells.tsv", [r for r in rows if r["class"] == "Julia-only"])
write("nonconverged_cells.tsv", [r for r in rows if r["class"] in ("nonconverged", "inconclusive")])
groups = collections.defaultdict(list)
for r in rows:
    if r["class"] in ("R-only", "R-only-no-Julia-syntax"): groups[r["j_sig"] or r["j_outcome"]].append(r)
with open(os.path.join(outdir, "julia_behind_groups.tsv"), "w") as fh:
    fh.write("n_cells\tjulia_signature\tareas\texample_ids\n")
    for sgn, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        fh.write("%d\t%s\t%s\t%s\n" % (len(rs), sgn, ",".join(sorted({r["area"] for r in rs})), " || ".join(r["id"] for r in rs[:3])))
by_area = collections.defaultdict(collections.Counter)
for r in rows: by_area[r["area"]][r["class"] + ((":" + r["sub"]) if r["sub"] else "")] += 1
with open(os.path.join(outdir, "outcome_table.tsv"), "w") as fh:
    fh.write("class\tn\n")
    for k, v in sorted(cnt.items()): fh.write("%s\t%d\n" % (k, v))
    fh.write("\narea\tclass\tn\n")
    for a in sorted(by_area):
        for k, v in sorted(by_area[a].items()): fh.write("%s\t%s\t%d\n" % (a, k, v))
json.dump(dict(cells=len(rows), counts=dict(cnt), r_fits=sum(r["r_outcome"] == "FITS" for r in rows), julia_fits=sum(r["j_outcome"] == "FITS" for r in rows)),
          open(os.path.join(outdir, "summary.json"), "w"), indent=1)
print(json.dumps(dict(cnt), indent=1))
