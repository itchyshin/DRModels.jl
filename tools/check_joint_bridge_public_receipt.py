#!/usr/bin/env python3
"""Independent checks of retained R public outputs against shared joint math."""
import hashlib
import json
import math
import statistics
import sys
import tomllib
from pathlib import Path

import check_joint_frontend_fit_receipt as oracle
import receipt_paths

require = oracle.require


def check(receipt, reference, direct, rroot, jroot, native=False):
    require(receipt.get("status") == "PASS", "public adapter status")
    require(receipt.get("source_unchanged") is True, "source changed")
    # Path-portable: recorded absolute paths are re-keyed by root and repo-relative
    # path; the per-file sha256 values must equal these checkouts' (receipt_paths.py).
    rroot, jroot = rroot.resolve(), jroot.resolve()
    rpaths = set(rroot.glob("R/*.R")) | {rroot / "NAMESPACE"}
    rpaths |= {p for p in (rroot / "src").rglob("*") if p.suffix in (".cpp", ".h", ".hpp")}
    current = receipt_paths.local(rpaths, rroot, set((jroot / "src").rglob("*.jl")), jroot, oracle.digest)
    recorded_root = receipt_paths.loaded_root(receipt.get("runtime", {}).get("source"))
    require(receipt.get("source_before") == receipt.get("source_after")
            and receipt_paths.recorded(receipt.get("source_before"), recorded_root) == current, "public source manifest")
    require(receipt.get("runner_sha256") == oracle.digest(rroot / "tools/run-julia-joint-public.R"), "runner hash")
    frozen_json = jroot / "docs/dev-log/evidence/julia-r-parity/missing-predictor-oracle/native-mi-oracle-003.json"
    require(receipt.get("fixture_sha256") == oracle.digest(frozen_json), "fixture hash")
    require(receipt.get("runtime", {}).get("threads") == 1 and receipt["runtime"].get("blas") == 1, "thread budget")
    require(set(receipt.get("cases", {})) == {"gaussian", "bernoulli"}, "case denominator")
    for kind in ("gaussian", "bernoulli"):
        case, frozen, julia = receipt["cases"][kind], reference[kind], direct["cases"][kind]
        require(case.get("status") == "PASS", kind + ": case status")
        p = 7 if kind == "gaussian" else 6
        theta, cov = case.get("raw_theta"), case.get("raw_covariance")
        oracle.finite_vector(theta, kind + ": theta", p)
        oracle.finite_square(cov, p, kind + ": covariance")
        require(max(abs(a-b) for a, b in zip(theta, julia["theta"])) <= 1e-10, kind + ": direct theta")
        require(max(abs(cov[i][j]-julia["covariance"][i][j]) for i in range(p) for j in range(p)) <= 1e-10, kind + ": direct covariance")
        require(abs(case.get("loglik", math.inf) - julia["loglik"]) <= 1e-10, kind + ": direct loglik")
        require(case.get("nobs") == sum(frozen["y_observed"]) == 156, kind + ": observed nobs")
        pub = case.get("coefficients", {})
        blocks = {"mu", "sigma", "mi_x"} | ({"sigma_mi_x"} if kind == "gaussian" else set())
        require(set(pub) == blocks, kind + ": public blocks")
        oracle.finite_vector(pub["mu"], kind + ": public mu block", 3)
        oracle.finite_vector(pub["mi_x"], kind + ": public predictor block", 2)
        require(isinstance(pub["sigma"], (int, float)) and math.isfinite(pub["sigma"]), kind + ": public sigma block")
        public_theta = pub["mu"] + [pub["sigma"]] + pub["mi_x"]
        jac = [1.0] * p
        if kind == "gaussian":
            public_theta.append(pub["sigma_mi_x"])
            jac[-1] = math.exp(theta[-1])
        expected_theta = theta[:-1] + [jac[-1]] if kind == "gaussian" else theta
        require(max(abs(a-b) for a, b in zip(public_theta, expected_theta)) <= 1e-10, kind + ": public scales")
        V = case.get("public_covariance")
        oracle.finite_square(V, p, kind + ": public covariance")
        require(max(abs(V[i][j] - cov[i][j]*jac[i]*jac[j]) for i in range(p) for j in range(p)) <= 1e-10, kind + ": delta covariance")
        rows = case.get("imputed_all")
        require(isinstance(rows, list) and len(rows) == 160, kind + ": row denominator")
        mu, sigma = case.get("training_mu"), case.get("training_sigma")
        oracle.finite_vector(mu, kind + ": training mu", 160)
        oracle.finite_vector(sigma, kind + ": training sigma", 160)
        fresh_ids = [i for i in range(160) if frozen["x_observed"][i] and frozen["y_observed"][i]][:6]
        fresh_mu, fresh_sigma = case.get("newdata_mu"), case.get("newdata_sigma")
        oracle.finite_vector(fresh_mu, kind + ": newdata mu", 6)
        oracle.finite_vector(fresh_sigma, kind + ": newdata sigma", 6)
        for j, i in enumerate(fresh_ids):
            expected = theta[0]+theta[1]*frozen["z"][i]+theta[2]*frozen["x"][i]
            require(abs(fresh_mu[j]-expected) <= 1e-10 and abs(fresh_sigma[j]-math.exp(theta[3])) <= 1e-10, kind + ": newdata values")
        dpars = ["mu"]*3 + ["sigma"] + ["mi_x"]*2 + (["sigma_mi_x"] if kind == "gaussian" else [])
        terms = ["(Intercept)", "z", "mi(x)", "(Intercept)", "(Intercept)", "z"] + (["x"] if kind == "gaussian" else [])
        summary, wald = case.get("summary"), case.get("wald")
        require(isinstance(summary, list) and len(summary) == p and isinstance(wald, list) and len(wald) == p, kind + ": inference denominator")
        critical = statistics.NormalDist().inv_cdf(0.975)
        for j, (s, w) in enumerate(zip(summary, wald)):
            sd = math.sqrt(V[j][j])
            label = dpars[j] + ":" + terms[j]
            natural = kind == "gaussian" and j == p-1
            require(s.get("dpar") == dpars[j] and s.get("term") == terms[j], kind + ": summary labels")
            require(abs(s.get("estimate", math.inf)-public_theta[j]) <= 1e-10 and abs(s.get("std.error", math.inf)-sd) <= 1e-10, kind + ": summary values")
            if natural:
                require(s.get("statistic") in (None, "NA", "NaN") and s.get("p.value") in (None, "NA", "NaN"), kind + ": SD boundary test")
            else:
                z = public_theta[j]/sd
                pvalue = math.erfc(abs(z)/math.sqrt(2))
                require(abs(s.get("statistic", math.inf)-z) <= 1e-9 and abs(s.get("p.value", math.inf)-pvalue) <= 1e-10, kind + ": summary tests")
            require(w.get("parm") == "fixef:"+label and w.get("tmb_parameter") == label and w.get("index") == j+1, kind + ": interval labels")
            require(w.get("level") == 0.95 and w.get("method") == "wald" and w.get("conf.status") == "ok", kind + ": interval status")
            require(w.get("scale") == ("response" if natural else "linear_predictor") and w.get("transformation") == ("exp" if natural else "identity"), kind + ": interval scale")
            # Native positive-scale Wald intervals use raw log coordinates;
            # summary SEs and public covariance above still use the full Jacobian.
            if natural:
                raw_sd = math.sqrt(cov[j][j])
                lower, upper = math.exp(theta[j]-critical*raw_sd), math.exp(theta[j]+critical*raw_sd)
            else:
                lower, upper = public_theta[j]-critical*sd, public_theta[j]+critical*sd
            require(abs(w.get("lower", math.inf)-lower) <= 1e-9 and abs(w.get("upper", math.inf)-upper) <= 1e-9, kind + ": interval values")
        for i, row in enumerate(rows):
            require(row.get("original_row") == frozen["original_row"][i] and row.get("model_row") == i+1, kind + ": row identity")
            require(row.get("observed") is frozen["x_observed"][i] and row.get("variable") == "x", kind + ": predictor identity")
            if kind == "gaussian":
                estimate, _, se, source = oracle.gaussian_prediction(theta, cov, frozen, i)
            else:
                estimate, _, se, source = oracle.bernoulli_prediction(theta, frozen, i)
            require(abs(row.get("estimate", math.inf)-estimate) <= 1e-9, kind + ": conditional mean")
            require(row.get("source") == source and row.get("uncertainty_status") == "ok", kind + ": row status")
            if se is None:
                require(row.get("std_error") in (None, "NA", "NaN"), kind + ": observed SE")
            else:
                require(isinstance(row.get("std_error"), (int, float)) and abs(row["std_error"]-se) <= 1e-9, kind + ": conditional SE")
            expected_mu = theta[0]+theta[1]*frozen["z"][i]+theta[2]*estimate
            require(abs(mu[i]-expected_mu) <= 1e-10 and abs(sigma[i]-math.exp(theta[3])) <= 1e-10, kind + ": fitted predictions")
        # Even adapter-only validation must preserve native operation failures.
        errors = case.get("native_errors", {})
        require(set(errors) == {"theta", "loglik", "imputed_mean", "imputed_se", "training_mean", "newdata_mean"}, kind + ": native denominator")
        native_theta = case.get("native_raw_theta")
        oracle.finite_vector(native_theta, kind + ": native theta", p)
        native_ll = case.get("native_loglik")
        require(isinstance(native_ll, (int, float)) and math.isfinite(native_ll), kind + ": native loglik")
        require(abs(native_ll - sum(oracle.expected_row_loglik(native_theta, frozen, kind))) <= 1e-8, kind + ": native likelihood oracle")
        calculated = {"theta": max(abs(a-b) for a,b in zip(theta, native_theta)),
                      "loglik": abs(case["loglik"]-native_ll)}
        native_rows = case.get("native_imputed_all")
        require(isinstance(native_rows, list) and len(native_rows) == 160, kind + ": native imputation denominator")
        for i, row in enumerate(native_rows):
            require(row.get("original_row") == frozen["original_row"][i] and row.get("model_row") == i+1, kind + ": native row identity")
            require(isinstance(row.get("estimate"), (int, float)) and math.isfinite(row["estimate"]), kind + ": native imputation mean")
            if not frozen["x_observed"][i]:
                require(isinstance(row.get("std_error"), (int, float)) and math.isfinite(row["std_error"]), kind + ": native imputation SE")
        calculated["imputed_mean"] = max(abs(a["estimate"]-b["estimate"]) for a,b in zip(rows, native_rows))
        calculated["imputed_se"] = max(abs(a["std_error"]-b["std_error"]) for i,(a,b) in enumerate(zip(rows,native_rows)) if not frozen["x_observed"][i])
        for key, actual, count in (("training_mean", mu, 160), ("newdata_mean", fresh_mu, 6)):
            field = "native_training_prediction" if key == "training_mean" else "native_newdata_prediction"
            operation = case.get(field, {})
            require(operation.get("status") in ("PASS", "ERROR"), kind + ": native operation status")
            if operation["status"] == "ERROR":
                require(isinstance(operation.get("message"), str) and bool(operation["message"]), kind + ": native error message")
                require(errors[key] in (None, "NA", "NaN") and case.get("native_status") == "FAIL", kind + ": native error concealed")
                calculated[key] = math.inf
            else:
                oracle.finite_vector(operation.get("value"), kind + ": native prediction", count)
                calculated[key] = max(abs(a-b) for a,b in zip(actual, operation["value"]))
        for key, delta in calculated.items():
            if math.isfinite(delta):
                require(isinstance(errors[key], (int, float)) and abs(delta-errors[key]) <= 1e-12, kind + ": native delta record")
        native_pass = all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 4e-6 for v in errors.values())
        require(case.get("native_status") == ("PASS" if native_pass else "FAIL"), kind + ": native verdict inconsistent")
        if native:
            require(case.get("native_status") == "PASS", kind + ": native workflow failure")
            require(all(v <= 4e-6 for v in calculated.values()), kind + ": native measured tolerance")
            require(all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 4e-6 for v in errors.values()), kind + ": native tolerance")
    expected_native = "PASS" if all(c["native_status"] == "PASS" for c in receipt["cases"].values()) else "FAIL"
    require(receipt.get("native_status") == expected_native, "native aggregate verdict")
    return True


if __name__ == "__main__":
    native = "--native" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--native"]
    if len(args) != 5:
        raise SystemExit("usage: check_joint_bridge_public_receipt.py PUBLIC_JSON REFERENCE_TOML DIRECT_TOML R_ROOT JULIA_ROOT [--native]")
    receipt_path, ref_path, direct_path, rroot, jroot = map(Path, args)
    receipt = json.loads(receipt_path.read_text())
    reference = tomllib.loads(ref_path.read_text())
    direct = tomllib.loads(direct_path.read_text())
    uncertainty = jroot / "test/fixtures/joint_missing_predictor/native_uncertainty.toml"
    oracle.check(ref_path, uncertainty, direct, jroot, native_theta=False)
    check(receipt, reference, direct, rroot, jroot, native=native)
    print("JOINT_PUBLIC_ORACLE_PASS cases=2 rows=320 native_checked=" + str(native).lower())
