"""Validate historical diagnostic receipts; never certify a profile interval."""
import copy
import math
import pathlib
import sys
import tomllib


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(d):
    require(d.get("status") == "MEASURED", "diagnostic did not complete")
    require(d.get("source_unchanged") is True, "source changed")
    require(d.get("source_before") == d.get("source_after"), "source manifests differ")
    require(bool(d.get("source_before")), "missing source manifest")
    require(d.get("fit_converged") is True, "reference fit did not converge")
    require(type(d.get("optimizer_converged")) is bool, "missing optimizer status")
    require(d.get("stored_gradient") is False and d.get("profile_derivative_mode") == "finite",
            "not the measured finite-difference route")
    for key in ("parameters", "species", "observations", "optimizer_iterations",
                "objective_calls", "optimizer_gradient_calls"):
        require(type(d.get(key)) is int and d[key] > 0, f"invalid {key}")
    require(d["parameters"] == 15 and d["observations"] == 2*d["species"], "wrong fixture shape")
    require(d["objective_calls"] == (2*(d["parameters"]-1)+1)*d["optimizer_gradient_calls"]+1,
            "finite-difference call accounting disagrees")
    for key in ("dense_nll_error", "constrained_seconds", "constrained_bytes",
                "warm_objective_seconds_median", "warm_objective_bytes_median", "nuisance_score_maxabs"):
        require(isinstance(d.get(key), (int, float)) and math.isfinite(d[key]) and d[key] >= 0,
                f"invalid {key}")
    require(d["dense_nll_error"] <= 1e-7, "independent Gaussian objective disagrees")
    require(d.get("threads") == 1 and d.get("blas") == 1, "wrong resource budget")
    # Optimizer failure is retained and reported, NOT turned into passing inference.
    return not d["optimizer_converged"]


def main():
    paths = [pathlib.Path(s) for s in sys.argv[1:] if s != "--damage"]
    require(bool(paths), "provide TOML receipts")
    failed = 0
    for p in paths:
        with p.open("rb") as f:
            d = tomllib.load(f)
        failed += validate(d)
    print(f"PROFILE_DIAGNOSTICS_VALID cases={len(paths)} nonconverged_solves={failed} inference_certified=false")
    if "--damage" in sys.argv:
        mutations = [
            lambda x: x.update(objective_calls=x["objective_calls"]+1),
            lambda x: x.update(dense_nll_error=1.0),
            lambda x: x.update(optimizer_converged="false"),
            lambda x: x.update(status="PASS"),
            lambda x: x.update(constrained_seconds=-1),
            lambda x: x["source_after"].update({"src/DRModels.jl": "damaged"}),
        ]
        for mutate in mutations:
            bad = copy.deepcopy(d)
            mutate(bad)
            try:
                validate(bad)
            except ValueError:
                continue
            raise ValueError("damaged diagnostic accepted")
        print(f"PROFILE_DIAGNOSTIC_DAMAGES_REJECTED {len(mutations)}")


if __name__ == "__main__":
    main()
