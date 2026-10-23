"""Verify a retained combined check against current inputs, without running Julia."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

root = Path(__file__).resolve().parent.parent

def expected_inputs():
    names = ["test_bridge_materialization_collision.jl", "test_bridge_formula_labels.jl",
             "test_bridge_lss_labels.jl", "test_bridge.jl",
             "test_bridge_formula_translation.jl", "test_bridge_profile_target.jl"]
    fixtures = ["native-numeric-labels.tsv", "native-nested-labels.tsv",
                "native-scalar-labels.tsv", "native-conditional-labels.tsv"]
    return set(map(str, [*root.joinpath("src").rglob("*.jl"),
        root / "Project.toml", root / "Manifest.toml",
        root / "tools/check_coefficient_labels.jl", root / "tools/run_coefficient_label_checks.py",
        root / "docs/src/r-julia-bridge.md", root / "test/runtests.jl",
        *(root / "test" / n for n in names),
        *(root / "docs/dev-log/evidence/julia-r-parity/coefficient-labels" / n for n in fixtures)]))

def require(condition, message="Invalid combined receipt"):
    if not condition:
        raise ValueError(message)

def validate(receipt, log):
    require(receipt["status"] == "PASS")
    require(type(receipt["exit"]) is int and receipt["exit"] == 0)
    require(receipt["timed_out"] is False)
    require(type(receipt["elapsed"]) in (int, float) and math.isfinite(receipt["elapsed"]) and 0 < receipt["elapsed"] < 180)
    require(receipt["cwd"] == str(root))
    require(receipt["command"] == ["julia", "--project=.", "tools/check_coefficient_labels.jl"])
    require(re.fullmatch(r"[0-9a-f]{40}", receipt["baseline_git"]))
    before = receipt["source_before"]
    require(before == receipt["source_after"])
    require(set(before) == expected_inputs())
    for path, digest in before.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path)
    require("COEFFICIENT_LABEL_COMBINED_PASS" in log)
    require(f"RUNTIME source={root}/src/DRModels.jl" in log)
    require("threads=1 BLAS=1" in log)
    return True

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("receipt", type=Path)
parser.add_argument("--self-test", action="store_true")
args = parser.parse_args()
receipt = json.loads(args.receipt.read_text())
log = args.receipt.with_suffix(".log").read_text()
validate(receipt, log)
print("COEFFICIENT_LABEL_COMBINED_RECEIPT_PASS", len(receipt["source_before"]), "current inputs")
if args.self_test:
    cases = []
    for key, value in [("status", "FAIL"), ("exit", 1), ("timed_out", True), ("elapsed", -1), ("elapsed", float("nan")),
                       ("cwd", "/wrong"), ("command", ["julia", "other.jl"])]:
        damaged = copy.deepcopy(receipt); damaged[key] = value; cases.append((damaged, log))
    critical = str(root / "src/bridge.jl")
    damaged = copy.deepcopy(receipt)
    for key in ("source_before", "source_after"): del damaged[key][critical]
    cases.append((damaged, log))
    damaged = copy.deepcopy(receipt)
    for key in ("source_before", "source_after"): damaged[key][critical] = "0" * 64
    cases.append((damaged, log))
    cases.append((receipt, log.replace("COEFFICIENT_LABEL_COMBINED_PASS", "REMOVED")))
    cases.append((receipt, log.replace("threads=1 BLAS=1", "threads=8 BLAS=8")))
    for damaged, output in cases:
        try:
            validate(damaged, output)
        except (ValueError, KeyError):
            continue
        raise ValueError("Damaged receipt was accepted")
    print("COEFFICIENT_LABEL_COMBINED_SELFTEST_PASS:11 damaged receipts rejected")
