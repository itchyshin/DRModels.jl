import subprocess, pathlib, os
E = "docs/dev-log/evidence/julia-r-parity"
RROOT = os.path.expanduser("~/claude-mr/1442/main")
fit = "tools/check_finite_fit_receipt.py"
pub = "tools/check_finite_public_receipt.py"
muts = {k:v for k,v in {
    "A_fit_always_pass": (fit, "passed=all(e<=4e-6 for e in errors.values())", "passed=True"),
    "B_fit_no_verdict_check": (fit, "require(v.get('parity_pass') is passed,'parity verdict')", "pass"),
    "C_fit_no_reported_error_check": (fit, "near(v['errors'][key],error,1e-10,'reported error '+key)", "pass"),
    "D_pub_always_pass": (pub, "native='PASS' if all(e<=4e-6 for e in errors.values()) else 'FAIL'", "native='PASS'"),
    "E_pub_no_verdict_check": (pub, "require(v.get('native_status')==native,'honest native verdict')", "pass"),
    "J_fit_tol_5e-6": (fit, "passed=all(e<=4e-6 for e in errors.values())", "passed=all(e<=5e-6 for e in errors.values())"),
    "K_pub_tol_5e-6": (pub, "native='PASS' if all(e<=4e-6 for e in errors.values()) else 'FAIL'", "native='PASS' if all(e<=5e-6 for e in errors.values()) else 'FAIL'"),
}.items() if k[0] in "JK"}
cmds = {fit: ["python3", "tools/test_finite_fit_receipt.py"],
        pub: ["python3", pub, E + "/finite-frontends/finite-public-" + os.environ.get("PUBN","005") + ".json", RROOT, "--damage"]}
for name, (f, a, b) in muts.items():
    p = pathlib.Path(f); orig = p.read_text(); assert orig.count(a) == 1, (name, orig.count(a))
    p.write_text(orig.replace(a, b))
    try:
        r = subprocess.run(cmds[f], capture_output=True, text=True)
        last = (r.stdout + r.stderr).strip().splitlines()[-1]
        verdict = "KILLED" if r.returncode else "SURVIVED"
        print(name, "rc=", r.returncode, verdict, "::", last[:120])
    finally:
        p.write_text(orig)
# Same-class probe (not in the review): joint public bridge verdict.
jb = "tools/check_joint_bridge_public_receipt.py"
REF = "test/fixtures/joint_missing_predictor/native_reference.toml"
jcmd = ["python3", "tools/test_joint_bridge_public_receipt.py", E + "/joint-bridge/joint-public-003.json", REF, E + "/joint-bridge/joint-direct-bridge-002.toml", RROOT, "."]
line = "native_pass = all(isinstance(v, (int, float)) and math.isfinite(v) and v <= 4e-6 for v in errors.values())"
for name, b in {"L_joint_tol_5e-6": line.replace("4e-6", "5e-6")}.items():
    p = pathlib.Path(jb); orig = p.read_text(); assert orig.count(line) == 1
    p.write_text(orig.replace(line, b))
    try:
        r = subprocess.run(jcmd, capture_output=True, text=True)
        last = (r.stdout + r.stderr).strip().splitlines()[-1]
        print(name, "rc=", r.returncode, "KILLED" if r.returncode else "SURVIVED", "::", last[:120])
    finally:
        p.write_text(orig)
