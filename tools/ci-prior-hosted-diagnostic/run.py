import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
out = root / "diagnostic-results"
out.mkdir(exist_ok=True)
script = Path(__file__).with_name("run.jl")
records = []
for mode in ("prefix",):
    for label in ("baseline", "candidate"):
        route = root / label
        victim = route / "test" / "test_binomial_sd_public.jl"
        pristine = victim.read_bytes()
        assert not subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "test/test_binomial_sd_public.jl"], cwd=route)
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=route, text=True).strip()
        env = os.environ.copy()
        env.update(DRM_TEST_SHARD="3/4", DRM_PARITY_TESTS="0")
        cmd = ["julia", "--startup-file=no", "--threads=1", "--check-bounds=yes",
               "--depwarn=yes", "--compiled-modules=yes", f"--code-coverage=@{route}",
               "--color=yes", "--warn-overwrite=yes", "--inline=yes",
               "--track-allocation=none", f"--project={root / (label + '-env')}",
               str(script), str(root), label, mode]
        logfile = out / f"{label}-{mode}.log"
        started = time.time()
        print(f"START {label} {mode} source={source}", flush=True)
        try:
            with logfile.open("w") as stream:
                try:
                    process = subprocess.Popen(cmd, env=env, cwd=route, stdout=stream,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    code = process.wait(timeout=1170)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    code = 124
        finally:
            victim.write_bytes(pristine)
            assert victim.read_bytes() == pristine
        elapsed = time.time() - started
        record = dict(label=label, mode=mode, source=source, command=cmd,
                      exit_code=code, elapsed_seconds=elapsed,
                      process_exit_code=process.returncode,
                      timeout_policy="SIGINT process group at1170s,15s grace, then SIGKILL group",
                      log_sha256=hashlib.sha256(logfile.read_bytes()).hexdigest())
        records.append(record)
        (out / "results.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"FINISH {label} {mode} exit={code} seconds={elapsed:.1f}", flush=True)
        print("\n".join(logfile.read_text(errors="replace").splitlines()[-12:]), flush=True)
        production_diff = subprocess.check_output(
            ["git", "diff", "HEAD", "--", "src", "test/test_binomial_sd_public.jl"], cwd=route)
        assert not production_diff, "Production source changed during diagnosis"
        for name in ("Project.toml", "Manifest.toml"):
            (out / f"{label}-{name}").write_bytes((root / (label + "-env") / name).read_bytes())
sys.exit(0 if all(r["exit_code"] == 0 for r in records) else 1)
