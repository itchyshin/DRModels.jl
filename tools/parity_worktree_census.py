#!/usr/bin/env python3
"""Collect and verify a metadata-only census of DRModels parity worktrees."""
import argparse, copy, json, os, re, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

REPOS = [Path("/Users/z3437171/Dropbox/Github Local/DRM.jl"),
         Path("/Users/z3437171/Dropbox/Github Local/drmTMB")]
PINNED = {"DRM.jl": "f47789646f27221ba4fad29a8ba1b3b8a790b521", "drmTMB": "b35642b4560072cadba7e595e66e00209ebdeb40"}
SCHEMA = "drm-julia-r-parity/worktree-census/v2"

def run(repo, *args):
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], encoding="utf-8", errors="surrogateescape",
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=float(os.environ.get("CENSUS_TIMEOUT_SECONDS", "30")))
    except subprocess.TimeoutExpired as exc:
        return 124, "", f"command timed out after {os.environ.get('CENSUS_TIMEOUT_SECONDS', '30')} seconds"
    return p.returncode, p.stdout, p.stderr

def sha(repo, ref):
    rc, out, err = run(repo, "rev-parse", ref)
    return out.strip() if rc == 0 else None

def common_dir(repo):
    rc, out, err = run(repo, "rev-parse", "--git-common-dir")
    return str((Path(repo) / out.rstrip("\n")).resolve()) if rc == 0 else None

def status(repo):
    rc, out, err = run(repo, "status", "--porcelain", "-z", "--untracked-files=all")
    if rc:
        return None, err.strip()
    names = []
    # Porcelain -z has a rename pair; preserving both names is safest.
    fields = out.split("\0")
    i = 0
    while i < len(fields) and fields[i]:
        rec = fields[i]; names.append(rec)
        if len(rec) >= 2 and any(c in rec[:2] for c in "RC"):
            if i + 1 < len(fields) and fields[i + 1]:
                names.append(fields[i + 1]); i += 1
        i += 1
    return names, None

def collect(repos=REPOS, pins=PINNED):
    now = datetime.now(timezone.utc).isoformat()
    data = {"schema": SCHEMA, "collected_at": now, "repos": [],
            "entries": [], "errors": [], "totals": {"registered": 0, "usable": 0,
            "missing": 0, "brokenlink": 0, "dirty": 0, "stashes": 0}}
    for repo in repos:
        rr = {"path": str(repo), "observed_checkout_sha": sha(repo, "HEAD"),
              "pinned_main_sha": pins.get(repo.name), "common_dir": common_dir(repo), "worktree_command": None,
              "stash_command": None}
        rc, out, err = run(repo, "worktree", "list", "--porcelain", "-z")
        rr["worktree_command"] = {"returncode": rc, "stderr": err.strip()}
        if rc:
            data["errors"].append({"repo": str(repo), "operation": "worktree list", "error": err.strip()})
        else:
            path = None; rec = {}
            for line in out.split("\0") + [""]:
                if line.startswith("worktree "):
                    path = line[9:]; rec = {"repo": str(repo), "kind": "worktree", "path": path}
                elif line.startswith("HEAD "):
                    rec["head_sha"] = line[5:]
                elif line.startswith("branch "):
                    rec["branch"] = line[7:]
                elif not line and path:
                    p = Path(path); rec.setdefault("head_sha", sha(repo, "HEAD"))
                    if not p.exists(): rec["path_state"] = "brokenlink" if p.is_symlink() else "missing"
                    else:
                        actual_common = common_dir(p)
                        rec["observed_common_dir"] = actual_common
                        rec["path_state"] = "usable" if actual_common and actual_common == rr["common_dir"] else "brokenlink"
                        if rec["path_state"] == "brokenlink": rec["gitlink_error"] = "missing or foreign Git linkage"
                    st, se = status(p) if rec["path_state"] == "usable" else (None, "path unavailable or broken")
                    rec["dirty_paths"] = st or []
                    rec["dirty"] = (bool(st) if st is not None else None)
                    if se: rec["status_error"] = se; data["errors"].append({"path": path, "operation": "status", "error": se})
                    rec["relation_to_pinned_main"] = ("pinned" if rec.get("head_sha") == rr["pinned_main_sha"] else "different") if rec.get("head_sha") else "unknown"
                    data["entries"].append(rec); path = None
        rc, out, err = run(repo, "stash", "list", "--format=%H%x09%gd%x09%gs")
        rr["stash_command"] = {"returncode": rc, "stderr": err.strip()}
        if rc: data["errors"].append({"repo": str(repo), "operation": "stash list", "error": err.strip()})
        else:
            for line in out.splitlines():
                bits = line.split("\t", 2)
                data["entries"].append({"repo": str(repo), "kind": "stash", "path": f"{repo}::{bits[1] if len(bits)>1 else line}", "stash_sha": bits[0], "ref": bits[1] if len(bits)>1 else None, "message": bits[2] if len(bits)>2 else ""})
        data["repos"].append(rr)
    data["totals"]["worktrees"] = sum(e["kind"] == "worktree" for e in data["entries"])
    # `registered` is deliberately the worktree denominator; stashes are a separate class.
    data["totals"]["registered"] = data["totals"]["worktrees"]
    data["totals"]["stashes"] = sum(e["kind"] == "stash" for e in data["entries"])
    data["totals"]["usable"] = sum(e.get("path_state") == "usable" for e in data["entries"])
    data["totals"]["missing"] = sum(e.get("path_state") == "missing" for e in data["entries"])
    data["totals"]["brokenlink"] = sum(e.get("path_state") == "brokenlink" for e in data["entries"])
    data["totals"]["dirty"] = sum(e.get("dirty") is True for e in data["entries"])
    return data

def verify(d, expected_repos=REPOS):
    if not isinstance(d, dict) or d.get("schema") != SCHEMA: return False
    es = d.get("entries"); t = d.get("totals")
    if not isinstance(es, list) or not isinstance(t, dict) or d.get("errors") is None: return False
    if len(d.get("repos", [])) != len(expected_repos) or any(r.get("worktree_command", {}).get("returncode") != 0 for r in d["repos"]): return False
    if {r.get("path") for r in d["repos"]} != {str(p) for p in expected_repos}: return False
    if any(not re.fullmatch('[0-9a-f]{40}', r.get(k, '') or '') for r in d['repos'] for k in ('pinned_main_sha', 'observed_checkout_sha')): return False
    repo_map = {r['path']: r for r in d['repos']}
    if any(e.get('repo') not in repo_map for e in es): return False
    paths = [e.get("path") for e in es if isinstance(e, dict)]
    if len(paths) != len(es) or any(not p for p in paths) or len(paths) != len(set(paths)): return False
    if t.get("registered") != t.get("worktrees") or t.get("worktrees") != sum(e.get("kind")=="worktree" for e in es): return False
    for k, pred in [("stashes", lambda e:e.get("kind")=="stash"), ("usable", lambda e:e.get("path_state")=="usable"), ("missing", lambda e:e.get("path_state")=="missing"), ("brokenlink", lambda e:e.get("path_state")=="brokenlink"), ("dirty", lambda e:e.get("dirty") is True)]:
        if t.get(k) != sum(pred(e) for e in es): return False
    if any(e.get("kind") not in ("worktree", "stash") for e in es): return False
    if any(e.get("kind") == "worktree" and e.get("path_state") not in ("usable", "missing", "brokenlink") for e in es): return False
    if any(e.get("kind") == "worktree" and e.get("path_state") == "usable" and e.get("dirty") is None for e in es): return False
    if any(e.get('path_state') == 'usable' and (e.get('status_error') or not e.get('observed_common_dir') or e['observed_common_dir'] != repo_map[e['repo']]['common_dir'] or type(e.get('dirty')) is not bool or e['dirty'] != bool(e.get('dirty_paths'))) for e in es): return False
    if any(r.get("stash_command", {}).get("returncode") != 0 for r in d["repos"]): return False
    if t.get("worktrees") != t.get("usable", 0) + t.get("missing", 0) + t.get("brokenlink", 0): return False
    return True

def self_test():
    with tempfile.TemporaryDirectory(prefix='parity-census-test-') as tmp:
        base = Path(tmp).resolve(); repos = [base / 'a', base / 'b']
        def git(repo, *args):
            rc, out, err = run(repo, *args)
            assert rc == 0, err
        for repo in repos:
            repo.mkdir(); git(repo, 'init', '-q')
            (repo / 'original').write_text('retained\n'); git(repo, 'add', 'original')
            git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        pins = {p.name: sha(p, 'HEAD') for p in repos}
        dirty, missing, wrong = base / 'with\nnewline', base / 'missing', base / 'wrong'
        for p in (dirty, missing, wrong): git(repos[0], 'worktree', 'add', '--detach', str(p))
        git(dirty, 'mv', 'original', 'renamed\nfile'); (dirty / 'untracked').write_text('x')
        missing.rename(base / 'preserved')
        (wrong / '.git').write_text(f'gitdir: {repos[1] / ".git"}\n')
        d = collect(repos, pins); assert verify(d, repos)
        assert d['totals'] == {'registered': 5, 'worktrees': 5, 'stashes': 0, 'usable': 3, 'missing': 1, 'brokenlink': 1, 'dirty': 1}
        row = next(e for e in d['entries'] if e['path'] == str(dirty))
        assert 'original' in row['dirty_paths'] and any('renamed\nfile' in s for s in row['dirty_paths'])
        for mutation in ('count', 'stash', 'status', 'link', 'head'):
            bad = copy.deepcopy(d)
            if mutation == 'count': bad['totals']['registered'] += 1
            if mutation == 'stash': bad['repos'][0]['stash_command']['returncode'] = 1
            if mutation == 'status': bad['entries'][0]['dirty'] = None
            if mutation == 'link': bad['entries'][0]['observed_common_dir'] = '/wrong'
            if mutation == 'head': bad['repos'][0]['observed_checkout_sha'] = 'z' * 40
            assert not verify(bad, repos), mutation
        assert not verify({}, repos)
    print('WORKTREE_CENSUS_SELF_TEST_OK')

def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect"); c.add_argument("--output", "-o", required=True)
    v = sp.add_parser("verify"); v.add_argument("--input", required=True)
    sp.add_parser("self-test")
    a = ap.parse_args()
    if a.cmd == "collect":
        Path(a.output).parent.mkdir(parents=True, exist_ok=True); Path(a.output).write_text(json.dumps(collect(), indent=2) + "\n"); return
    if a.cmd == "self-test":
        self_test(); return
    try: d = json.loads(Path(a.input).read_text())
    except Exception: d = None
    if verify(d): print("WORKTREE_CENSUS_VALID"); return
    print("WORKTREE_CENSUS_INVALID"); sys.exit(1)
if __name__ == "__main__": main()
