#!/usr/bin/env python3
"""Reconcile the two parity evidence stores against each other.

DRModels.jl records R-parity in two places that have never been compared:

  Route 1  `test/parity/fixtures/<slug>/expected.toml` — committed drmTMB
           outputs, replayed offline by `runparity.jl` with no R present.
  Route 2  `docs/dev-log/evidence/parity-*.tsv` — live `engine="tmb"` vs
           `engine="julia"` measurements, run by hand from `tools/parity_*.R`.

Both touch overlapping capabilities (Gaussian location-scale is in both). Until
now nothing checked whether they agree, so they could drift apart silently and
no test would notice.

WHAT THIS CAN AND CANNOT CHECK -- read before trusting the output.

It does NOT compare logLik or coefficients across the two routes, because they
do not fit the same data. `gaussian-locscale` is seed 20260604 at n=180;
Route 2's `base_gaussian_location_scale` is seed 20260814 at n=120. Comparing
their numbers would be a category error dressed up as a check, and it would
"pass" or "fail" for reasons that mean nothing.

What it checks instead is structural, and that is where the real risk lives:
  1. COVERAGE     -- a capability with evidence on one route and none on the other.
  2. VERSION SKEW -- a fixture frozen at an older drmTMB than the live measurement.
  3. STATUS       -- a capability whose two routes disagree about whether it passes.
  4. LINKABILITY  -- see below; this is the finding that motivated the tool.

THE LINK IS MANUAL, AND THAT IS THE POINT. Fixture metadata carries no
`capability_id` field, so nothing mechanically ties a fixture slug to the
capability row it is evidence for. The mapping below is therefore hand-written
and auditable rather than derived. Every entry is a claim. An unmapped fixture
is reported, never silently dropped -- silence is how a store drifts.

    python3 tools/parity_crosscheck.py
    python3 tools/parity_crosscheck.py --inject-contradiction   # positive control
"""

import argparse
import glob
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(REPO, "test", "parity", "fixtures")
EVIDENCE = os.path.join(REPO, "docs", "dev-log", "evidence")

# Fixture slug -> drmTMB capability_id. HAND-WRITTEN: fixture metadata has no
# capability_id field. Each line is a claim that this fixture is evidence for
# that capability row. Add a slug here only if that is actually true.
SLUG_TO_CAPABILITY = {
    "gaussian-locscale":        "base_gaussian_location_scale",
    "gaussian-bivariate-rho12": "biv_gaussian_residual",
    "binomial-trials":          "plain_binomial_nonphylo",
    # Deliberately unmapped, with reasons -- these are family-level fixtures that
    # do not correspond to a capability ROW in drmTMB's julia-capabilities.tsv:
    #   count-poisson, count-nbinom2, positive-gamma, positive-lognormal,
    #   proportion-beta, robust-student, nbinom2-dispersion, meta-analysis-V
    #   -> family coverage, not a bridge capability row.
    #   xfam-external-gllvm -> skipped by both runners; not drmTMB evidence.
}

PASS_TOKENS = {"PARITY_PASS", "SE_PASS", "PASS"}
NEUTRAL_TOKENS = {"NO_NATIVE_COMPARATOR", "BOUNDARY_NOT_COMPARABLE", "NEGATIVE_CONTROL_OK"}


def read_tsv(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        cells = ln.split("\t")
        if len(cells) < 2:
            continue  # a wrapped multi-line `note` field, not a row
        rows.append(dict(zip(header, cells)))
    return rows


def load_route2():
    """capability_id -> list of (tsv, cell_id, status)."""
    out = {}
    for path in sorted(glob.glob(os.path.join(EVIDENCE, "parity-*.tsv"))):
        for row in read_tsv(path):
            cap = row.get("capability_id") or row.get("cell_id")
            if not cap:
                continue
            status = (row.get("status") or "").strip()
            cell = (row.get("cell_id") or row.get("label") or "").strip()
            out.setdefault(cap, []).append((os.path.basename(path), cell, status))
    return out


def load_route1():
    """slug -> {version, seed, family, n}."""
    out = {}
    for meta in sorted(glob.glob(os.path.join(FIXTURES, "*", "expected.meta.toml"))):
        slug = os.path.basename(os.path.dirname(meta))
        text = open(meta).read()

        def field(name):
            m = re.search(rf'^{name}\s*=\s*"?([^"\n]+)"?', text, re.M)
            return m.group(1).strip() if m else None

        exp = os.path.join(os.path.dirname(meta), "expected.toml")
        n = family = None
        if os.path.exists(exp):
            etext = open(exp).read()
            mf = re.search(r'^family\s*=\s*"([^"]+)"', etext, re.M)
            mn = re.search(r"^n\s*=\s*([0-9]+)", etext, re.M)
            family = mf.group(1) if mf else None
            n = mn.group(1) if mn else None
        out[slug] = {
            "version": field("drmtmb_version"),
            "seed": field("seed"),
            "family": family,
            "n": n,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-version", default="0.7.0",
                    help="the drmTMB version Route 2 was measured against")
    ap.add_argument("--inject-contradiction", action="store_true",
                    help="positive control: fabricate a disagreement and prove "
                         "this tool reports it. Touches no files.")
    args = ap.parse_args()

    route2 = load_route2()
    route1 = load_route1()

    if args.inject_contradiction:
        route2["base_gaussian_location_scale"] = [
            ("parity-fixtures.tsv", "INJECTED", "PARITY_FAIL")
        ]

    problems = []
    print("ROUTE 1 (committed fixtures): %d" % len(route1))
    print("ROUTE 2 (live evidence rows): %d capability ids" % len(route2))
    print()

    # 1 + 3. Linked pairs: coverage confirmed, and status must not contradict.
    print("LINKED (fixture <-> capability row)")
    for slug, cap in sorted(SLUG_TO_CAPABILITY.items()):
        f = route1.get(slug)
        rows = route2.get(cap)
        if f is None:
            problems.append("mapped fixture %r does not exist on disk" % slug)
            print("  %-26s -> %-32s MISSING FIXTURE" % (slug, cap))
            continue
        if not rows:
            problems.append("fixture %r maps to %r, which has no Route-2 evidence" % (slug, cap))
            print("  %-26s -> %-32s NO LIVE EVIDENCE" % (slug, cap))
            continue
        bad = [r for r in rows if r[2] not in PASS_TOKENS and r[2] not in NEUTRAL_TOKENS]
        verdict = "ok" if not bad else "STATUS DISAGREEMENT: %s" % ", ".join(r[2] for r in bad)
        if bad:
            problems.append("%s: Route 2 reports %s while a committed Route-1 fixture "
                            "asserts this capability passes" % (cap, [r[2] for r in bad]))
        print("  %-26s -> %-32s %s" % (slug, cap, verdict))
    print()

    # 2. Version skew -- a fixture frozen older than the live measurement.
    print("VERSION SKEW (fixture drmTMB vs live %s)" % args.live_version)
    skew = [(s, v["version"]) for s, v in sorted(route1.items())
            if v["version"] and v["version"] != args.live_version]
    for slug, ver in skew:
        print("  %-26s %s" % (slug, ver))
    if not skew:
        print("  none")
    print()

    # 4. Linkability -- unmapped on either side.
    unmapped = sorted(set(route1) - set(SLUG_TO_CAPABILITY))
    print("UNMAPPED FIXTURES (%d) -- no capability_id in metadata, so the link "
          "cannot be derived" % len(unmapped))
    for slug in unmapped:
        print("  %-26s family=%s" % (slug, route1[slug]["family"]))
    print()

    # `drmjl_only:*` marks a DRModels.jl cell with no counterpart row in drmTMB's
    # julia-capabilities.tsv. It is real evidence, just not evidence FOR a
    # capability row, so it is not an unmatched id.
    uncovered = sorted(c for c in set(route2) - set(SLUG_TO_CAPABILITY.values())
                       if not c.startswith("drmjl_only:"))
    drmjl_only = sorted(c for c in route2 if c.startswith("drmjl_only:"))
    print("ROUTE-2 CAPABILITIES WITH NO MAPPED FIXTURE (%d)" % len(uncovered))
    for cap in uncovered:
        print("  %-40s %s" % (cap, route2[cap][0][0]))
    print()

    if drmjl_only:
        print("DRModels.jl-ONLY CELLS (%d) -- real evidence, no drmTMB capability row"
              % len(drmjl_only))
        for cap in drmjl_only:
            print("  %-40s %s" % (cap, route2[cap][0][0]))
        print()

    if problems:
        print("CROSSCHECK: FAIL")
        for p in problems:
            print("  - %s" % p)
        return 1
    print("CROSSCHECK: PASS -- no status contradiction between the two stores.")
    if skew:
        print("  NOTE: %d fixture(s) are frozen at an older drmTMB than the live "
              "measurement. That is staleness, not a contradiction; regenerating "
              "them is what closes it." % len(skew))
    print("  NOTE: this tool does NOT compare numbers across routes -- the two "
          "routes fit different data (different seeds and n), so a numeric "
          "comparison would be meaningless. Coverage, version and status only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
