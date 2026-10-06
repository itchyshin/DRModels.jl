#!/usr/bin/env python3
"""Detect stale `file.jl:LINE` citations in docs/design/capability-status.md.

Why this exists
---------------
capability-status.md is the Julia-side input to the R<->Julia parity board, and
its own status vocabulary makes citations load-bearing: a row is `implemented`
only if real code and a test that actually runs can be pointed at. But the
citations are *line numbers*, and a line number rots whenever a line is added
above it. On 2026-08-24 six had drifted -- one claiming to be the chi-bar-square
exports while pointing at `include("visualization.jl")`.

A stale citation does not merely inconvenience a reader. It makes the claim
unauditable, which for a parity ledger is indistinguishable from unsupported.

What it checks
--------------
Only what it can prove, so that a failure always means something. Three checks:

1. RANGE     -- every cited file exists and every cited line is within it.
2. INCLUDE   -- when the doc says a file is "included" at `src/X.jl:N` (or that a
                test "is in the default suite" at `test/runtests.jl:N`), the
                include line is recomputed from the source and compared to N.
                This is exact: the include statement is found by name, not
                guessed at, so a mismatch is real drift and the correct line is
                printed.
3. WIRED     -- any `test/test_*.jl` the doc claims is in the default suite is
                actually included by test/runtests.jl.

Citations it cannot resolve precisely -- docstring examples, ranges spanning
several unrelated includes -- are reported as SKIPPED, never as failures. A
checker that cries wolf trains people to ignore it.

Usage:
    python3 tools/check_capability_citations.py            # verify
    python3 tools/check_capability_citations.py --verbose  # also show skips/OKs

Exit codes: 0 = nothing provably stale, 1 = at least one provable failure.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtests_files import wired_test_files  # noqa: E402

DOC = "docs/design/capability-status.md"
RUNTESTS = "test/runtests.jl"

CITATION = re.compile(r"`([A-Za-z0-9_./-]+\.jl):([0-9]+(?:[-,][0-9]+)*)`")
# a nearby `something.jl` that could be the include target
NEARBY_JL = re.compile(r"`([A-Za-z0-9_./-]+\.jl)`")


def expand(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return out


def include_lines(path: str, lines: list[str]) -> dict[str, int]:
    """basename -> 1-based line of its `include("basename")` in `path`."""
    found: dict[str, int] = {}
    for i, line in enumerate(lines, start=1):
        # Strip a trailing Julia comment FIRST. Without this, a commented-out
        # `# include("test_x.jl")` still matches and the file is reported as wired --
        # which silently defeats the WIRED check. Caught by deliberately commenting
        # out an include and finding the guard stayed green.
        code = line.split("#", 1)[0]
        m = re.search(r'include\(\s*"([^"]+)"\s*\)', code)
        if m:
            found.setdefault(os.path.basename(m.group(1)), i)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--verbose", action="store_true", help="also report SKIPPED and OK citations")
    args = ap.parse_args()

    if not os.path.exists(DOC):
        print(f"ERROR: {DOC} not found -- run from the repository root.", file=sys.stderr)
        return 1

    text = open(DOC, encoding="utf-8").read()
    src_cache: dict[str, list[str]] = {}
    inc_cache: dict[str, dict[str, int]] = {}

    failures: list[str] = []
    skipped: list[str] = []
    ok = 0

    def load(path: str) -> list[str]:
        if path not in src_cache:
            src_cache[path] = open(path, encoding="utf-8").read().split("\n")
        return src_cache[path]

    def includes(path: str) -> dict[str, int]:
        if path not in inc_cache:
            inc_cache[path] = include_lines(path, load(path))
        return inc_cache[path]

    for m in CITATION.finditer(text):
        path, spec = m.group(1), m.group(2)
        doc_line = text[: m.start()].count("\n") + 1
        tag = f"{DOC}:{doc_line}  `{path}:{spec}`"

        # --- check 1: RANGE -------------------------------------------------
        if not os.path.exists(path):
            failures.append(f"{tag} -> FILE DOES NOT EXIST")
            continue
        lines = load(path)
        nums = expand(spec)
        out_of_range = [n for n in nums if not (0 < n <= len(lines))]
        if out_of_range:
            failures.append(f"{tag} -> LINE(S) {out_of_range} OUT OF RANGE ({path} has {len(lines)} lines)")
            continue

        # --- check 2: INCLUDE -----------------------------------------------
        # Resolve only the unambiguous case: exactly one cited line, and exactly
        # one nearby `foo.jl` that `path` actually includes.
        window = text[max(0, m.start() - 400) : m.end() + 400]
        candidates = {
            os.path.basename(j)
            for j in NEARBY_JL.findall(window)
            if os.path.basename(j) != os.path.basename(path)
        }
        table = includes(path)
        resolvable = sorted(candidates & table.keys())

        if len(nums) == 1 and len(resolvable) == 1:
            target = resolvable[0]
            actual = table[target]
            if actual != nums[0]:
                failures.append(
                    f"{tag} -> STALE: `include(\"{target}\")` is at {path}:{actual}, not {nums[0]}"
                )
            else:
                ok += 1
                if args.verbose:
                    print(f"OK   {tag} -> include(\"{target}\") at :{actual}")
            continue

        # A multi-line citation (`:296,374,376` or `:49-51`) that names exactly as many
        # includable files nearby is still provable: compare the two sorted sets. This
        # was originally skipped, and four genuinely stale citations hid in that gap --
        # the skip bucket was the majority of the file, so leaving it unchecked meant the
        # guard was mostly decorative.
        if len(resolvable) > 1 and len(resolvable) == len(nums):
            expected = sorted(table[t] for t in resolvable)
            got = sorted(nums)
            if expected != got:
                pairs = ", ".join(f'{t} -> :{table[t]}' for t in sorted(resolvable, key=lambda t: table[t]))
                failures.append(f"{tag} -> STALE: cited {got}, actual {expected}  ({pairs})")
            else:
                ok += 1
                if args.verbose:
                    print(f"OK   {tag} -> {len(resolvable)} include(s) at {expected}")
            continue

        # A contiguous range (`:172-178`) whose nearby names all fall inside it, and whose
        # endpoints are themselves includes, is verifiable as a span.
        if len(resolvable) >= 2 and "-" in spec and "," not in spec:
            lines_of = [table[t] for t in resolvable]
            lo, hi = min(nums), max(nums)
            outside = [t for t in resolvable if not (lo <= table[t] <= hi)]
            # Containment only. NOT tight-spanning: the prose may name a subset of the
            # includes the range covers, so demanding the endpoints match a partial set
            # produces false positives -- which this check did on its first outing.
            if outside:
                where = ", ".join(f'{t} -> :{table[t]}' for t in sorted(outside, key=lambda t: table[t]))
                failures.append(f"{tag} -> STALE: range {lo}-{hi} does not contain {where}")
            else:
                ok += 1
                if args.verbose:
                    print(f"OK   {tag} -> range {lo}-{hi} spans {len(resolvable)} include(s)")
            continue

        skipped.append(f"{tag} -> not precisely resolvable ({len(nums)} line(s), {len(resolvable)} include match(es))")

    # --- check 3: WIRED -----------------------------------------------------
    if os.path.exists(RUNTESTS):
        # runtests.jl auto-discovers test_*.jl (see runtests_files.py)
        wired = set(includes(RUNTESTS)) | wired_test_files(os.path.dirname(RUNTESTS))
        for tf in sorted(set(re.findall(r"`(?:test/)?(test_[A-Za-z0-9_]+\.jl)`", text))):
            if not os.path.exists(os.path.join("test", tf)):
                continue
            # only complain when the doc asserts default-suite membership near it
            for tm in re.finditer(re.escape(tf), text):
                near = text[max(0, tm.start() - 300) : tm.end() + 300]
                if re.search(r"in the default suite|default-suite", near, re.I):
                    if tf not in wired:
                        failures.append(
                            f"{DOC}: `{tf}` is described as being in the default suite, "
                            f"but {RUNTESTS} does not include it"
                        )
                    break

    if args.verbose and skipped:
        print(f"\nSKIPPED ({len(skipped)}) -- not provably wrong, not checked:")
        for s in skipped:
            print(f"  {s}")

    if failures:
        print(f"\nSTALE CITATIONS ({len(failures)}):\n")
        for f in failures:
            print(f"  {f}")
        print(
            f"\nFAIL -- {len(failures)} provable failure(s). "
            f"({ok} verified, {len(skipped)} not precisely resolvable.)\n"
            "A capability ledger whose citations do not resolve cannot be audited. Re-point them."
        )
        return 1

    print(f"OK -- {ok} citation(s) verified against the source, {len(skipped)} skipped as not precisely resolvable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
