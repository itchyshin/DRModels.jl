#!/usr/bin/env python3
"""Verify that every test file the PUBLIC capability page cites as evidence actually runs.

Why this exists
---------------
`docs/src/capabilities.md` is what a user reads to decide whether DRModels.jl does the
thing they need, and its Status column cites test files as the evidence. Two ways
that can be false, both found on 2026-08-25:

  * A capability is declared **absent while it ships** — the cross-family bivariate
    model was listed "Absent -- no cross-family bivariate model is implemented"
    while `src/mixed_family.jl`, three wired test files and a ~450-line methods
    guide were all on main (DRModels.jl#490). A user went elsewhere past a working
    feature. That direction is not mechanically checkable and is not what this
    guard does.

  * A capability cites a test that **never runs** — `test_q4_laplace.jl` was cited
    as evidence for the q=4 sparse-Laplace row while being deliberately unwired
    (it exercises the `bench/` proof-of-concept, not the `src/` file the row
    names, and its gradient role is superseded). That direction IS mechanically
    checkable, and this guard checks it.

The second matters beyond tidiness: public documentation is one of the four limbs
`docs/design/168` requires before a capability row may be called `covered`. A cited
test that never executes is a limb resting on nothing.

Prove-or-skip. Only a cited test path that EXISTS but is absent from
`runtests.jl`'s include set is reported. `runtests.jl` citing itself is not a
finding. Both `test/test_foo.jl` and a bare `test_foo.jl` count as citations, so
dropping the directory prefix is not an escape hatch. A note EXPLAINING why some
test is not cited must therefore avoid backticking its filename -- that archaeology
belongs in the dev-log, not on the page a user reads to choose a package.
Exit 1 on any finding.

Usage:
  python3 tools/check_doc_test_citations.py [docs/src/capabilities.md ...]
"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtests_files import wired_test_files  # noqa: E402

RUNNER = "test/runtests.jl"

def included_basenames(runner):
    src = open(runner, encoding="utf-8").read()
    # strip comments so a commented-out include() is not counted as wired
    code = "\n".join(l.split("#", 1)[0] for l in src.splitlines())
    literal = {os.path.basename(m) for m in re.findall(r'include\(\s*"([^"]+)"\s*\)', code)}
    # runtests.jl auto-discovers test_*.jl (see runtests_files.py)
    return literal | wired_test_files(os.path.dirname(runner) or ".")

def main(argv):
    docs = argv[1:] or ["docs/src/capabilities.md"]
    if not os.path.exists(RUNNER):
        print(f"SKIP -- {RUNNER} not found (wrong working directory?)")
        return 0
    wired = included_basenames(RUNNER)
    findings, checked = [], 0
    for doc in docs:
        if not os.path.exists(doc):
            print(f"SKIP -- {doc} not found")
            continue
        text = open(doc, encoding="utf-8").read()
        for cite in sorted(set(re.findall(r'`(test/[A-Za-z0-9_./-]+\.jl)`', text))):
            if os.path.basename(cite) == "runtests.jl":
                continue                      # the runner naming itself is not evidence
            if not os.path.exists(cite):
                continue                      # a stale path is a different guard's job
            checked += 1
            if os.path.basename(cite) not in wired:
                findings.append((doc, cite))
    if findings:
        print("PUBLIC DOCS CITE TESTS THAT NEVER RUN\n")
        for doc, cite in findings:
            print(f"  {doc}\n    cites {cite} as evidence, but it is not included by {RUNNER}.")
        print("\nEither wire the test, or stop citing it. A capability row's documentation limb")
        print("must not rest on a file that never executes (docs/design/168).")
        return 1
    print(f"OK -- {checked} test citation(s) in {len(docs)} doc(s) all resolve to tests that actually run.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv))
