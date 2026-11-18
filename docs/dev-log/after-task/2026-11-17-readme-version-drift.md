# 2026-11-17 — README status line: drop the hard-coded version (refs #1288)

Perspectives: **Ada** (slice), **Pat** (reader-facing README), **Rose** (claim vs evidence). No subagents.

## What changed

`README.md` status callout said **version 0.7.1** while `Project.toml` on
`origin/main` (`6e79fb570`) is **0.8.2**. That is the same class as closed
#678 (README 0.1.2 while the tree was 0.7.0): a version bump does not touch
the README line, so it drifts again. The 16 Nov comment on #1288 recorded the
same 0.7.1 vs 0.8.2 gap.

The line now says **experimental (0.x)** and points at
[`Project.toml`](../../../Project.toml) and [`NEWS.md`](../../../NEWS.md)
instead of a number. `CITATION.cff` already matches `Project.toml` (`0.8.2`);
this PR does not retouch it.

## Why no automated check

#1288 offered a README/Project.toml test as an alternative. This repo has no
existing check of that class (Rose's Workflow F is a pre-tag audit, not a CI
job; `tools/tests/test_docs_dev_links.jl` only guards `/dev` tutorial URLs).
The slice instruction was: extend an existing drift check only if that is
trivial; do not build new CI; no test changes. Dropping the number is the
durable fix.

## Verification

- `Project.toml` line 4: `version = "0.8.2"`.
- `CITATION.cff` line 19: `version: 0.8.2`.
- README status line: no `0.7.1` / `0.8.2`; links `Project.toml` and `NEWS.md`.
- `git diff --check` on the working tree (CI prose-only fast path).
- No `src/`, `test/`, tolerance, or timeout files touched.

## Rose

**Claim:** the public README no longer advertises a package version that can
disagree with `Project.toml`. **Evidence:** the status line contains no
`x.y.z`; the tree version remains in `Project.toml`. **Scope honesty:** docs
only; not a release, not a registry claim, not a CITATION bump. **License:**
no drmTMB source. **Drift:** #678 was a one-off number edit (`bb7ea14e8`
later set 0.7.2, then 0.7.1 remained through 0.8.0–0.8.2); this wording
cannot recur that way.

Does not close #1288.
