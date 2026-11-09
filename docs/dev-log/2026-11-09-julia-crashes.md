# Julia crashes from gap map v2: root causes and dispositions (2026-11-09)

Reader: whoever owns the R-Julia parity lane. Purpose: say what each of the 22
uncaught Julia exceptions in `docs/dev-log/2026-11-09-gap-map-v2.md` was, what
changed, and what was left alone and why. Base: `0bfcde2bb` (tip of PR #1272),
Julia 1.10.12 and 1.12. R reference: drmTMB 0.8.1 from `~/drm_campaign_20261108/Rlib`.

## Pre-registered thresholds (written before the fixes were run)

A cell that Julia now fits counts as agreeing with R when all of these hold:

- `loglik`: `abs(julia - r) <= 1e-4`. This is the gap-map v2 rule for "agree".
- `nobs`: equal.
- Every coefficient: `abs(julia - r) <= 1e-3`. The cells here are closed-form or
  well-conditioned fixed-effect fits, so 1e-3 is generous; it is not widened
  after the fact.

A cell that is turned into a refusal counts as done when `drm()` throws an
`ArgumentError` whose message names the cause and what to do, and a test pins
the exact message. A refusal is not a claim that R refuses too unless the table
says so.

R references: `test/fixtures/gap_v2_crashes/make_references.R` (one single-process
call on Totoro, `nice -n 10`, one thread) writes `r_reference.csv` next to it.
Data: `test/fixtures/gap_v2_crashes/data_v2.csv`, the gap-map v2 data
(md5 `ae47edf77c31bcd8c3b4c933e9201239`), so Julia and R read the same bytes.
