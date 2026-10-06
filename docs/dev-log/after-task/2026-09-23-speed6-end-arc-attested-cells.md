# After-task: speed6 land #781 + end-of-arc attested cells (2026-09-23)

## Scope

Cursor Grok lane on `~/local-scratch/lanes/DRM.jl-speed6-20260919`.
Lease: `cursor:DRM.jl-speed6:speed6-cells-20260923`. Did not touch reader-doc PRs.

## Outcome

1. **PR [#781](https://github.com/itchyshin/DRModels.jl/pull/781) MERGED** at
   `cf058168b` (2026-09-23T11:38:48Z) after `pr_merge_when_green.sh` verified
   every check COMPLETED SUCCESS/SKIPPED on head `45ff42de8`.
2. **Detached HEAD repaired:** checked out `claude/lane-speed6-20260919` at the
   remote tip, then fast-forwarded through the merge; receipts branch cut from
   `origin/main`.
3. **End-of-arc walls:** re-timed #372 six + #389 plus5 bridge cohorts Julia vs
   drmTMB **0.7.1** on the merged tip. **10 paired OK cells**, median **18.3×**,
   range **7.2×–55.2×**. One R fail (`proportion-beta`, missing `a`).
4. **q4 phylo fit wall** on tip: p=100 **0.976 s**, p=1000 **10.375 s**,
   chol fallbacks **0**. Compared to banked S3: **honest identity / no wall gain**
   from speed6's reuse path on this grid.

## Evidence

`docs/dev-log/evidence/2026-09-23-speed6-end-arc-cells/` (README + TOML/JSON/TSV).

## Rose

- Numbers only from this session's script outputs and banked S3 TSV headers.
- No public README/NEWS speed claim in this slice.
- Speed6 src work is not sold as a fit-wall win; bridge re-time is the attested
  multi-cell speedup set.

## Checks run

- `gh api …/check-runs` on `45ff42de8` (all SUCCESS/SKIPPED)
- `~/shinichi-brain/tools/pr_merge_when_green.sh itchyshin/DRModels.jl 781`
- Bridge Julia+R six and plus5; `profile_q4_sections.jl --gate tsv --p 100,1000`

## Follow-ups

- Optional: Totoro re-time of the same 10 cells (D-50; not required for the bar).
- Optional: repair R `proportion-beta` timing arm (fixture / drmTMB 0.7.1 API).
- Beta-block / Gst assembly section work remains the next measured src target if
  a q4 wall gain is wanted.
