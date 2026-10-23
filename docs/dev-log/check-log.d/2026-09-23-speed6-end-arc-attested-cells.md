# check-log · 2026-09-23 · speed6 end-arc attested cells

| date | slice | command | result |
|---|---|---|---|
| 2026-09-23 | speed6 land | `pr_merge_when_green.sh itchyshin/DRModels.jl 781` | MERGED `cf058168b`; checks SUCCESS/SKIPPED on `45ff42de8` |
| 2026-09-23 | bridge six | `DRM_BRIDGE_TIMING_COHORT=six` Julia + R arms, reps=5 | 5/6 paired OK; `proportion-beta` R FAIL |
| 2026-09-23 | bridge plus5 | `DRM_BRIDGE_TIMING_COHORT=plus5` Julia + R arms, reps=5 | 5/5 paired OK |
| 2026-09-23 | q4 sections | `profile_q4_sections.jl --gate tsv --p 100,1000` on tip | p100=0.976s p1000=10.375s; G5e.3 PASS; TSV `q4_sections_cf058168b.tsv` |
| 2026-09-23 | receipt | evidence dir `docs/dev-log/evidence/2026-09-23-speed6-end-arc-cells/` | 10 paired cells median 18.3× (7.2–55.2); q4 identity-no-gain vs S3 bank |

Host: `w-kw3k3y6229.psych.ualberta.ca`. drmTMB 0.7.1. Julia 1.10.0.
