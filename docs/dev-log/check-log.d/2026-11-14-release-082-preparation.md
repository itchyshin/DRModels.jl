# DRModels.jl 0.8.2 release preparation

The candidate starts from merged main `25b128cd21da201561daa59062da4eaf306d1e63`. The release metadata and notes now use version 0.8.2. The release assembly consumed 60 tracked news fragments. An independent check found 56 new entries and four entries already present in 0.8.1; the four remain in the historical section only. The 0.8.1 section remains byte-identical.

The news assembly and `git diff --check` passed. The local Julia release launcher was unavailable; Julia 1.6 ran the standalone news assembler only. Full release-source checks, broad claims and license review, platform evidence, and the drmTMB companion merge remain open. This candidate is not a release.

A follow-up release review identified two numerical claims whose underlying results were not retained in the repository. Those claims were removed from the assembled notes; the related implementation and test descriptions remain.
