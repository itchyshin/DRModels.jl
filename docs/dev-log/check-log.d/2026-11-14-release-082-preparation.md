# DRModels.jl 0.8.2 release preparation

The candidate starts from merged main `25b128cd21da201561daa59062da4eaf306d1e63`. The release metadata and notes now use version 0.8.2. The release assembly consumed 60 tracked news fragments. An independent check found 56 new entries and four entries already present in 0.8.1; the four remain in the historical section only. The 0.8.1 section remains byte-identical.

The news assembly and `git diff --check` passed. The local Julia release launcher was unavailable; Julia 1.6 ran the standalone news assembler only. Full release-source checks, broad claims and license review, platform evidence, and the drmTMB companion merge remain open. This candidate is not a release.

A follow-up release review identified two numerical claims whose underlying results were not retained in the repository. Those claims were removed from the assembled notes; the related implementation and test descriptions remain.

The Binomial restart timings are labeled as measurements made before the acceptance rule was tightened; no timing for the final rule is asserted.

Two duplicate spatial-kernel entries were consolidated into the more complete entry, which retains the LogNormal and explicit-range caveats.

The Gamma multi-start timing figures were removed because their raw timing and environment records were not located during review.

The Tweedie crossed standard-error timing was removed because its raw timing record was not located during review.

A Tweedie structured-mode timing and estimate comparison was removed because the cited note does not retain a source-pinned before-and-after timing receipt.

Bootstrap coverage and width figures, SkewNormal timing ratios, and a crossed-Tweedie speed ratio were omitted because supporting source-pinned runs were not located during review.

The SkewNormal guide and release note now state that only an accepted replacement must pass the restart checks. The original fit can remain unconverged or degenerate. The unpinned runtime ratios were removed from the guide.
