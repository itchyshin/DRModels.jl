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

A separately authored reference builder was added at `test/fixtures/struct_sd_counts/ref_struct_sd_counts_independent.R`. On Totoro it reproduced all 99 rows of the stored 14-case reference byte-for-byte using an identified drmTMB 0.8.1 installation. The installed build's source commit was not recorded; the first run did not record execution-time hashes of all inputs. An independent review compared the replacement with the previously identified GPL runner and found no substantial consecutive match. The original copied generator remains present and is a release blocker; its lane overlap and historical references still need resolution. This reproduction does not establish independent numerical correctness.

A second 14-case run matched all 99 rows exactly and recorded the loaded package path, R and BLAS settings, and input and script hashes before and after execution. The installed drmTMB source commit remains unknown.

The package DESCRIPTION and compiled-library hashes were recorded in the environment receipt and matched at review; they were not hashed before and after execution.

On 2026-11-15, with maintainer authorization, the copied structured-SD reference runner was removed from the isolated release candidate, and the independently authored builder remains. The removed script remains in Git history. PR #1274 has not been changed and needs reconciliation. Final source-tree and release-archive license audits remain open.
