# DRModels.jl 0.8.2 candidate preparation

The draft candidate updates version metadata and assembles the release notes from the merged main source. It preserves the four fragments already published in 0.8.1 only in their historical section and represents all 56 new fragments in 0.8.2. An independent reviewer verified that assembly.

The prior-table review is narrower than the full changes since 0.8.1. Full release-source checks, platform evidence, and a broader claims and license audit remain necessary before a tag or release. The corresponding drmTMB integration is still awaiting its new CI run.

A follow-up release review identified two numerical claims whose underlying results were not retained in the repository. Those claims were removed from the assembled notes; the related implementation and test descriptions remain.

The Binomial restart timings are labeled as measurements made before the acceptance rule was tightened; no timing for the final rule is asserted.

Two duplicate spatial-kernel entries were consolidated into the more complete entry, which retains the LogNormal and explicit-range caveats.

The Gamma multi-start timing figures were removed because their raw timing and environment records were not located during review.

The Tweedie crossed standard-error timing was removed because its raw timing record was not located during review.

A Tweedie structured-mode timing and estimate comparison was removed because the cited note does not retain a source-pinned before-and-after timing receipt.

Bootstrap coverage and width figures, SkewNormal timing ratios, and a crossed-Tweedie speed ratio were omitted because supporting source-pinned runs were not located during review.

The SkewNormal guide and release note now state that only an accepted replacement must pass the restart checks. The original fit can remain unconverged or degenerate. The unpinned runtime ratios were removed from the guide.

A separately authored builder reproduced the stored 14-case, 99-row reference byte-for-byte on an identified drmTMB 0.8.1 installation. An independent reviewer found no substantial consecutive match with the GPL runner. The original copied generator remains in the tree and blocks release; its overlapping lane and historical references need resolution. The installed build's source commit remains unknown; the first run did not record execution-time hashes of all inputs.

A second 14-case run matched all 99 rows exactly and recorded the loaded package path, R and BLAS settings, and input and script hashes before and after execution. The installed drmTMB source commit remains unknown.

The package DESCRIPTION and compiled-library hashes were recorded in the environment receipt and matched at review; they were not hashed before and after execution.
