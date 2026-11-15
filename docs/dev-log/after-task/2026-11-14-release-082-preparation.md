# DRModels.jl 0.8.2 candidate preparation

The draft candidate updates version metadata and assembles the release notes from the merged main source. It preserves the four fragments already published in 0.8.1 only in their historical section and retains all 56 new fragments in 0.8.2. An independent reviewer verified that assembly.

The prior-table review is narrower than the full changes since 0.8.1. Full release-source checks, platform evidence, and a broader claims and license audit remain necessary before a tag or release. The corresponding drmTMB integration is still awaiting its new CI run.

A follow-up release review identified two numerical claims whose underlying results were not retained in the repository. Those claims were removed from the assembled notes; the related implementation and test descriptions remain.
