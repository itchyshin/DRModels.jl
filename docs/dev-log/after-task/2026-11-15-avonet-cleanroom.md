# After task: independent AVONET preparation helper

A benchmark data-preparation block had a substantive text match with a drmTMB script and unresolved reuse provenance. With maintainer approval, the Julia benchmark now uses an independently authored preparation helper. The path discovery and benchmark fit workflow remain in place.

The helper was written from a behavior specification without consulting the excluded benchmark or drmTMB sources. A separate reviewer compared the candidate with the previous helper. The checks and their limits are recorded in `docs/dev-log/check-log.d/2026-11-15-avonet-cleanroom.md`. Other source matches and the release-wide GPL-to-MIT audit remain open.
