# After task: data-only parity fixture replay

Four historical R fixture generators contained code described as copied or mirrored from drmTMB. With maintainer approval, they were removed from the release tree. Their committed numerical CSV data remain unchanged, and the old generators remain in Git history. The release tree no longer regenerates these data from their original seeds.

An independently authored runner, `tools/refit-parity-fixtures.R`, reads each retained CSV and refits its stated ordinal or Tweedie model. It records the installed package version, input and loaded-library hashes, diagnostics and fitted outputs in a new output directory. Its clean-room author did not inspect the excluded generators or drmTMB source. The accompanying check log records the bounded evidence and its limits.

For example, from the repository root with drmTMB and jsonlite installed:

```sh
Rscript tools/refit-parity-fixtures.R cumlogit-intercept test/parity/fixtures/cumlogit-mu-ranef/data.csv /tmp/new-parity-replay
```

The other case names are `cumlogit-slope-only`, `tweedie-intercept` and `tweedie-slope-only`; supply the corresponding CSV and a new output directory for each.

Other matched source files remain under provenance review. This replacement is not a complete GPL-to-MIT clearance or a claim that the historical data-generating process has been reproduced.
