# Focused drmTMB current-main associate-pairs test

- Source: drmTMB main `129c09c8c8d56fa91bb09146f02dff24e65e6e7b`; live
  `origin/main` matched this SHA when archived.
- Archive: source subset containing `DESCRIPTION`, `NAMESPACE`, `R/`, `src/`,
  `inst/`, `tests/testthat.R`, and `tests/testthat/`; SHA-256
  `df361671a9b1dbbbe4cd60e142bd099900693990eab460b0ceb78c35e567441b`.
- Runtime: Totoro, Linux x86_64, R 4.6.1, one BLAS/OpenMP thread; initial
  load was about 289.
- Command: `devtools::test(<archive>/drmTMB,
  filter = "associate-pairs-bernoulli-bernoulli", reporter = "summary")`.
- Result: the focused file passed with no failures; 36 test expectations
  completed and the process exited zero. Its original raw output is
  `focused-test.log.gz`; the SHA-256 below is of the uncompressed bytes:
  `1798fbdfbc97dcdd5e55c6d4a09049dead66ca11ebfe2297853f0eb8637a6856`.
- Test file SHA-256 at this source pin:
  `f487548071de2808d3f546bbeb90a83419beb7e66d6ff1b4b86846c484ec5be4`.
- The log reports that no live Julia bridge tests ran. This was a focused test,
  not a complete R CMD check.

The raw log is gzip-compressed to preserve its original bytes while keeping
the repository whitespace check clean. Recover it with
`gzip -dc focused-test.log.gz`.

PR #1659 previously reported a failure of the reverse-margin `eta` assertion
on a macOS arm64 baseline at `b535bfbc`. The unchanged assertion passed on this
current-main Linux run. This does not invalidate the macOS result or establish
the source of the platform difference; it is evidence that the failure is not
reproduced on every platform. The engine and test file are unchanged from the
older baseline named in the PR report.
