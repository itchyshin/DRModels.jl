pkg <- "/home/snakagaw/hsq_work/drmtmb-biv-lognormal-main-b535bfbc-20261028/source"
test_dir <- file.path(pkg, "tests", "testthat")
testthat::test_dir(
  test_dir,
  filter = "julia-family-hurdle_nbinom2",
  reporter = "summary",
  load_package = "source",
  stop_on_failure = TRUE
)
