#!/usr/bin/env bash
set -euo pipefail
base=/home/snakagaw/hsq_work/drm-parity-hurdle-nb2-20261101/native-controls
pkg=/home/snakagaw/hsq_work/drmtmb-biv-lognormal-main-b535bfbc-20261028/source
export R_LIBS_USER=/home/snakagaw/hsq_work/drm-r461-ou-repro-20261028/library:/home/snakagaw/hsq_work/drmtmb-julia-bridge-meta-v-20261028-1632/rlib:/home/snakagaw/Rlibs/drmtmb-h2h-20260923:/home/snakagaw/Rlib
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1 NOT_CRAN=true
set +e
/usr/bin/time -p Rscript --vanilla -e "testthat::test_dir(file.path('$pkg', 'tests', 'testthat'), filter = 'hurdle-nbinom2', reporter = 'summary', load_package = 'source')" > "$base/r-native.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$base/r-native.exit"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$base/r-native.done"
exit "$rc"
