#!/usr/bin/env bash
set -euo pipefail
base=/home/snakagaw/hsq_work/drm-parity-hurdle-nb2-20261101
jl=/home/snakagaw/.julia/juliaup/julia-1.10.12+0.x64.linux.gnu
export R_LIBS_USER=/home/snakagaw/hsq_work/drm-r461-ou-repro-20261028/library:/home/snakagaw/hsq_work/drmtmb-julia-bridge-meta-v-20261028-1632/rlib:/home/snakagaw/Rlibs/drmtmb-h2h-20260923:/home/snakagaw/Rlib
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 RCPP_PARALLEL_NUM_THREADS=1
export JULIA_DEPOT_PATH=/home/snakagaw/.julia JULIA_BINDIR="$jl/bin" JULIA_HOME="$jl/bin" DRM_JL_JULIA_HOME="$jl/bin"
export DRM_JL_PATH=/home/snakagaw/hsq_work/DRModels-ci-64ccb2d8 DRMODELS_JL_PATH=/home/snakagaw/hsq_work/DRModels-ci-64ccb2d8
export DRMTMB_JULIA_TESTS=true NOT_CRAN=true
export LD_LIBRARY_PATH="$jl/lib:$jl/lib/julia:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$jl/lib/julia/libunwind.so.8"
set +e
/usr/bin/time -p Rscript --vanilla "$base/paired-diagnostic.R" > "$base/paired-diagnostic.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$base/paired-diagnostic.exit"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$base/paired-diagnostic.done"
exit "$rc"
