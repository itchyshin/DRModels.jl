#!/usr/bin/env bash
set -euo pipefail
base=/home/snakagaw/hsq_work/drm-parity-hurdle-nb2-20261101/native-controls
project=/home/snakagaw/hsq_work/DRModels-ci-64ccb2d8
jl=/home/snakagaw/.julia/juliaup/julia-1.10.12+0.x64.linux.gnu
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_DEPOT_PATH=/home/snakagaw/.julia
set +e
/usr/bin/time -p "$jl/bin/julia" --project="$project" "$project/test/test_bridge_zi_marginal_mean.jl" > "$base/julia-bridge.log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$base/julia-bridge.exit"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$base/julia-bridge.done"
exit "$rc"
