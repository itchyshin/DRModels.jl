#!/usr/bin/env bash
set -u
prior_base=/home/snakagaw/hsq_work/drm-priors-20261105-01a1ad0b
prior_shard="$1"
cd "$prior_base/julia-validation8" || exit 99
prior_evidence="$prior_base/evidence/validation8"
date -u +%FT%TZ > "$prior_evidence/shard-$prior_shard.started"
printf '%s\n' "$$" > "$prior_evidence/shard-$prior_shard.pid"
env JULIA_DEPOT_PATH=/home/snakagaw/hsq_work/julia-depot:/home/snakagaw/.julia JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 DRM_TEST_SHARD="$prior_shard/8" DRM_PARITY_TESTS=0 timeout --signal=TERM --kill-after=30 10200 /home/snakagaw/hsq_work/julia-1.10.10/bin/julia --project=. -e 'using Pkg; Pkg.test(;allow_reresolve=false)' > "$prior_evidence/shard-$prior_shard.log" 2>&1
prior_code=$?
printf '%s\n' "$prior_code" > "$prior_evidence/shard-$prior_shard.exit"
date -u +%FT%TZ > "$prior_evidence/shard-$prior_shard.ended"
exit "$prior_code"
