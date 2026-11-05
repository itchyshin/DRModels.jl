#!/usr/bin/env bash
set -u
prior_base=/home/snakagaw/hsq_work/drm-priors-20261105-01a1ad0b
for prior_shard in 1 2 3 4 5 6 7 8; do
  setsid bash "$prior_base/run-prior-validation-one.sh" "$prior_shard" &
  printf '%s\t%s\n' "$prior_shard" "$!" >> "$prior_base/evidence/validation8/process-groups.tsv"
done
wait
