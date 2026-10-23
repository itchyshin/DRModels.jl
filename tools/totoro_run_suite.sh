#!/bin/bash
# DRModels.jl full suite on Totoro (D-205 checks; D-142 leave nothing behind; D-143 <=150 cores — this uses 1)
set -u
SHA="$1"; BR="${2:-feat/563-s7b-lss-sparse-multi}"
cd ~/s7b_work/DRModels.jl || exit 2
git fetch -q origin "$BR" && git checkout -q "$SHA" || { echo "CHECKOUT_FAILED"; exit 2; }
echo "HEAD=$(git rev-parse --short HEAD) START=$(date -u +%FT%TZ)" | tee ~/s7b_work/suite-$SHA.log
export OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
J=~/.juliaup/bin/julia
$J --project=. -e "using Pkg; Pkg.instantiate()" >/dev/null 2>&1
timeout 5400 $J --project=. -e "using Pkg; Pkg.test()" 2>&1 | tee -a ~/s7b_work/suite-$SHA.log | grep -E "Test Summary|Testing DRModels tests|Fail|Error During|Broken|Some tests did not pass" | tail -40
RC=${PIPESTATUS[0]}
echo "END=$(date -u +%FT%TZ) SUITE_EXIT=$RC" | tee -a ~/s7b_work/suite-$SHA.log
pkill -u "$USER" -f "julia.*s7b_work" 2>/dev/null; sleep 1
echo "julia_procs_left=$(pgrep -u "$USER" -f julia | wc -l)"
exit $RC
