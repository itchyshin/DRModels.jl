#!/bin/bash
# usage: final_checks.sh TREE OUT JULIA_BIN_DIR -- ledger CHECKs on the COMMITTED receipt/fixture paths + affected tests
set -u
TREE=$(realpath "$1"); OUT=$2; JBIN=$3; mkdir -p "$OUT"; OUT=$(realpath "$OUT")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 MKL_NUM_THREADS=1 R_LIBS_USER=/nonexistent
J=$JBIN/julia; PY=/usr/bin/python3; RROOT=$HOME/claude-mr/1442/main
E=docs/dev-log/evidence/julia-r-parity
REF=test/fixtures/joint_missing_predictor/native_reference.toml
UNC=test/fixtures/joint_missing_predictor/native_uncertainty.toml
cd "$TREE"
gate() { local label=$1; shift; if [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]]; then return; fi; ( eval "$*" ) > "$OUT/$label.log" 2>&1; local rc=$?
  echo "GATE $label exit=$rc :: $(grep -v '^\s*$' "$OUT/$label.log" | tail -n 2 | tr '\n' ' ' | cut -c1-400)" | tee -a "$OUT/summary.txt"; }
echo "ONLY=${ONLY:-all} TREE=$TREE JULIA=$($J --version) $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
if [ -z "${TESTS_ONLY:-}" ]; then
gate S9-finite-state-evidence_G1 "$PY tools/check_finite_native_reference.py $E/finite-state/finite-native-003.json && $PY tools/test_finite_native_reference.py && $PY -O tools/test_finite_native_reference.py"
gate S9-finite-state-evidence_G2 "$PY tools/check_finite_fit_receipt.py $E/finite-state/finite-fit-002.toml && $PY tools/test_finite_fit_receipt.py && $PY -O tools/test_finite_fit_receipt.py"
gate S9-finite-state-evidence_G4 "$PY tools/check_finite_fit_receipt.py $E/finite-state/finite-fit-002.toml --require-parity"
gate S9-finite-public_G2 "$PY tools/check_finite_public_receipt.py $E/finite-frontends/finite-public-007.json $RROOT --damage && $PY -O tools/check_finite_public_receipt.py $E/finite-frontends/finite-public-007.json $RROOT --damage"
gate S9-finite-public_G3 "$PY -c 'import json; r=json.load(open(\"$E/finite-frontends/finite-public-007.json\")); statuses={k:v[\"native_status\"] for k,v in r[\"cases\"].items()}; print(statuses); ok=set(statuses)=={\"ordinal\",\"categorical\"} and all(v==\"PASS\" for v in statuses.values()); print(\"FINITE_NATIVE_PARITY_PASS\" if ok else \"FINITE_NATIVE_PARITY_FAIL\"); raise SystemExit(0 if ok else 1)'"
gate S9-joint-fit-parity_G1 "$PY tools/check_joint_predictor_fit_receipt.py $REF $E/joint-prototype/joint-fit-003.toml . --native-parity"
gate S9-joint-public-fit_G5 "$PY tools/check_joint_frontend_fit_receipt.py $REF $UNC $E/joint-frontend/joint-frontend-fit-002.toml . --native-theta"
gate S9-r-joint-native_G1 "$PY tools/check_joint_bridge_public_receipt.py $E/joint-bridge/joint-public-003.json $REF $E/joint-bridge/joint-direct-bridge-002.toml $RROOT $TREE --native"
gate S9-r-joint-public-negatives "$PY tools/test_joint_bridge_public_receipt.py $E/joint-bridge/joint-public-003.json $REF $E/joint-bridge/joint-direct-bridge-002.toml $RROOT $TREE && $PY -O tools/test_joint_bridge_public_receipt.py $E/joint-bridge/joint-public-003.json $REF $E/joint-bridge/joint-direct-bridge-002.toml $RROOT $TREE"
gate S9-joint-native_G2 "$PY tools/check_joint_predictor_receipt.py $REF $E/joint-prototype/joint-native-003.toml . && $PY tools/test_joint_predictor_receipt.py $REF $E/joint-prototype/joint-native-003.toml . && $PY -O tools/test_joint_predictor_receipt.py $REF $E/joint-prototype/joint-native-003.toml ."
gate S10-finite-stopping-diagnostic "$PY tools/check_finite_stopping_diagnostic.py --check $E/finite-stopping/diagnostic-001.json --damage"
fi
for f in test/test_joint_missing_*.jl; do
  b=$(basename $f .jl)
  gate "test_${b}" "JULIA_NUM_THREADS=1 $J --project=test -e 'using LinearAlgebra; LinearAlgebra.BLAS.set_num_threads(1); using Test; @testset \"$b\" begin include(\"$f\") end'"
done
echo "DONE $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
