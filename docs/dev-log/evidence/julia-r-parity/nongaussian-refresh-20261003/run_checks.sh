#!/bin/bash
# usage: run_checks.sh TREE OUT JULIA_BIN_DIR   (DRModels.jl #606 fixture refresh)
set -u
TREE=$(realpath "$1"); OUT=$2; JBIN=$3
mkdir -p "$OUT"; OUT=$(realpath "$OUT")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 MKL_NUM_THREADS=1
export R_LIBS_USER=/nonexistent
J=$JBIN/julia; PY=/usr/bin/python3
RLIB=$HOME/claude-mr/1442/Rlib-main; RROOT=$HOME/claude-mr/1442/main
SHIM=$HOME/claude-606b/privlib_shim.R
REF=test/fixtures/joint_missing_predictor/native_reference.toml
UNC=test/fixtures/joint_missing_predictor/native_uncertainty.toml
ORACLE=docs/dev-log/evidence/julia-r-parity/missing-predictor-oracle/native-mi-oracle-003.json
FNAT=docs/dev-log/evidence/julia-r-parity/finite-state/finite-native-003.json
cd "$TREE"
gate() { local label=$1; shift; if [ -n "${ONLY:-}" ] && ! [[ $label =~ $ONLY ]]; then return; fi; ( eval "$*" ) > "$OUT/$label.log" 2>&1; local rc=$?
  echo "GATE $label exit=$rc :: $(grep -v '^\s*$' "$OUT/$label.log" | tail -n 2 | tr '\n' ' ' | cut -c1-400)" | tee -a "$OUT/summary.txt"; }
echo "ONLY=${ONLY:-all} TREE=$TREE OUT=$OUT JULIA=$($J --version) HEAD=$(git rev-parse --short HEAD) $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
# --- joint, prepared kernel at frozen native parameters (leaf-S9-joint-native)
gate jn_G1 "JULIA_NUM_THREADS=1 $J --project=. tools/check_joint_predictor_reference.jl $REF $OUT/joint-native.toml"
gate jn_G2 "$PY tools/check_joint_predictor_receipt.py $REF $OUT/joint-native.toml ."
gate jn_G3 "$PY tools/test_joint_predictor_receipt.py $REF $OUT/joint-native.toml ."
gate jn_G4 "$PY -O tools/test_joint_predictor_receipt.py $REF $OUT/joint-native.toml ."
# --- joint prepared default fits (leaf-S9-joint-fit, leaf-S9-joint-fit-parity)
gate jf_G1 "JULIA_NUM_THREADS=1 $J --project=. tools/check_joint_predictor_fit.jl $REF $OUT/joint-fit.toml"
gate jf_G2 "$PY tools/check_joint_predictor_fit_receipt.py $REF $OUT/joint-fit.toml ."
gate jf_G3 "$PY tools/test_joint_predictor_fit_receipt.py $REF $OUT/joint-fit.toml ."
gate jf_G4 "$PY -O tools/test_joint_predictor_fit_receipt.py $REF $OUT/joint-fit.toml ."
gate jfp_G1 "$PY tools/check_joint_predictor_fit_receipt.py $REF $OUT/joint-fit.toml . --native-parity"
# --- joint public formula fits (leaf-S9-joint-public-fit)
gate jpf_G1 "JULIA_NUM_THREADS=1 $J --project=. tools/check_joint_frontend_fit.jl $REF $UNC $OUT/joint-frontend-fit.toml"
gate jpf_G2 "$PY tools/check_joint_frontend_fit_receipt.py $REF $UNC $OUT/joint-frontend-fit.toml ."
gate jpf_G3 "$PY tools/test_joint_frontend_fit_receipt.py $REF $UNC $OUT/joint-frontend-fit.toml ."
gate jpf_G4 "$PY -O tools/test_joint_frontend_fit_receipt.py $REF $UNC $OUT/joint-frontend-fit.toml ."
gate jpf_G5 "$PY tools/check_joint_frontend_fit_receipt.py $REF $UNC $OUT/joint-frontend-fit.toml . --native-theta"
# --- native imputation uncertainty (leaf-S9-native-uncertainty)
gate nu_G1 "Rscript tools/joint_native_uncertainty_probe.R $ORACLE $RLIB $OUT/joint-native-uncertainty-preflight.json --preflight"
gate nu_G2 "Rscript tools/joint_native_uncertainty_probe.R $ORACLE $RLIB $OUT/joint-native-uncertainty.json"
# --- joint R public bridge (leaf-S9-r-joint-native; S10 matched-public)
gate jp_run "cd $RROOT && LD_PRELOAD=$JBIN/../lib/julia/libunwind.so.8 JULIA_HOME=$JBIN DRMTMB_PRIVLIB=$RLIB Rscript $SHIM tools/run-julia-joint-public.R $TREE $TREE/$ORACLE $OUT/joint-public"
gate jp_oracle "$PY tools/check_joint_bridge_public_receipt.py $OUT/joint-public.json $REF $OUT/joint-frontend-fit.toml $RROOT $TREE"
gate jp_neg "$PY tools/test_joint_bridge_public_receipt.py $OUT/joint-public.json $REF $OUT/joint-frontend-fit.toml $RROOT $TREE && $PY -O tools/test_joint_bridge_public_receipt.py $OUT/joint-public.json $REF $OUT/joint-frontend-fit.toml $RROOT $TREE"
gate rjn_G1 "$PY tools/check_joint_bridge_public_receipt.py $OUT/joint-public.json $REF $OUT/joint-frontend-fit.toml $RROOT $TREE --native"
# --- finite state (leaf-S9-finite-state-evidence, leaf-S9-finite-public)
gate fse_G1 "$PY tools/check_finite_native_reference.py $FNAT && $PY tools/test_finite_native_reference.py && $PY -O tools/test_finite_native_reference.py"
gate fk_run "JULIA_NUM_THREADS=1 $J --project=. tools/check_finite_joint_reference.jl $OUT/finite-julia.toml"
gate ff_run "JULIA_NUM_THREADS=1 $J --project=. tools/check_finite_joint_fit.jl $OUT/finite-fit.toml"
gate fse_G2 "$PY tools/check_finite_fit_receipt.py $OUT/finite-fit.toml"
gate fse_G4 "$PY tools/check_finite_fit_receipt.py $OUT/finite-fit.toml --require-parity"
gate fp_run "cd $RROOT && LD_PRELOAD=$JBIN/../lib/julia/libunwind.so.8 JULIA_HOME=$JBIN DRMTMB_PRIVLIB=$RLIB Rscript $SHIM tools/run-julia-joint-finite-public.R $TREE $TREE/$FNAT $OUT/finite-public"
gate fp_G2 "$PY tools/check_finite_public_receipt.py $OUT/finite-public.json $RROOT --damage && $PY -O tools/check_finite_public_receipt.py $OUT/finite-public.json $RROOT --damage"
gate fp_G3 "$PY -c 'import json; r=json.load(open(\"$OUT/finite-public.json\")); s={k:v[\"native_status\"] for k,v in r[\"cases\"].items()}; print(s); ok=set(s)=={\"ordinal\",\"categorical\"} and all(v==\"PASS\" for v in s.values()); print(\"FINITE_NATIVE_PARITY_PASS\" if ok else \"FINITE_NATIVE_PARITY_FAIL\"); raise SystemExit(0 if ok else 1)'"
gate fstop "$PY tools/check_finite_stopping_diagnostic.py --check docs/dev-log/evidence/julia-r-parity/finite-stopping/diagnostic-001.json --damage"
echo "DONE $(date -u +%FT%TZ)" | tee -a "$OUT/summary.txt"
