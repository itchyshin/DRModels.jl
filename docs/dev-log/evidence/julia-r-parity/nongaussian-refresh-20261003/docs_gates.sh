#!/bin/bash
set -u
cd ~/claude-606b/after
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1
J=~/.julia/juliaup/julia-1.10.12+0.x64.linux.gnu/bin/julia
O=~/claude-606b/docs-110; mkdir -p $O
timeout 1800 $J --project=docs -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()" > $O/instantiate.log 2>&1; echo "instantiate exit=$?" >> $O/summary.txt
mkdir -p docs/build
DRM_DOC_GATE_TMP=$(mktemp -d docs/build/finite-gate.XXXXXX) && out=$(timeout 1800 $J --project=docs tools/parity_docs_subset.jl --build-dir "$DRM_DOC_GATE_TMP/rendered" --page reference/engine-internals.md 2>&1) ; printf "%s\n" "$out" > $O/fse_G3.log; examples=$(printf "%s\n" "$out" | grep -oE "examples=[0-9]+" | tail -1 | cut -d= -f2); if [ -n "$examples" ] && [ "$examples" -ge 3 ]; then echo "GATE S9-finite-state-evidence_G3 DOCS_SUBSET_EXAMPLES_ATLEAST3_OK $(grep DOCS_SUBSET $O/fse_G3.log | tail -1)" >> $O/summary.txt; else echo "GATE S9-finite-state-evidence_G3 FAIL" >> $O/summary.txt; fi
build=$(mktemp -d docs/build/finite-doc-check.XXXXXX) && timeout 1800 $J --project=docs tools/parity_docs_subset.jl --build-dir "$build/output" --page reference/engine-internals.md --page r-julia-bridge.md > $O/fp_G4.log 2>&1; echo "GATE S9-finite-public_G4 exit=$? $(grep DOCS_SUBSET $O/fp_G4.log | tail -1)" >> $O/summary.txt
echo DONE >> $O/summary.txt
