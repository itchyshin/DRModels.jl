#!/bin/bash
# PR #934 review fixes: mutation before/after, Newton cross-check, final_checks on 1.10.12 and 1.13.1.
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=4 MKL_NUM_THREADS=1
C=~/claude-606c; T=$C/tree; D=docs/dev-log/evidence/julia-r-parity/nongaussian-refresh-20261003
{ echo "# mutation test BEFORE: PR head 89aa0fc57 bytes, run in ~/claude-606b/after (path-matching tree), finite-public-005.json $(date -u +%FT%TZ)"
  (cd ~/claude-606b/after && PUBN=005 python3 $C/mut.py); } > $C/mutation-before.txt 2>&1
{ echo "# mutation test AFTER: review-fix head $(git -C $T rev-parse --short HEAD), run in ~/claude-606c/tree, finite-public-007.json $(date -u +%FT%TZ)"
  (cd $T && PUBN=007 python3 $C/mut.py && echo "# finer tolerance mutants (4e-6 -> 5e-6), caught by the 4.004e-6 threshold control" && PUBN=007 python3 $C/mut5.py
   echo "# reviewer forged_pass.py (~/claude-rv934) on this tree"; python3 ~/claude-rv934/forged_pass.py); } > $C/mutation-after.txt 2>&1
(cd $T && python3 $D/newton_xcheck.py) > $C/newton-xcheck-out.txt 2>&1
for v in 1.10.12 1.13.1; do
  J=~/.julia/juliaup/julia-$v+0.x64.linux.gnu/bin/julia
  (cd $T && JULIA_NUM_THREADS=1 timeout 1800 $J --project=test -e "using Pkg; Pkg.develop(path=\".\"); Pkg.instantiate()") > $C/inst-$v.log 2>&1; echo "instantiate $v rc=$?" >> $C/progress.txt
  bash $T/$D/final_checks.sh $T $C/final-$v ~/.julia/juliaup/julia-$v+0.x64.linux.gnu/bin >> $C/progress.txt 2>&1
  git -C $T checkout -q -- test/Project.toml 2>/dev/null
done
echo ALLDONE >> $C/progress.txt
