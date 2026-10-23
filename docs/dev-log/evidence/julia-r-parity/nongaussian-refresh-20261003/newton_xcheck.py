# PR #934 review (m1) cross-check: pre-polish anchor + one independent Newton step vs the polished anchor.
# Run from the repository root on Totoro (python3, one BLAS thread): python3 docs/dev-log/evidence/julia-r-parity/nongaussian-refresh-20261003/newton_xcheck.py
import sys,json; sys.path.insert(0,"tools")
import check_finite_stopping_diagnostic as d
from check_finite_native_reference import ROOT,permutation
old=json.loads(d.REF.read_text())
new=json.loads((ROOT/"docs/dev-log/evidence/julia-r-parity/finite-state/finite-native-003.json").read_text())
for kind,c in old["cases"].items():
    p=permutation(c);t=[c["theta"][i] for i in p]
    step=d.positive_solve(d.hessian(c,t),[-x for x in d.gradient(c,t)])
    n=new["cases"][kind];target=[n["theta"][i] for i in permutation(n)]
    print(kind,"max|old+step-new|=%.3g"%max(abs(a+b-x) for a,b,x in zip(t,step,target)),"max|old-new|=%.3g"%max(abs(a-x) for a,x in zip(t,target)))
