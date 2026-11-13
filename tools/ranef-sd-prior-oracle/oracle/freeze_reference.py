"""One-time generator: groupwise scalar likelihood, mpmath numerical differentiation.
Never call in a package comparison; expected.json is the frozen reference.
"""
import json
from pathlib import Path
import mpmath as mp

root=Path(__file__).resolve().parent
mp.mp.dps=60
fixture=json.loads((root/'fixture.json').read_text(),parse_float=mp.mpf)

def f(case, p):
    b0,b1,alpha,gamma,eta=p
    x=case['x']; y=case['y']; groups=case['group']; t=mp.exp(2*eta)
    out=len(x)*mp.log(2*mp.pi)/2
    for k in sorted(set(groups)):
        ids=[i for i,g in enumerate(groups) if g==k]
        weights=[mp.exp(-2*(alpha+gamma*x[i])) for i in ids]
        res=[y[i]-b0-b1*x[i] for i in ids]
        denom=1+t*sum(weights)
        out+=(sum(2*(alpha+gamma*x[i]) for i in ids)+mp.log(denom)
              +sum(r*r*w for r,w in zip(res,weights))
              -t*sum(r*w for r,w in zip(res,weights))**2/denom)/2
    return out+3*mp.exp(eta)-eta-2*mp.log(3)+mp.loggamma(2)

results=[]
for c,case in enumerate(fixture['cases']):
    for q,p in enumerate(fixture['points']):
        fun=lambda *xs: f(case,xs)
        g=[mp.diff(fun,tuple(p),tuple(1 if k==i else 0 for k in range(5))) for i in range(5)]
        H=[[mp.diff(fun,tuple(p),tuple((k==i)+(k==j) for k in range(5))) for j in range(5)] for i in range(5)]
        results.append(dict(case=c,point=q,objective=mp.nstr(f(case,p),55),gradient=[mp.nstr(v,55) for v in g],hessian=[[mp.nstr(v,55) for v in row] for row in H]))
out=dict(method='mpmath 60 decimal digits, numerical differentiation of groupwise determinant-lemma scalar objective', coordinates=['beta0','beta1','alpha','gamma','eta'],results=results)
path=root/'expected.json'
if path.exists(): raise SystemExit('Refusing to overwrite frozen expected.json')
path.write_text(json.dumps(out,indent=2)+'\n')
