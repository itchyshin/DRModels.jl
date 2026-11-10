"""Freeze additional references only; refuses overwrite. No package dependencies."""
import json
from pathlib import Path
import mpmath as mp

mp.mp.dps=60
root=Path(__file__).resolve().parent
fixture=json.loads((root/'fixture.json').read_text(),parse_float=mp.mpf)

def pieces(case, p, a, b):
    b0,b1,alpha,gamma,eta=p
    x,y,groups=case['x'],case['y'],case['group']; t=mp.exp(2*eta)
    nll=len(x)*mp.log(2*mp.pi)/2
    for group in sorted(set(groups)):
        ids=[i for i,g in enumerate(groups) if g==group]
        w=[mp.exp(-2*(alpha+gamma*x[i])) for i in ids]
        r=[y[i]-b0-b1*x[i] for i in ids]
        nll+=(sum(2*(alpha+gamma*x[i]) for i in ids)+mp.log(1+t*sum(w))
              +sum(ri*ri*wi for ri,wi in zip(r,w))
              -t*sum(ri*wi for ri,wi in zip(r,w))**2/(1+t*sum(w)))/2
    penalty=b*mp.exp(eta)-(a-1)*eta-a*mp.log(b)+mp.loggamma(a)
    return nll, penalty

out=dict(method='60 decimal digit mpmath computation; 55 significant digits stored; numerical derivatives of groupwise objective',shape=2.5,rate=1.7,results=[],default_separation=[])
for c,case in enumerate(fixture['cases']):
    for q,p in enumerate(fixture['points']):
        a,b=mp.mpf('2.5'),mp.mpf('1.7')
        fun=lambda *xs: sum(pieces(case,xs,a,b))
        nll,penalty=pieces(case,p,a,b)
        gradient=[mp.diff(fun,tuple(p),tuple(int(k==i) for k in range(5))) for i in range(5)]
        hessian=[[mp.diff(fun,tuple(p),tuple(int(k==i)+int(k==j) for k in range(5))) for j in range(5)] for i in range(5)]
        fmt=lambda v: mp.nstr(v,55)
        out['results'].append(dict(case=c,point=q,data_nll=fmt(nll),penalty=fmt(penalty),objective=fmt(nll+penalty),gradient=list(map(fmt,gradient)),hessian=[list(map(fmt,row)) for row in hessian]))
        nll,penalty=pieces(case,p,mp.mpf(2),mp.mpf(3))
        out['default_separation'].append(dict(case=c,point=q,data_nll=fmt(nll),penalty=fmt(penalty)))
path=root/'expected_noninteger.json'
if path.exists(): raise SystemExit('Refusing to overwrite frozen expected_noninteger.json')
path.write_text(json.dumps(out,indent=2)+'\n')
