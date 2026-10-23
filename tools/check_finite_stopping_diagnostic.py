#!/usr/bin/env python3
"""Explain historical stopping differences, never replace the native comparator."""
import copy,json,math,sys
from pathlib import Path
from check_finite_native_reference import ROOT,REF,REFERENCE_SHA256,sha,require,near,nll,permutation,row

# This diagnostic explains the PRE-POLISH native stopping point. The live native
# anchor (check_finite_native_reference.REF) was regenerated from drmTMB 0eb0467851
# with Newton polish (#606), so pin the historical native fixture it explains.
REF=ROOT/"docs/dev-log/evidence/julia-r-parity/finite-stopping/finite-native-003-prepolish.json"
REFERENCE_SHA256="d8f75d1d4652d5580cee935b3eeb22d003b7ccb33d7b58f149cef190a712e7cb"

PUBLIC=ROOT/'docs/dev-log/evidence/julia-r-parity/finite-frontends/finite-public-003.json'
PUBLIC_SHA='0147b2657c81b223e5c4e5742e0d66b90ca570b576da5ce04887e7dcf3ef2ee2'

def gradient(c,t):
    result=[]
    for i,value in enumerate(t):
        h=1e-5*max(1,abs(value));a=t.copy();b=t.copy();a[i]+=h;b[i]-=h
        result.append((nll(c,a)-nll(c,b))/(2*h))
    return result

def hessian(c,t):
    d=len(t);H=[[0.]*d for _ in t];base=nll(c,t)
    for i,value in enumerate(t):
        h=1e-4*max(1,abs(value));a=t.copy();b=t.copy();a[i]+=h;b[i]-=h
        H[i][i]=(nll(c,a)-2*base+nll(c,b))/(h*h)
        for j in range(i):
            k=1e-4*max(1,abs(t[j]));values=[]
            for si,sj in ((1,1),(1,-1),(-1,1),(-1,-1)):
                z=t.copy();z[i]+=si*h;z[j]+=sj*k;values.append(nll(c,z))
            H[i][j]=H[j][i]=(values[0]-values[1]-values[2]+values[3])/(4*h*k)
    return H

def positive_solve(H,b):
    d=len(b);L=[[0.]*d for _ in b]
    for i in range(d):
        for j in range(i+1):
            s=H[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j:
                require(s>0,'independent local Hessian positive definite');L[i][j]=math.sqrt(s)
            else:L[i][j]=s/L[j][j]
    y=[0.]*d;x=[0.]*d
    for i in range(d):y[i]=(b[i]-sum(L[i][j]*y[j] for j in range(i)))/L[i][i]
    for i in reversed(range(d)):x[i]=(y[i]-sum(L[j][i]*x[j] for j in range(i+1,d)))/L[i][i]
    return x

def generate():
    require(sha(REF)==REFERENCE_SHA256 and sha(PUBLIC)==PUBLIC_SHA,'frozen inputs')
    native=json.loads(REF.read_text());public=json.loads(PUBLIC.read_text())
    out={'schema':'finite_stopping_explanation_v1','native_sha256':sha(REF),'public_sha256':sha(PUBLIC),
      'diagnostic_only':True,'native_replaced':False,'native_parity_tolerance':4e-6,
      'interpretation':'Local stopping-resolution evidence, not a global optimum certificate or a parity waiver.',
      'parameter_order':'prepared kernel; explicit native permutation','cases':{}}
    for kind,c in native['cases'].items():
        p=permutation(c);v=public['cases'][kind]
        t=[c['theta'][i] for i in p];target=[v['raw_theta'][i] for i in p]
        native_g=[c['points'][0]['gradient'][i] for i in p]
        gn=gradient(c,t);gj=gradient(c,target);H=hessian(c,t)
        # Use independently evaluated gradient AND curvature, not Julia V.
        step=positive_solve(H,[-x for x in gn]);delta=[b-a for a,b in zip(t,target)]
        corrected=[a+b for a,b in zip(t,step)]
        gradient_error=max(abs(a-b) for a,b in zip(gn,native_g))
        residual=max(abs(a-b) for a,b in zip(step,delta))
        predicted_gain=-.5*sum(a*b for a,b in zip(gn,step))
        actual_gain=v['loglik']-c['loglik']
        require(gradient_error<=2e-7,'independent/native gradient agreement')
        require(max(map(abs,gj))<=2e-7,'Julia independent stationary gradient')
        require(residual<=1e-8,'local Newton displacement explanation')
        require(actual_gain>0 and abs(predicted_gain-actual_gain)<=1e-11,'quadratic gain explanation')
        pred_error=max(abs(row(c,corrected,i)[4]-v['prediction'][i]) for i in range(c['n']))
        out['cases'][kind]={'native_max_gradient':max(map(abs,native_g)),
          'independent_native_max_gradient':max(map(abs,gn)),
          'julia_independent_max_gradient':max(map(abs,gj)),
          'independent_native_gradient_error':gradient_error,
          'independent_native_hessian':H,'diagnostic_step':step,'observed_displacement':delta,
          'displacement_residual_max':residual,'predicted_loglik_gain':predicted_gain,
          'observed_loglik_gain':actual_gain,'diagnostic_prediction_vs_julia_max':pred_error,
          'native_4e6_status':v['native_status'],'original_errors':v['native_errors']}
    return out

def check(r):
    expected=generate()
    # Deterministic arithmetic, same pinned generated inputs; exact JSON round-trip.
    require(r==expected,'diagnostic receipt changed or incomplete')
    require(all(v['native_4e6_status']=='FAIL' for v in r['cases'].values()),'original failures retained')
    return {k:{key:v[key] for key in ('native_max_gradient','julia_independent_max_gradient','displacement_residual_max')} for k,v in r['cases'].items()}

def damage(r):
    changes=[lambda x:x.update(native_replaced=True),lambda x:x.update(native_parity_tolerance=1),
      lambda x:x['cases'].pop('categorical'),lambda x:x['cases']['ordinal'].update(native_4e6_status='PASS'),
      lambda x:x['cases']['ordinal'].update(displacement_residual_max=0),
      lambda x:x['cases']['categorical']['independent_native_hessian'][0].__setitem__(0,1000)]
    for change in changes:
        bad=copy.deepcopy(r);change(bad)
        try:check(bad)
        except ValueError:continue
        raise ValueError('damaged stopping receipt accepted')
    return len(changes)

if __name__=='__main__':
    action,path=sys.argv[1:3];path=Path(path)
    if action=='--write':
        require(not path.exists(),'refusing stale output');path.parent.mkdir(parents=True,exist_ok=True)
        r=generate();path.write_text(json.dumps(r,indent=2)+'\n')
    elif action=='--check':r=json.loads(path.read_text())
    else:raise ValueError('use --write NEW_JSON or --check JSON [--damage]')
    print('FINITE_STOPPING_DIAGNOSTIC_PASS',check(r))
    if '--damage' in sys.argv:print('FINITE_STOPPING_DAMAGES_REJECTED',damage(r))
