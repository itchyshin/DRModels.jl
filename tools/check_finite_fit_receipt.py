#!/usr/bin/env python3
"""Validate retained default-fit results independently; report strict parity losses."""
import json,math,sys,tomllib
from pathlib import Path
from check_finite_native_reference import ROOT,REF,REFERENCE_SHA256,sha,require,near,vec,matrix,row,nll,permutation

def inverse_hessian_check(c,theta,V):
    # Independent central-difference Hessian; absolute ||H V - I||_max <= 1e-4.
    # No Julia derivative/covariance routine is used by this oracle.
    d=len(theta);H=[[0.]*d for _ in theta];base=nll(c,theta)
    for i in range(d):
        h=1e-4*max(1,abs(theta[i]));plus=theta.copy();minus=theta.copy();plus[i]+=h;minus[i]-=h
        H[i][i]=(nll(c,plus)-2*base+nll(c,minus))/(h*h)
        for j in range(i):
            k=1e-4*max(1,abs(theta[j]));values=[]
            for si,sj in [(1,1),(1,-1),(-1,1),(-1,-1)]:
                t=theta.copy();t[i]+=si*h;t[j]+=sj*k;values.append(nll(c,t))
            H[i][j]=H[j][i]=(values[0]-values[1]-values[2]+values[3])/(4*h*k)
    # Independent Cholesky must exist, not merely positive diagonals.
    L=[[0.]*d for _ in theta]
    for i in range(d):
        for j in range(i+1):
            v=V[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j:
                require(v>0,'covariance positive definite');L[i][j]=math.sqrt(v)
            else:L[i][j]=v/L[j][j]
    error=max(abs(sum(H[i][k]*V[k][j] for k in range(d))-(1. if i==j else 0.)) for i in range(d) for j in range(d))
    require(error<=1e-4,'independent Hessian inverse')
    return error

def check(r):
    require(sha(REF)==REFERENCE_SHA256,'native reference hash')
    reference=json.loads(REF.read_text())
    require(r.get('tolerance')==4e-6,'unchanged tolerance')
    require(r.get('reference_sha256')==sha(REF.with_name('finite-reference-003.toml')),'transport hash')
    require(r.get('runner_sha256')==sha(ROOT/'tools/check_finite_joint_fit.jl'),'runner hash')
    source={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src').rglob('*') if p.is_file()}
    require(r.get('source_before')==source and r.get('source_after')==source and r.get('source_unchanged') is True,'current source')
    runtime=r.get('runtime',{})
    require(runtime.get('julia_threads')==1 and runtime.get('blas_threads')==1 and runtime.get('julia_version')=='1.10.12','runtime')
    require(Path(runtime.get('loaded_source','')).resolve()==(ROOT/'src/DRModels.jl').resolve(),'loaded source')
    require(type(r.get('seconds')) in (int,float) and math.isfinite(r['seconds']) and r['seconds']>0,'elapsed')
    require(set(r.get('cases',{}))=={'ordinal','categorical'},'case denominator')
    verdict={}
    for kind,c in reference['cases'].items():
        v=r['cases'][kind];perm=permutation(c);theta=v.get('theta');vec(theta,len(perm),'theta')
        require(v.get('optimizer_status')=='converged' and v.get('covariance_status')=='observed_information_inverse','fit statuses')
        matrix(v.get('covariance'),len(theta),len(theta),'raw covariance')
        for i in range(len(theta)):
            require(v['covariance'][i][i]>0,'covariance diagonal')
            for j in range(len(theta)):near(v['covariance'][i][j],v['covariance'][j][i],1e-10,'covariance symmetry')
        inverse_hessian_check(c,theta,v['covariance'])
        calculations=[row(c,theta,i) for i in range(c['n'])]
        vec(v.get('imputation_sd'),c['n'],'actual imputation SD')
        available=[kind=='ordinal' and not o for o in c['observed_x']]
        require(v.get('imputation_sd_available')==available,'imputation SD availability')
        for i,(calc,has) in enumerate(zip(calculations,available)):
            near(v['imputation_sd'][i],calc[3] if has else 0.,1e-8,'actual imputation SD')
        near(v.get('loglik'),-nll(c,theta),1e-8,'independent likelihood')
        for field,index in [('prediction',4),('imputation',2)]:
            vec(v.get(field),c['n'],field)
            for actual,expected in zip(v[field],calculations):near(actual,expected[index],1e-8,field)
        require(v.get('imputation_status')==['route_conditional_se_unavailable' if kind=='categorical' and not o else 'ok' for o in c['observed_x']],'imputation statuses')
        require(type(v.get('gradient_max')) in (int,float) and 0<=v['gradient_max']<=1e-7,'convergence gradient')
        for j,t in enumerate(theta):
            h=1e-5*max(1,abs(t));hi=theta.copy();lo=theta.copy();hi[j]+=h;lo[j]-=h
            require(abs((nll(c,hi)-nll(c,lo))/(2*h))<=1e-6,'independent near-stationarity')
        errors={'theta':max(abs(theta[i]-c['theta'][perm[i]]) for i in range(len(theta))),
            'loglik':abs(v['loglik']-c['loglik']),
            'prediction':max(abs(a-b) for a,b in zip(v['prediction'],c['prediction'])),
            'imputation':max(abs(a-b['estimate']) for a,b in zip(v['imputation'],c['imputation']))}
        if kind=='ordinal':errors['conditional_sd']=max(abs(a-b['std_error']) for a,b,o in zip(v['imputation_sd'],c['imputation'],c['observed_x']) if not o)
        require(set(v.get('errors',{}))==set(errors),'error fields')
        for key,error in errors.items():near(v['errors'][key],error,1e-10,'reported error '+key)
        passed=all(e<=4e-6 for e in errors.values())
        require(v.get('parity_pass') is passed,'parity verdict')
        verdict[kind]=passed
    return verdict

if __name__=='__main__':
    verdict=check(tomllib.loads(Path(sys.argv[1]).read_text()))
    print('FINITE_FIT_RECEIPT_VERIFIED',verdict)
    if '--require-parity' in sys.argv and not all(verdict.values()):raise SystemExit(1)
