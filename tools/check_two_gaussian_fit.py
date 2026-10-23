#!/usr/bin/env python3
"""Independent dense-oracle and source-bound validation of two-Gaussian kernel."""
import hashlib
import math
import sys
import tomllib
from pathlib import Path
import check_two_gaussian_reference as oracle
require,vector=oracle.require,oracle.vector
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'test/fixtures/joint_missing_predictor/two_gaussian_reference.toml'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def square(A,n,label):
    require(isinstance(A,list) and len(A)==n,label+': shape')
    for r in A: vector(r,n,label)
    require(max(abs(A[i][j]-A[j][i]) for i in range(n) for j in range(n))<1e-8,label+': symmetry')

def delta(a,b,tol,label):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),label+': nonfinite')
    require(abs(a-b)<=tol,label+': mismatch')

def point(p,f):
    t=p.get('theta');vector(t,11,'point theta')
    ll=p.get('row_loglik');vector(ll,160,'row likelihood')
    means=p.get('mean');C=p.get('conditional_covariance');status=p.get('status')
    require(isinstance(means,list) and len(means)==160 and isinstance(C,list) and len(C)==160,'moment dimensions')
    require(isinstance(status,list) and len(status)==160,'status denominator')
    for i in range(160):
        l,m,c=oracle.row(t,f,i);vector(means[i],2,'conditional mean');square(C[i],2,'conditional covariance')
        delta(ll[i],l,1e-8,'row likelihood')
        require(isinstance(status[i],list) and len(status[i])==2,'status shape')
        for j,key in enumerate(('x1','x2')):
            delta(means[i][j],m[j],1e-8,'conditional mean')
            for k in range(2):delta(C[i][j][k],c[j][k],1e-8,'conditional covariance')
            expected='observed' if f[key+'_observed'][i] else 'gaussian_posterior' if f['y_observed'][i] else 'predictor_only'
            require(status[i][j]==expected,'conditional status')
    delta(p.get('nll'),-sum(ll),1e-8,'total likelihood')
    vector(p.get('gradient'),11,'gradient')
    for k in range(11):
        hi,lo=list(t),list(t);h=1e-5*max(1,abs(t[k]));hi[k]+=h;lo[k]-=h
        d=(oracle.nll(hi,f)-oracle.nll(lo,f))/(2*h)
        delta(p['gradient'][k],d,1e-5,'independent gradient')

def table(q,t,V,f,j,public=True,se=True):
    key='x'+str(j+1);vector(q.get('estimate'),160,'imputation estimate');vector(q.get('std_error'),160,'imputation SE')
    require(isinstance(q.get('se_available'),list) and len(q['se_available'])==160 and all(type(v) is bool for v in q['se_available']),'SE availability')
    require(q.get('uncertainty_status')==['ok']*160,'uncertainty status')
    if public:
        require(q.get('variable')==[key]*160 and q.get('original_row')==f['original_row'] and q.get('model_row')==list(range(1,161)),'imputation identity')
        require(q.get('observed')==f[key+'_observed'],'imputation masks')
        require(q.get('source')==['observed' if v else 'conditional_mode' for v in f[key+'_observed']],'imputation source')
    for i in range(160):
        _,m,C=oracle.row(t,f,i);delta(q['estimate'][i],m[j],1e-8,'imputation mean')
        available=se and not f[key+'_observed'][i]
        require(q['se_available'][i] is available,'imputation SE mask')
        if available:
            J=oracle.jacobian(t,f,i)[j];parameter=sum(J[k]*V[k][l]*J[l] for k in range(11) for l in range(11))
            expected=math.sqrt(C[j][j]+parameter)
        else:expected=0.0
        delta(q['std_error'][i],expected,1e-7,'imputation SE')
        if not public:
            delta(q.get('conditional_variance',[None]*160)[i],C[j][j],1e-8,'conditional variance')
            delta(q.get('parameter_variance',[None]*160)[i],parameter if available else 0.0,1e-7,'parameter variance')

def check(r,native=False):
    ref=tomllib.loads(REF.read_text());f={k:ref[k] for k in ('y','x1','x2','z','y_observed','x1_observed','x2_observed','original_row')}
    require(r.get('schema')=='two_gaussian_fit_v1','schema')
    require(r.get('reference_sha256')==sha(REF),'reference hash')
    require(r.get('runner_sha256')==sha(ROOT/'tools/check_two_gaussian_fit.jl'),'runner hash')
    sources={str(p.relative_to(ROOT/'src')):sha(p) for p in (ROOT/'src').rglob('*') if p.is_file()}
    require(r.get('source_unchanged') is True and r.get('source_before')==sources==r.get('source_after'),'source provenance')
    runtime=r.get('runtime',{});require(runtime.get('threads')==1 and runtime.get('blas')==1 and runtime.get('source')==str(ROOT/'src/DRModels.jl'),'runtime')
    require(r.get('original_row')==f['original_row'] and r.get('observed_y')==f['y_observed'] and r.get('observed_x1')==f['x1_observed'] and r.get('observed_x2')==f['x2_observed'],'rows and masks')
    require(r.get('snapshot_isolated') is True,'snapshot isolation')
    vector(r.get('initial'),11,'initial parameters')
    points=r.get('points');require(isinstance(points,list) and len(points)==3,'point denominator')
    for actual,expected in zip(points,ref['points']):
        require(actual.get('theta')==expected['theta'],'reference coordinates');point(actual,f)
        delta(actual['nll'],expected['nll'],1e-7,'native likelihood')
        for a,b in zip(actual['gradient'],expected['gradient']):delta(a,b,1e-6,'native gradient')
    fitted=r.get('fitted',{});point(fitted,f);t=fitted['theta'];V=fitted.get('covariance');H=fitted.get('hessian')
    square(V,11,'covariance');square(H,11,'Hessian');oracle.cholesky(V);oracle.cholesky(H)
    require(max(abs(x) for x in fitted['gradient'])<=1e-6,'fit stationarity')
    require(fitted.get('converged') is True and fitted.get('optimizer_status')=='converged' and fitted.get('covariance_status')=='observed_information_inverse','fit statuses')
    require(fitted.get('nobs')==156,'fit nobs');delta(fitted.get('loglik'),-fitted['nll'],1e-9,'fit loglik')
    identity=max(abs(sum(H[i][k]*V[k][j] for k in range(11))-(1 if i==j else 0)) for i in range(11) for j in range(11))
    require(identity<=1e-6,'Hessian covariance identity')
    # Independent finite-difference Hessian. Scaled error criterion is fixed
    # before running: 2e-5; step2e-4 balances second-difference cancellation.
    steps=[2e-4*max(1,abs(x)) for x in t];base=oracle.nll(t,f)
    for i in range(11):
        for j in range(i+1):
            if i==j:
                hi,lo=list(t),list(t);hi[i]+=steps[i];lo[i]-=steps[i]
                expected=(oracle.nll(hi,f)-2*base+oracle.nll(lo,f))/steps[i]**2
            else:
                values=[]
                for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
                    v=list(t);v[i]+=a*steps[i];v[j]+=b*steps[j];values.append(oracle.nll(v,f))
                expected=(values[0]-values[1]-values[2]+values[3])/(4*steps[i]*steps[j])
            delta(H[i][j],expected,2e-5*(1+abs(expected)),'independent Hessian')
    fixed=r.get('fixed_native_accessor',{});require(fixed.get('scope')=='fixed-parameter uncertainty, not a fit','fixed accessor scope')
    for j in range(2):
        key='imputed'+str(j+1)
        table(fixed.get(key,{}),ref['theta'],ref['covariance'],f,j,public=False)
        table(fitted.get(key,{}),t,V,f,j)
        table(fitted.get(key+'_no_se',{}),t,V,f,j,se=False)
    expected={'theta':max(abs(a-b) for a,b in zip(t,ref['theta'])),'loglik':abs(fitted['nll']-ref['points'][0]['nll'])}
    for j in (1,2):
        key='imputed'+str(j)
        expected[key+'_mean']=max(abs(a-b) for a,b in zip(fitted[key]['estimate'],ref[key+'_mean']))
        expected[key+'_se']=max(abs(a-b) for a,b in zip(fitted[key]['std_error'],ref[key+'_se']))
    require(set(r.get('native_errors',{}))==set(expected),'native denominator')
    for k,v in expected.items():delta(r['native_errors'][k],v,1e-12,'native error record')
    status='PASS' if all(v<=4e-6 for v in expected.values()) else 'FAIL'
    require(r.get('native_status')==status,'native verdict')
    if native:require(status=='PASS','strict native parity')
    return expected

if __name__=='__main__':
    r=tomllib.loads(Path(sys.argv[1]).read_text());errors=check(r,native='--native' in sys.argv)
    print('TWO_GAUSSIAN_FIT_ORACLE_PASS native='+r['native_status']+' '+str(errors))
