#!/usr/bin/env python3
"""Check the real JuliaCall result against dense math and immutable native data."""
import copy,hashlib,json,math,sys
from pathlib import Path
import check_two_gaussian_reference as oracle
from check_two_gaussian_fit import square,delta
require,vector=oracle.require,oracle.vector
ROOT=Path(__file__).resolve().parents[1]
RROOT=ROOT.parent/'drmTMB'
REF=ROOT/'docs/dev-log/evidence/julia-r-parity/two-gaussian/two-gaussian-native-001.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(r):
    require(r.get('fixture_sha256')==sha(REF),'native fixture hash')
    require(r.get('runner_sha256')==sha(RROOT/'tools/run-julia-joint-two-public.R'),'runner hash')
    paths=sorted((RROOT/'R').glob('*.R'))+[RROOT/'NAMESPACE']+sorted((ROOT/'src').rglob('*.jl'))
    sources={str(p):sha(p) for p in paths}
    require(r.get('source_unchanged') is True and r.get('source_before')==sources==r.get('source_after'),'source provenance')
    rt=r.get('runtime',{});require(rt.get('threads')==1 and rt.get('blas')==1 and rt.get('source')==str(ROOT/'src/DRModels.jl'),'runtime')
    require(r.get('status')=='PASS' and r.get('native_tolerance')==4e-6,'receipt status/tolerance')
    q=r.get('result',{});require(q.get('status')=='PASS' and q.get('native_status')=='PASS','result status')
    flag_names={'converged','nobs','rows1','rows2','masks1','masks2','residuals','summary','wald','no_se'}
    flags=q.get('flags');require(isinstance(flags,dict) and set(flags)==flag_names and all(v is True for v in flags.values()),'public operation flags')
    adapter=q.get('adapter_errors');require(isinstance(adapter,dict) and set(adapter)=={'public_theta','public_vcov','training','newdata'},'adapter error denominator')
    for value in adapter.values():require(type(value) in (float,int) and math.isfinite(value) and 0<=value<=1e-10,'adapter error finite tolerance')
    ref=json.loads(REF.read_text());f=ref['fixture']
    raw=q.get('raw_theta');vector(raw,11,'raw theta');order=[0,3,1,2,4,5,6,7,8,9,10];t=[raw[i] for i in order]
    V=q.get('raw_covariance');square(V,11,'raw covariance');oracle.cholesky(V)
    W=[[V[i][j] for j in order] for i in order]
    P=q.get('public_covariance');square(P,11,'public covariance');J=[1.0]*11
    for k in (7,10):J[k]=math.exp(raw[k])
    measured={'public_vcov':max(abs(P[i][j]-V[i][j]*J[i]*J[j]) for i in range(11) for j in range(11)),'public_theta':0.0}
    for i in range(11):
        for j in range(11):delta(P[i][j],V[i][j]*J[i]*J[j],1e-10,'public covariance transform')
    expected_blocks={'mu':raw[:4],'sigma':[raw[4]],'mi_x1':raw[5:7],'sigma_mi_x1':[J[7]],'mi_x2':raw[8:10],'sigma_mi_x2':[J[10]]}
    require(set(q.get('coef',{}))==set(expected_blocks),'coefficient blocks')
    for block,vals in expected_blocks.items():
        actual=q['coef'][block];actual=actual if isinstance(actual,list) else [actual]
        vector(actual,len(vals),'coefficient block')
        for a,b in zip(actual,vals):
            delta(a,b,1e-10,'coefficient scale');measured['public_theta']=max(measured['public_theta'],abs(a-b))
    delta(q.get('loglik'),-oracle.nll(t,f),1e-8,'dense likelihood')
    vector(q.get('prediction'),160,'training prediction')
    C=q.get('conditional_covariance');require(isinstance(C,list) and len(C)==160,'conditional covariance rows')
    errs={'theta':max(abs(a-b) for a,b in zip(t,ref['theta'])),'loglik':abs(q['loglik']-ref['loglik']),
          'training':max(abs(a-b) for a,b in zip(q['prediction'],ref['prediction']))}
    for j,key in enumerate(('x1','x2')):
        tab=q.get('imputation',{}).get(key);require(isinstance(tab,list) and len(tab)==160,'imputation denominator')
        errs['imputed'+str(j+1)]=0;errs['imputed'+str(j+1)+'_se']=0
        for i,row in enumerate(tab):
            require(row.get('variable')==key and row.get('original_row')==f['original_row'][i] and row.get('model_row')==i+1 and row.get('observed') is f[key+'_observed'][i],'imputation rows/masks')
            require(row.get('source')==('observed' if row['observed'] else 'conditional_mode') and row.get('uncertainty_status')=='ok','imputation status')
            _,m,c=oracle.row(t,f,i);delta(row.get('estimate'),m[j],1e-8,'conditional mean');square(C[i],2,'conditional covariance')
            for k in range(2):delta(C[i][j][k],c[j][k],1e-8,'conditional covariance')
            ekey='imputed'+str(j+1);native=ref[ekey][i]
            errs[ekey]=max(errs[ekey],abs(row['estimate']-native['estimate']))
            if row['observed']:require(row.get('std_error')=='NA','observed SE unavailable')
            else:
                jac=oracle.jacobian(t,f,i)[j];expected=math.sqrt(c[j][j]+sum(jac[a]*W[a][b]*jac[b] for a in range(11) for b in range(11)))
                delta(row.get('std_error'),expected,1e-7,'imputation uncertainty')
                errs[ekey+'_se']=max(errs[ekey+'_se'],abs(row['std_error']-native['std_error']))
            delta(q['prediction'][i],t[0]+t[1]*f['z'][i]+t[2]*q['imputation']['x1'][i]['estimate']+t[3]*q['imputation']['x2'][i]['estimate'],1e-10,'training prediction')
    nd=q.get('newdata');require(isinstance(nd,list) and len(nd)==6,'newdata denominator');vector(q.get('newdata_prediction'),6,'newdata prediction')
    for row,pred in zip(nd,q['newdata_prediction']):delta(pred,raw[0]+raw[1]*row['x1']+raw[2]*row['x2']+raw[3]*row['z'],1e-10,'newdata prediction')
    measured['training']=max(abs(q['prediction'][i]-(raw[0]+raw[1]*q['imputation']['x1'][i]['estimate']+raw[2]*q['imputation']['x2'][i]['estimate']+raw[3]*f['z'][i])) for i in range(160))
    measured['newdata']=max(abs(pred-(raw[0]+raw[1]*row['x1']+raw[2]*row['x2']+raw[3]*row['z'])) for row,pred in zip(nd,q['newdata_prediction']))
    for key,value in measured.items():delta(adapter[key],value,1e-12,'reported adapter error')
    require(set(q.get('native_errors',{}))==set(errs),'native error denominator')
    for k,v in errs.items():delta(q['native_errors'][k],v,1e-12,'native error');require(v<=4e-6,'native tolerance')
    return errs
if __name__=='__main__':
    print('TWO_PUBLIC_DENSE_ORACLE_PASS',check(json.loads(Path(sys.argv[1]).read_text())))
