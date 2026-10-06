#!/usr/bin/env python3
"""Independent covariance/density/row-restoration oracle for two frozen tree cases."""
import copy, hashlib, json, math, sys
from pathlib import Path

def require(ok, message):
    if not ok: raise ValueError(message)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def near(a,b,tol,label):
    require(isinstance(a,(int,float)) and math.isfinite(a) and math.isfinite(b) and abs(a-b)<=tol,label)
def vector(v,n,label):
    require(isinstance(v,list) and len(v)==n and all(isinstance(x,(int,float)) and math.isfinite(x) for x in v),label)
def matrix(v,n,label):
    require(isinstance(v,list) and len(v)==n,label)
    for row in v: vector(row,n,label)
def chol(v):
    n=len(v); L=[[0.]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            a=v[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j: require(a>0,'positive covariance');L[i][j]=math.sqrt(a)
            else: L[i][j]=a/L[j][j]
    return L

def solve(L,b):
    n=len(b);z=[0.]*n;x=[0.]*n
    for i in range(n): z[i]=(b[i]-sum(L[i][j]*z[j] for j in range(i)))/L[i][i]
    for i in reversed(range(n)): x[i]=(z[i]-sum(L[j][i]*x[j] for j in range(i+1,n)))/L[i][i]
    return x

def check(r,rroot,jroot):
    require(r.get('status')=='PASS' and r.get('source_unchanged') is True,'run status')
    paths=list(rroot.glob('R/*.R'))+[rroot/'NAMESPACE']+[p for p in (rroot/'src').rglob('*') if p.suffix in ('.cpp','.h','.hpp')]+list((jroot/'src').rglob('*.jl'))
    current={str(p.resolve()):sha(p) for p in paths}
    require(r.get('source_before')==current==r.get('source_after'),'current source manifest')
    require(r.get('runner_sha256')==sha(rroot/'tools/run-julia-polytomy-public.R'),'runner source')
    require(r.get('native_tolerance')==4e-6,'frozen tolerance')
    runtime=r.get('runtime',{})
    require(runtime.get('threads')==1 and runtime.get('blas')==1 and runtime.get('source')==str(jroot/'src/DRModels.jl'),'loaded runtime')
    require(isinstance(runtime.get('julia'),str) and isinstance(r.get('R_version'),str),'runtime versions')
    require(r.get('loaded_native_DLL_sha256')==sha(rroot/'src/drmTMB.so'),'loaded native DLL')
    require(set(r.get('cases',{}))=={'star','mixed'},'denominator')
    for kind,c in r['cases'].items():
        require(c.get('status')=='PASS' and c.get('native_status')=='PASS',kind+' verdict')
        labels=['s'+str(i) for i in range(1,13)]
        require(c.get('tip_labels')==labels,kind+' native labels')
        expected_tree = '('+','.join(s+':2' for s in labels)+');' if kind=='star' else '('+','.join('('+','.join(s+':1' for s in labels[g:g+3])+'):1' for g in range(0,12,3))+');'
        require(c.get('source_tree')==expected_tree,'frozen topology')
        require(c.get('serialized')=={'newick':expected_tree,'tip_order':labels},'serialized topology and order')
        C=[[2. if i==j else (1. if kind=='mixed' and i//3==j//3 else 0.) for j in range(12)] for i in range(12)]
        for field,scale in [('raw_covariance',1.),('correlation',.5)]:
            matrix(c[field],12,field)
            for i in range(12):
                for j in range(12):near(c[field][i][j],C[i][j]*scale,1e-12,field)
        d=c['new_data'];require(len(d)==60,'row denominator')
        require(all(sum(row['species']==s for row in d)==5 for s in labels),'repeated tip count')
        ids=[labels.index(row['species']) for row in d]
        for key in ['x','y']:vector([row[key] for row in d],60,key)
        direct=c['outputs']['direct'];order=direct['tip_order']
        require(sorted(order)==sorted(labels),'direct labels')
        near(direct['height'],2.,1e-12,'height')
        require(direct['n_total']==(13 if kind=='star' else 17),'actual nodes')
        matrix(direct['tree_covariance'],12,'direct covariance')
        cov_error=0.
        for i in range(12):
            for j in range(12):
                delta=abs(direct['tree_covariance'][i][j]-C[labels.index(order[i])][labels.index(order[j])]);cov_error=max(cov_error,delta)
        near(cov_error,0.,1e-12,'tip covariance identity');near(c['tree_covariance_error'],cov_error,1e-12,'covariance error record')
        require(set(c['outputs'])=={'native','bridge','direct'},'engine denominator')
        for name,v in c['outputs'].items():
            require(v.get('converged') is True,name+' converged')
            vector(v['mu'],2,name+' mu')
            near(v['sigma'],v['sigma'],0,name+' sigma');near(v['sd_corr'],v['sd_corr'],0,name+' sd')
            require(v['sd_corr']>0,'phylogenetic SD')
            S=[[v['sd_corr']**2*C[ids[i]][ids[j]]/2 for j in range(60)] for i in range(60)]
            V=[[S[i][j]+(math.exp(2*v['sigma']) if i==j else 0.) for j in range(60)] for i in range(60)]
            mu=[v['mu'][0]+v['mu'][1]*row['x'] for row in d]
            e=[row['y']-m for row,m in zip(d,mu)];L=chol(V);a=solve(L,e)
            ll=-.5*(60*math.log(2*math.pi)+2*sum(math.log(L[i][i]) for i in range(60))+sum(x*y for x,y in zip(e,a)))
            near(v['loglik'],ll,1e-6,name+' independently reconstructed likelihood')
            near(c['dense_loglik_error'][name],abs(v['loglik']-ll),1e-10,name+' likelihood error record')
            if name!='native':
                vector(v['fitted'],60,name+' fitted')
                expected=[mu[i]+sum(S[i][j]*a[j] for j in range(60)) for i in range(60)]
                require(max(abs(x-y) for x,y in zip(v['fitted'],expected))<4e-6,name+' independent conditional fitted and row order')
        for name,field in [('native','native_errors'),('bridge','bridge_errors')]:
            v=c['outputs'][name]
            errors={'mu':max(abs(x-y) for x,y in zip(v['mu'],direct['mu'])),'sigma':abs(v['sigma']-direct['sigma']),'sd':abs(v['sd_corr']-direct['sd_corr']),'loglik':abs(v['loglik']-direct['loglik'])}
            if name=='bridge':errors['fitted']=max(abs(x-y) for x,y in zip(v['fitted'],direct['fitted']))
            require(set(c[field])==set(errors),'error denominator')
            for key,value in errors.items():near(c[field][key],value,1e-12,field+key);require(value<=4e-6,field+' threshold')
    return True

def damages(receipt,rroot,jroot):
    changes=[lambda r:r.update(native_tolerance=1),lambda r:r['cases'].pop('mixed'),
        lambda r:r['cases']['star']['raw_covariance'][0].__setitem__(0,3),
        lambda r:r['cases']['mixed']['correlation'][0].__setitem__(1,0),
        lambda r:r['cases']['mixed']['outputs']['direct'].update(n_total=23),
        lambda r:r['cases']['star']['outputs']['direct'].update(sd_corr=1),
        lambda r:r['cases']['star']['outputs']['bridge']['fitted'].reverse(),
        lambda r:r['cases']['mixed']['outputs']['native'].update(loglik=0),
        lambda r:r['cases']['star']['new_data'][0].update(x=100),
        lambda r:r['cases']['mixed']['outputs']['direct']['tree_covariance'][0].__setitem__(1,0),
        lambda r:r['cases']['star']['outputs']['native'].update(converged=False),
        lambda r:r.update(source_after={}),lambda r:r['cases']['star']['serialized'].update(newick='wrong;'),lambda r:r['cases']['mixed'].update(source_tree='wrong;'),lambda r:r['runtime'].update(source='wrong'),lambda r:r.update(loaded_native_DLL_sha256='bad')]
    for change in changes:
        bad=copy.deepcopy(receipt);change(bad)
        try:check(bad,rroot,jroot)
        except (ValueError,KeyError,TypeError,IndexError):continue
        raise ValueError('damaged receipt accepted')
    return len(changes)

if __name__=='__main__':
    r=json.loads(Path(sys.argv[1]).read_text());rroot=Path(sys.argv[2]).resolve();jroot=Path(__file__).resolve().parents[1]
    check(r,rroot,jroot);print('POLYTOMY_PUBLIC_ORACLE_PASS cases=2 rows=120 native_and_direct_and_bridge=true')
    if '--damage' in sys.argv:print('POLYTOMY_PUBLIC_DAMAGES_REJECTED',damages(r,rroot,jroot))
