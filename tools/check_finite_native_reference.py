#!/usr/bin/env python3
"""Independent finite-state mixture oracle and generated-fixture converter."""
import hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'docs/dev-log/evidence/julia-r-parity/finite-state/finite-native-003.json'
REFERENCE_SHA256="734560e416cf983d447aed529941bfba5b6395ab89a9a15689b27e43bf949c7b"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def vec(v,n,label):
    require(isinstance(v,list) and len(v)==n,label+' dimensions')
    require(all(type(x) in (float,int) and math.isfinite(x) for x in v),label+' finite')
def matrix(A,n,p,label):
    require(isinstance(A,list) and len(A)==n,label+' rows')
    for row in A:vec(row,p,label)
def near(a,b,tol,label):
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b),label+' finite')
    require(abs(a-b)<=tol,label+' mismatch')
def lse(v):
    a=max(v);return a+math.log(sum(math.exp(x-a) for x in v))
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def sigmoid(v):return 1/(1+math.exp(-v)) if v>=0 else math.exp(v)/(1+math.exp(v))
def dimensions(c):
    return len(c['mu_names']),len(c['sigma_names']) if isinstance(c['sigma_names'],list) else 1,len(c['X_predictor'][0])
def permutation(c):
    p,r,q=dimensions(c)
    return list(range(p+r))+list(range(p+r+c['K']-1,p+r+c['K']-1+q))+list(range(p+r,p+r+c['K']-1)) if c['kind']=='ordinal' else list(range(p+r+q*(c['K']-1)))
def row(c,theta,i):
    p,r,q=dimensions(c);K=c['K'];alpha=theta[p+r:]
    if c['kind']=='ordinal':
        eta=dot(c['X_predictor'][i],alpha[:q]);cuts=[alpha[q]]
        for t in alpha[q+1:]:cuts.append(cuts[-1]+math.exp(t))
        cumulative=[0]+[sigmoid(t-eta) for t in cuts]+[1]
        probs=[cumulative[j+1]-cumulative[j] for j in range(K)]
        require(all(v>0 for v in probs),'independent moderate-fixture ordinal probabilities')
        lp=[math.log(v) for v in probs]
    else:
        logits=[0]+[dot(c['X_predictor'][i],alpha[j*q:(j+1)*q]) for j in range(K-1)]
        den=lse(logits);lp=[x-den for x in logits]
    mu=[dot(c['state_design'][i*K+j],theta[:p]) for j in range(K)]
    logsigma=dot(c['X_sigma'][i],theta[p:p+r]);sigma=math.exp(logsigma)
    terms=[lp[j]-.5*math.log(2*math.pi)-logsigma-.5*((c['y'][i]-mu[j])/sigma)**2 if c['observed_y'][i] else lp[j] for j in range(K)]
    if c['observed_x'][i]:
        k=c['x'][i]-1;posterior=[float(j==k) for j in range(K)];ll=terms[k]
    else:
        den=lse(terms);posterior=[math.exp(v-den) for v in terms];ll=den if c['observed_y'][i] else 0.0
    expected_score=sum((j+1)*v for j,v in enumerate(posterior))
    estimate=expected_score if c['kind']=='ordinal' else posterior.index(max(posterior))+1
    variance=sum((j+1-expected_score)**2*v for j,v in enumerate(posterior))
    return ll,posterior,estimate,math.sqrt(variance),dot(posterior,mu)
def nll(c,t):return -sum(row(c,t,i)[0] for i in range(c['n']))
def check(r):
    require(r.get('schema')=='finite_joint_native_v1','schema')
    require(sha(REF)==REFERENCE_SHA256, 'immutable reference hash')
    anchor=json.loads(REF.read_text())
    for key in ('source_before','source_after','runner_sha256','R_version','TMB_version','loaded_native_DLL_sha256'):
        require(r.get(key)==anchor[key],'frozen provenance '+key)
    require(r.get('source_unchanged') is True and r['source_before']==r['source_after'],'source changed')
    require(set(r.get('cases',{}))=={'ordinal','categorical'},'case denominator');errors={}
    for kind,c in r['cases'].items():
        require(c.get('kind')==kind and c.get('n')==180 and c.get('K')==3,'case shape');n,K=c['n'],c['K']
        require(c.get('levels')==anchor['cases'][kind]['levels'],'level order')
        require(c.get('native_convergence')==0 and c.get('optimizer_control')==[],'native defaults/convergence')
        require(c.get('control_argument')=='omitted_defaults' and c.get('fit_control')==anchor['cases'][kind]['fit_control'], 'native complete controls')
        require(c.get('original_row')==list(range(1,n+1)),'original rows')
        for key in ('observed_x','observed_y'):require(isinstance(c.get(key),list) and len(c[key])==n and all(type(x) is bool for x in c[key]),'masks')
        require(len(set(zip(c['observed_x'],c['observed_y'])))==4,'mask denominator')
        for key in ('y','x'):vec(c.get(key),n,key)
        require(all((type(v) is int and 1<=v<=K) if o else v==0 for v,o in zip(c['x'],c['observed_x'])),'observed category codes')
        p,sd,q=dimensions(c);matrix(c['X_mu'],n,p,'Xmu');matrix(c['X_sigma'],n,sd,'Xsigma');matrix(c['X_predictor'],n,q,'Xp');matrix(c['state_design'],n*K,p,'state design')
        require(c.get('state_layout')=='row_then_state','state layout')
        names=['beta_mu']*p+['beta_sigma']*sd+(['theta_ord']*(K-1)+['beta_mi']*q if kind=='ordinal' else ['beta_mi']*(q*(K-1)))
        require(c.get('raw_names')==names,'raw parameter order')
        perm=permutation(c);vec(c.get('theta'),len(perm),'theta')
        t=[c['theta'][j] for j in perm];matrix(c.get('covariance'),len(t),len(t),'covariance')
        points=c.get('points');require(isinstance(points,list) and len(points)==3,'point denominator');mx=0;mg=0
        for point in points:
            vec(point.get('theta'),len(t),'point theta');tt=[point['theta'][j] for j in perm]
            value=nll(c,tt);near(value,point.get('nll'),1e-8,'native likelihood');mx=max(mx,abs(value-point['nll']))
            vec(point.get('gradient'),len(t),'gradient')
            for j in range(len(t)):
                h=1e-5*max(1,abs(tt[j]));hi=tt.copy();lo=tt.copy();hi[j]+=h;lo[j]-=h
                g=(nll(c,hi)-nll(c,lo))/(2*h);native=point['gradient'][perm[j]]
                near(g,native,1e-6,'native gradient');mg=max(mg,abs(g-native))
        near(c.get('loglik'),-nll(c,t),1e-8,'fit loglik')
        ids=[i+1 for i in range(n) if not c['observed_x'][i]];require(c.get('conditional_model_rows')==ids,'conditional rows')
        matrix(c.get('conditional_probabilities'),len(ids),K,'conditional probabilities')
        table=c.get('imputation');require(isinstance(table,list) and len(table)==n,'imputation denominator');vec(c.get('prediction'),n,'prediction')
        for i,item in enumerate(table):
            ll,prob,estimate,se,pred=row(c,t,i);observed=c['observed_x'][i]
            require(item.get('model_row')==i+1 and item.get('original_row')==i+1 and item.get('observed') is observed,'summary rows')
            near(item.get('estimate'),estimate,1e-8,'imputation estimate');near(c['prediction'][i],pred,1e-8,'state-weighted prediction')
            expected_source='observed' if observed else 'conditional_expected_score' if kind=='ordinal' else 'conditional_modal_category'
            require(item.get('source')==expected_source,'summary source')
            if observed or kind=='categorical':require(item.get('std_error')=='NA','unavailable SE')
            else:near(item.get('std_error'),se,1e-8,'ordinal conditional SD')
            require(item.get('uncertainty_status')==('route_conditional_se_unavailable' if kind=='categorical' and not observed else 'ok'),'uncertainty status')
            if not observed:
                for a,b in zip(c['conditional_probabilities'][ids.index(i+1)],prob):near(a,b,1e-8,'posterior probabilities')
        errors[kind]={'nll':mx,'gradient':mg}
    return errors
if __name__=='__main__':print('FINITE_NATIVE_ORACLE_PASS',check(json.loads(Path(sys.argv[1]).read_text())))
