#!/usr/bin/env python3
"""Independently replay finite-state bridge outputs; retain native parity losses."""
import copy,json,sys
from pathlib import Path
from check_finite_native_reference import ROOT,REF,REFERENCE_SHA256,sha,require,near,vec,matrix,row,nll,permutation
from check_finite_fit_receipt import inverse_hessian_check

def check(r,rroot):
    require(r.get('fixture_sha256')==REFERENCE_SHA256==sha(REF),'native fixture hash')
    require(r.get('runner_sha256')==sha(rroot/'tools/run-julia-joint-finite-public.R'),'runner hash')
    files=list((rroot/'R').glob('*.R'))+[rroot/'NAMESPACE']+list((ROOT/'src').rglob('*.jl'))
    source={str(p.resolve()):sha(p) for p in files}
    require(r.get('source_before')==source==r.get('source_after') and r.get('source_unchanged') is True,'current source')
    require(r.get('native_tolerance')==4e-6 and r.get('adapter_tolerance')==1e-10,'fixed tolerances')
    runtime=r.get('runtime',{})
    require(runtime.get('threads')==1 and runtime.get('blas')==1,'runtime threads')
    require(Path(runtime.get('source','')).resolve()==ROOT/'src/DRModels.jl','loaded source')
    require(r.get('status')=='PASS' and set(r.get('cases',{}))=={'ordinal','categorical'},'case denominator')
    reference=json.loads(REF.read_text());verdict={}
    for kind,c in reference['cases'].items():
        v=r['cases'][kind];raw=v.get('raw_theta');perm=permutation(c)
        vec(raw,len(perm),'theta');theta=[raw[j] for j in perm]
        V=v.get('raw_covariance');matrix(V,len(raw),len(raw),'raw covariance')
        inverse_hessian_check(c,theta,[[V[i][j] for j in perm] for i in perm])
        keep=[0,1,2,3,4,7] if kind=='ordinal' else list(range(len(raw)))
        public=v.get('public_covariance');matrix(public,len(keep),len(keep),'public covariance')
        for i,ri in enumerate(keep):
            for j,rj in enumerate(keep):near(public[i][j],V[ri][rj],1e-10,'public covariance axes')
        coef=v.get('coefficients',{});require(set(coef)=={'mu','sigma','mi_x'},'public coefficient blocks')
        flat=[];lengths={'mu':4,'sigma':1,'mi_x':1 if kind=='ordinal' else 4}
        for block in ('mu','sigma','mi_x'):
            x=coef[block];values=x if isinstance(x,list) else [x]
            vec(values,lengths[block],'coefficient block '+block);flat.extend(values)
        terms=v.get('coefficient_terms',{})
        expected_mi=['z'] if kind=='ordinal' else [level+':'+term for level in c['levels'][1:] for term in ['(Intercept)','z']]
        expected_terms={'mu':c['mu_names'],'sigma':['(Intercept)'],'mi_x':expected_mi}
        require(set(terms)==set(expected_terms),'coefficient term blocks')
        for b,expected in expected_terms.items():require((terms[b] if isinstance(terms[b],list) else [terms[b]])==expected,'coefficient terms')
        vec(flat,len(keep),'public coefficients')
        for a,b in zip(flat,[raw[i] for i in keep]):near(a,b,1e-10,'public coefficient values')
        near(v.get('loglik'),-nll(c,theta),1e-8,'independent likelihood')
        table=v.get('imputation');require(isinstance(table,list) and len(table)==c['n'],'imputation denominator')
        prediction=v.get('prediction');vec(prediction,c['n'],'prediction')
        info=v.get('predictor_info',{});ids=[i for i,o in enumerate(c['observed_x']) if not o]
        require(info.get('levels')==c['levels'],'levels')
        if kind=='ordinal':
            import math
            cuts=info.get('cutpoints');vec(cuts,2,'predictor cutpoints')
            near(cuts[0],raw[5],1e-10,'first predictor cut');near(cuts[1],raw[5]+math.exp(raw[6]),1e-10,'second predictor cut')
        probs=info.get('conditional_probabilities');matrix(probs,len(ids),c['K'],'posterior dimensions')
        errors={'theta':max(abs(a-b) for a,b in zip(raw,c['theta'])),
            'prediction':max(abs(a-b) for a,b in zip(prediction,c['prediction'])),
            'loglik':abs(v['loglik']-c['loglik']),
            'imputation':max(abs(a['estimate']-b['estimate']) for a,b in zip(table,c['imputation'])),
            'posterior':max(abs(a-b) for ra,rb in zip(probs,c['conditional_probabilities']) for a,b in zip(ra,rb))}
        if kind=='ordinal':errors['imputation_sd']=max(abs(table[i]['std_error']-c['imputation'][i]['std_error']) for i in ids)
        require(set(v.get('native_errors',{}))==set(errors),'error denominator')
        for key,value in errors.items():near(v['native_errors'][key],value,1e-12,'native error '+key)
        for i,item in enumerate(table):
            ll,prob,point,sd,pred=row(c,theta,i);observed=c['observed_x'][i]
            require(item.get('variable')=='x' and item.get('original_row')==i+1 and item.get('model_row')==i+1 and item.get('observed') is observed,'rows/masks')
            near(item.get('estimate'),point,1e-8,'actual imputation');near(prediction[i],pred,1e-8,'state-weighted prediction')
            if observed or kind=='categorical':require(item.get('std_error')=='NA','unavailable SD')
            else:near(item.get('std_error'),sd,1e-8,'actual conditional SD')
            expected='route_conditional_se_unavailable' if kind=='categorical' and not observed else 'ok'
            require(item.get('uncertainty_status')==expected,'uncertainty status')
            expected_source='observed' if observed else 'conditional_expected_score' if kind=='ordinal' else 'conditional_modal_category'
            require(item.get('source')==expected_source,'imputation source')
            if not observed:
                for a,b in zip(probs[ids.index(i)],prob):near(a,b,1e-8,'posterior')
        summary='conditional_expected_score' if kind=='ordinal' else 'conditional_modal_category'
        require(info.get('summary')==summary and info.get(summary)==[table[i]['estimate'] for i in ids],'conditional summary metadata')
        native='PASS' if all(e<=4e-6 for e in errors.values()) else 'FAIL'
        require(v.get('native_status')==native,'honest native verdict')
        fresh=[i for i in range(c['n']) if c['observed_x'][i] and c['observed_y'][i]][:6]
        require(v.get('newdata_rows')==[i+1 for i in fresh],'newdata row IDs')
        newdata=v.get('newdata');require(isinstance(newdata,list) and len(newdata)==6,'newdata retained')
        ndpred=v.get('newdata_prediction');vec(ndpred,6,'newdata predictions')
        ndexpected=[]
        for j,i in enumerate(fresh):
            require(newdata[j].get('x')==c['levels'][c['x'][i]-1],'newdata state label')
            near(newdata[j].get('z'),c['z'][i],1e-12,'newdata covariate')
            X=c['state_design'][i*c['K']+c['x'][i]-1]
            ndexpected.append(sum(a*b for a,b in zip(X,raw[:4])))
            near(ndpred[j],ndexpected[-1],1e-10,'actual newdata prediction')
        calculations=[row(c,theta,i) for i in range(c['n'])]
        adapter={'coef':max(abs(a-raw[i]) for a,i in zip(flat,keep)),
          'covariance':max(abs(public[i][j]-V[ri][rj]) for i,ri in enumerate(keep) for j,rj in enumerate(keep)),
          'training':max(abs(a-calc[4]) for a,calc in zip(prediction,calculations)),
          'newdata':max(abs(a-b) for a,b in zip(ndpred,ndexpected))}
        require(set(v.get('adapter_errors',{}))==set(adapter),'adapter error denominator')
        for key,value in adapter.items():
            reported=v['adapter_errors'][key]
            near(reported,value,1e-10,'adapter error '+key)
            require(0<=reported<=1e-10 and 0<=value<=1e-10,'adapter acceptance threshold '+key)
        flag_names={'converged','nobs','rows','masks','residuals','coefficient_blocks','no_ordinal_response','summary','wald','level_names','no_se'}
        flag_names.update({'cutpoints','expected_score'} if kind=='ordinal' else {'modal_category','unavailable'})
        require(v.get('status')=='PASS' and set(v.get('flags',{}))==flag_names and all(x is True for x in v['flags'].values()),'public operation flags')
        verdict[kind]=native
    return verdict

def damages(r,rroot):
    mutations=[lambda v:v.update(fixture_sha256='bad'),lambda v:v.update(native_tolerance=1),
      lambda v:v['cases'].pop('categorical'),lambda v:v['cases']['ordinal'].update(native_status={'PASS':'FAIL'}.get(v['cases']['ordinal']['native_status'],'PASS')),
      lambda v:v['cases']['ordinal']['raw_covariance'][0].__setitem__(0,1000),
      lambda v:v['cases']['ordinal']['public_covariance'][0].__setitem__(0,1000),
      lambda v:v['cases']['ordinal']['imputation'][6].__setitem__('std_error',99),
      lambda v:v['cases']['ordinal']['prediction'].__setitem__(0,99),
      lambda v:v['cases']['categorical']['predictor_info'].pop('conditional_modal_category'),
      lambda v:v['cases']['categorical']['imputation'][6].__setitem__('observed',True),
      lambda v:v['cases']['ordinal']['flags'].pop('wald'),
      lambda v:v['cases']['ordinal']['adapter_errors'].__setitem__('newdata',99),
      lambda v:v['cases']['ordinal']['predictor_info']['cutpoints'].__setitem__(0,99),
      lambda v:v['cases']['ordinal']['imputation'][6].__setitem__('source','observed'),
      lambda v:v['cases']['ordinal']['newdata_prediction'].__setitem__(0,99),
      lambda v:v['cases']['ordinal']['coefficients'].__setitem__('mu',v['cases']['ordinal']['coefficients']['mu'][:3]),
      lambda v:(v['cases']['ordinal']['prediction'].__setitem__(0,v['cases']['ordinal']['prediction'][0]+1e-9),
                v['cases']['ordinal']['adapter_errors'].__setitem__('training',1e-9))]
    for change in mutations:
        damaged=copy.deepcopy(r);change(damaged)
        try:check(damaged,rroot)
        except (ValueError,KeyError,TypeError,IndexError):continue
        raise ValueError('damaged receipt accepted')
    return len(mutations)

if __name__=='__main__':
    receipt=json.loads(Path(sys.argv[1]).read_text());rroot=Path(sys.argv[2]).resolve()
    print('FINITE_PUBLIC_ORACLE_PASS',check(receipt,rroot))
    if '--damage' in sys.argv:print('FINITE_PUBLIC_DAMAGES_REJECTED',damages(receipt,rroot))
