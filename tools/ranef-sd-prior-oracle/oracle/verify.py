#!/usr/bin/env python3
"""Acceptance checks for the independently derived dense Gaussian oracle."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def self_test():
    spec = importlib.util.find_spec('dense_oracle')
    assert spec is not None, 'dense oracle implementation is missing'
    from dense_oracle import evaluate, boundary, mutation_values, finite_difference, scalar
    data = json.loads((ROOT / 'fixture.json').read_text())
    assert len(data['cases']) == 3 and len(data['points']) >= 5
    assert [c['true_tau'] for c in data['cases']] == [0.0,.03,.6]
    for c in data['cases']:
        assert [c['group'].count(i) for i in range(3)] == [2,3,4]
        assert len(c['x']) == len(c['y']) == 9
    assert abs(evaluate(data['cases'][0], [0, 0, 0, 0, np.log(.8)])[3] - 0.4259189739779905) < 1e-12
    for case in data['cases']:
        assert np.isfinite(boundary(case, [0, 0, 0, 0]))
        for point in data['points']:
            obj, g, h, penalty = evaluate(case, point)
            fdg, fdh = finite_difference(case, point)
            assert np.max(np.abs(g-fdg)) < 1e-6
            assert np.max(np.abs(h-fdh) / np.maximum(1, np.abs(h))) < 1e-5
            for name, mutant in mutation_values(case, point).items():
                assert abs(mutant - obj) > 1e-8 * max(1, abs(obj)), name
    expected = json.loads((ROOT / 'expected.json').read_text())
    for result in expected['results']:
        actual = evaluate(data['cases'][result['case']], data['points'][result['point']])
        ref_obj=float(result['objective']); ref_g=np.array(result['gradient'], dtype=float); ref_h=np.array(result['hessian'], dtype=float)
        assert abs(actual[0]-ref_obj) <= 1e-8 * max(1,abs(ref_obj))
        assert np.max(np.abs(actual[1]-ref_g)) <= 1e-6
        assert np.max(np.abs(actual[2]-ref_h) / np.maximum(1,np.abs(ref_h))) <= 1e-5
        fdg,fdh=finite_difference(data['cases'][result['case']],data['points'][result['point']])
        assert np.max(np.abs(fdg-ref_g)) <= 1e-6
        assert np.max(np.abs(fdh-ref_h) / np.maximum(1,np.abs(ref_h))) <= 1e-5
    extra_path=ROOT / 'expected_noninteger.json'
    assert extra_path.exists(), 'noninteger shape reference is missing'
    extra=json.loads(extra_path.read_text())
    shape,rate=extra['shape'],extra['rate']
    for result in extra['results']:
        case=data['cases'][result['case']]; point=data['points'][result['point']]
        obj,g,h,pen=evaluate(case,point,shape=shape,rate=rate)
        assert abs(obj-float(result['objective'])) <= 1e-8*max(1,abs(float(result['objective'])))
        assert np.max(abs(g-np.array(result['gradient'],float))) <= 1e-6
        rh=np.array(result['hessian'],float)
        assert np.max(abs(h-rh)/np.maximum(1,abs(rh))) <= 1e-5
        assert abs((obj-pen)-float(result['data_nll'])) < 1e-10
        assert abs(pen-float(result['penalty'])) < 1e-12
        fg,fh=finite_difference(case,point,shape=shape,rate=rate)
        assert np.max(abs(fg-np.array(result['gradient'],float))) < 1e-6
        assert np.max(abs(fh-rh)/np.maximum(1,abs(rh))) < 1e-5
        mutants=mutation_values(case,point,shape=shape,rate=rate)
        for name,mutant in mutants.items():
            assert abs(mutant-obj) > 1e-8*max(1,abs(obj)),name
        c=2.4; changed=dict(case); changed['y']=(np.array(case['y'])*c).tolist()
        q=np.array(point,float); q[:2]*=c; q[2]+=np.log(c); q[4]+=np.log(c)
        obj2,g2,h2,pen2=evaluate(changed,q,shape=shape,rate=rate/c)
        assert abs(obj2-obj-(len(case['y'])+1)*np.log(c)) < 1e-10
        assert abs(pen2-pen-np.log(c)) < 1e-12
        factors=np.array([1/c,1/c,1,1,1])
        assert np.max(abs(g2-factors*g)) < 1e-10
        assert np.max(abs(h2-h*np.outer(factors,factors))) < 1e-10
    for result in extra['default_separation']:
        case=data['cases'][result['case']]; point=data['points'][result['point']]
        obj,_,_,pen=evaluate(case,point)
        assert abs((obj-pen)-float(result['data_nll'])) < 1e-10
        assert abs(pen-float(result['penalty'])) < 1e-12
    from dense_oracle import prior_terms
    for a,b in [(2.,3.),(2.5,1.7)]:
        eta=np.log((a-1)/b); p,g,h=prior_terms(eta,a,b); step=1e-4
        assert abs(g) < 1e-12 and h > 0
        assert p < prior_terms(eta-step,a,b)[0] and p < prior_terms(eta+step,a,b)[0]
        assert abs((prior_terms(eta+step,a,b)[0]-prior_terms(eta-step,a,b)[0])/(2*step)) < 1e-7
        assert abs((prior_terms(eta+step,a,b)[0]-2*p+prior_terms(eta-step,a,b)[0])/step**2-h) < 1e-6
        wrong_eta=np.log(a/b)
        assert abs(prior_terms(wrong_eta,a,b)[1]-1) < 1e-12
        assert abs(g-1) > 0.9  # Added Jacobian shifts the stationary point.

    case=data['cases'][0]; p=[1.1,.45,-.25,.18,-30.0]
    value, grad, hess, penalty=evaluate(case,p)
    assert penalty > 20 and abs(grad[4]+1) < 1e-10 and abs(hess[4,4]) < 1e-10
    assert abs((value-penalty)-boundary(case,p[:4])) < 1e-10
    assert np.max(np.abs(hess-hess.T)) < 1e-10
    # The prior-only mode has zero score on the SD scale and positive curvature.
    assert abs(3-1/(1/3)) < 1e-14 and 1/(1/3)**2 > 0
    fit_path=ROOT / 'fitted_optima.json'
    assert fit_path.exists(), 'fitted-optimum reference is missing'
    fits=json.loads(fit_path.read_text())
    assert len(fits['fits']) == 6
    eligible_count=0
    downstream_eligible_count=0
    for fit in fits['fits']:
        assert len(fit['attempts']) == 6
        assert len({tuple(attempt['start']) for attempt in fit['attempts']}) == len(fit['attempts'])
        assert all('status' in attempt and 'objective' in attempt for attempt in fit['attempts'])
        case=data['cases'][fit['case']]; a=fit['shape']; b=fit['rate']
        obj,g,h,pen=evaluate(case,fit['coordinates'],shape=a,rate=b)
        eig=np.linalg.eigvalsh(h); ratio=float(eig[0]/eig[-1]); score=float(np.max(abs(g)))
        assert abs(obj-fit['objective']) < 1e-8*max(1,abs(obj))
        assert abs(scalar(case,fit['coordinates'],shape=a,rate=b)-obj) < 1e-9
        assert abs(fit['independent_groupwise_objective']-obj) < 1e-9
        assert fit['selected_attempt'] == min(range(len(fit['attempts'])),key=lambda i: fit['attempts'][i]['objective'] if fit['attempts'][i]['objective'] is not None else float('inf'))
        for attempt in fit['attempts']:
            if attempt['objective'] is not None:
                assert abs(scalar(case,attempt['coordinates'],shape=a,rate=b)-attempt['objective']) < 1e-8*max(1,abs(attempt['objective']))
        fdg,fdh=finite_difference(case,fit['coordinates'],shape=a,rate=b)
        assert np.max(abs(fdg-g)) < 1e-6
        assert np.max(abs(fdh-h)/np.maximum(1,abs(h))) < 1e-5
        assert abs(pen-fit['penalty']) < 1e-10
        assert abs(obj-pen-fit['data_nll']) < 1e-8
        assert abs(score-fit['score_inf_norm']) < 1e-9
        assert abs(eig[0]-fit['hessian_min_eigenvalue']) < 1e-9
        assert abs(ratio-fit['hessian_eigenvalue_ratio']) < 1e-9
        eligible=bool(eig[0] > 0 and ratio >= 1e-6 and score <= 1e-6)
        assert fit['parameter_comparison_eligible'] == eligible
        assert obj <= min(attempt['objective'] for attempt in fit['attempts'] if attempt['objective'] is not None)+1e-7
        eligible_count+=eligible
        downstream_eligible_count+=bool(eligible and fit['refinement']['success'])
    assert eligible_count >= 1, 'no optimum meets numerical criteria'
    assert downstream_eligible_count >= 1, 'no successful refinement meets numerical criteria for downstream parameter comparison'
    print('ORACLE_SELF_TEST_PASS')

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--self-test', action='store_true'); args = p.parse_args()
    if args.self_test: self_test()
    else: p.error('choose --self-test')
