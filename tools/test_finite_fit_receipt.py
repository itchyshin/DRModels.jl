#!/usr/bin/env python3
import copy,tomllib
from pathlib import Path
from check_finite_fit_receipt import check,ROOT
p=ROOT/'docs/dev-log/evidence/julia-r-parity/finite-state/finite-fit-002.toml'
b=tomllib.loads(p.read_text());check(b)
damages={
 'tolerance':lambda r:r.update(tolerance=1e-3),
 'source':lambda r:r['source_before'].update({'src/DRModels.jl':'bad'}),
 'runtime':lambda r:r['runtime'].update(julia_threads=2),
 'denominator':lambda r:r['cases'].pop('ordinal'),
 'false_pass':lambda r:r['cases']['categorical'].update(parity_pass=not r['cases']['categorical']['parity_pass']),
 'theta':lambda r:r['cases']['ordinal']['theta'].__setitem__(0,0.),
 'loglik':lambda r:r['cases']['ordinal'].update(loglik=0.),
 'prediction':lambda r:r['cases']['categorical']['prediction'].__setitem__(6,0.),
 'conditional':lambda r:r['cases']['ordinal']['imputation'].__setitem__(6,0.),
 'reported_error':lambda r:r['cases']['ordinal']['errors'].update(prediction=r['cases']['ordinal']['errors']['prediction']+1e-6),
 'status':lambda r:r['cases']['categorical']['imputation_status'].__setitem__(6,'ok'),
 'covariance':lambda r:r['cases']['ordinal']['covariance'][0].__setitem__(0,-1.),
 'gradient':lambda r:r['cases']['categorical'].update(gradient_max=1.),
  'covariance_scale':lambda r:r['cases']['ordinal'].update(covariance=[[1000.*(i==j) for i in range(8)] for j in range(8)]),
 'actual_sd':lambda r:r['cases']['ordinal']['imputation_sd'].__setitem__(6,0.),
 'availability':lambda r:r['cases']['ordinal']['imputation_sd_available'].__setitem__(6,False),
 'nonfinite':lambda r:r['cases']['ordinal']['prediction'].__setitem__(0,float('nan')),
}
for name,damage in damages.items():
 r=copy.deepcopy(b);damage(r)
 try:check(r)
 except (ValueError,TypeError,KeyError,IndexError):pass
 else:raise RuntimeError('accepted damaged '+name)

# Verdict-direction controls (PR #934 review B1). Once the honest verdict is PASS,
# flipping parity_pass only tests a false FAIL. These controls move the native
# anchor in a temporary copy, update the reported theta error to the honest value
# against that anchor, and keep the receipt's verdict at PASS. A validator that
# always says PASS, or that loosens 4e-6, accepts the forged receipt.
import json,tempfile
import check_finite_fit_receipt as m
def against_moved_anchor(move):
 anchor=json.loads(m.REF.read_text());r=copy.deepcopy(b)
 for kind,c in anchor['cases'].items():
  perm=m.permutation(c);theta=r['cases'][kind]['theta'];moved=c['theta'].copy()
  for i in range(len(theta)):moved[perm[i]]=move(c['theta'][perm[i]],theta[i])
  c['theta']=moved
  r['cases'][kind]['errors']['theta']=max(abs(theta[i]-c['theta'][perm[i]]) for i in range(len(theta)))
  if r['cases'][kind]['parity_pass'] is not True:raise RuntimeError('verdict controls need an honest PASS receipt')
 saved=m.REF,m.REFERENCE_SHA256
 with tempfile.TemporaryDirectory() as tmp:
  moved=Path(tmp)/m.REF.name;moved.write_text(json.dumps(anchor))
  (Path(tmp)/'finite-reference-003.toml').write_bytes(m.REF.with_name('finite-reference-003.toml').read_bytes())
  m.REF,m.REFERENCE_SHA256=moved,m.sha(moved)
  try:return m.check(r)
  finally:m.REF,m.REFERENCE_SHA256=saved
verdict_controls={
 # A genuine 1e-5 native break with the verdict forged to PASS.
 'forged_pass_1e-5':lambda a,t:a+1e-5,
 # Just above the bar: honest theta error 4.004e-6 > 4e-6, verdict forged to PASS.
 'threshold_above_4e-6':lambda a,t:t+4e-6*(1+1e-3),
}
for name,move in verdict_controls.items():
 try:against_moved_anchor(move)
 except ValueError as e:
  if str(e)!='parity verdict':raise RuntimeError(name+' rejected for the wrong reason: '+str(e))
 else:raise RuntimeError('accepted forged PASS '+name)
# Positive control: just below the bar (3.996e-6) the same forged receipt is honest
# and must be accepted, so the two controls above fail only on the verdict.
if against_moved_anchor(lambda a,t:t+4e-6*(1-1e-3))!={'ordinal':True,'categorical':True}:
 raise RuntimeError('below-threshold control not accepted as PASS')
print('FINITE_FIT_NEGATIVE_CONTROLS_PASS',len(damages)+len(verdict_controls)+1)
