#!/usr/bin/env python3
import copy,tomllib
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
print('FINITE_FIT_NEGATIVE_CONTROLS_PASS',len(damages))
