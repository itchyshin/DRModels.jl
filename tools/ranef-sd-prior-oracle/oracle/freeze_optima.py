"""One-time independent optimum search. Refuses to replace frozen artifacts."""
import datetime
import json
import math
import platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import minimize
from dense_oracle import evaluate, scalar

root=Path(__file__).resolve().parent
path=root/'fitted_optima.json'
if path.exists(): raise SystemExit('Refusing to overwrite frozen fitted_optima.json')
data=json.loads((root/'fixture.json').read_text())
starts=[[.8,.3,-.1,.12,-.35],[1.2,.55,-.3,.2,-1.2],[.5,.7,.1,-.15,-.7],
        [1.4,-.1,-.4,.3,.25],[1.,.4,-.2,.05,-2.],[-.4,1.2,.6,-.5,.7]]
fits=[]
for a,b in [(2.,3.),(2.5,1.7)]:
    for ci,case in enumerate(data['cases']):
        def objective(p):
            try:
                with np.errstate(all='raise'):
                    v=scalar(case,p,shape=a,rate=b)
                return v if math.isfinite(v) else 1e100
            except (OverflowError, FloatingPointError, ValueError, ZeroDivisionError):
                return 1e100
        attempts=[]
        for start in starts:
            result=minimize(objective,start,method='BFGS',options={'gtol':1e-8,'maxiter':1000})
            attempts.append(dict(start=start,status=int(result.status),success=bool(result.success),message=str(result.message),
                                 iterations=int(result.nit),evaluations=int(result.nfev),coordinates=result.x.tolist(),
                                 objective=float(result.fun) if math.isfinite(result.fun) else None))
        selected=min(range(len(attempts)),key=lambda i: attempts[i]['objective'] if attempts[i]['objective'] is not None else math.inf)
        initial=attempts[selected]['coordinates']
        # Refine groupwise objective with separately derived dense score/Hessian.
        refined=minimize(objective,initial,method='trust-exact',jac=lambda p:evaluate(case,p,a,b)[1],
                         hess=lambda p:evaluate(case,p,a,b)[2],options={'gtol':1e-10,'maxiter':500})
        p=refined.x; obj,g,h,pen=evaluate(case,p,a,b); eigen=np.linalg.eigvalsh(h)
        ratio=float(eigen[0]/eigen[-1]); score=float(np.max(abs(g)))
        fits.append(dict(case=ci,shape=a,rate=b,attempts=attempts,selected_attempt=selected,
                         refinement=dict(method='trust-exact, groupwise objective, dense analytical gradient and Hessian',
                                         status=int(refined.status),success=bool(refined.success),message=str(refined.message),
                                         iterations=int(refined.nit),objective=float(refined.fun)),
                         coordinates=p.tolist(),objective=obj,data_nll=obj-pen,penalty=pen,
                         independent_groupwise_objective=objective(p),score_inf_norm=score,
                         hessian_min_eigenvalue=float(eigen[0]),hessian_max_eigenvalue=float(eigen[-1]),
                         hessian_eigenvalue_ratio=ratio,
                         parameter_comparison_eligible=bool(eigen[0]>0 and ratio>=1e-6 and score<=1e-6)))
receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             objective='normalized ML plus Gamma density in SD measure; groupwise determinant-lemma scalar',
             coordinates=['beta0','beta1','alpha','gamma','eta=log(tau)'],
             initial_method='SciPy BFGS with objective numerical derivatives, all starts retained',
             eligibility='positive definite Hessian, minimum/maximum eigenvalue >=1e-6, full score infinity norm <=1e-6',
             versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),fits=fits)
path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
