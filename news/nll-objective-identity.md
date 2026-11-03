- **`fit.nll` is now the objective the fit reports on four more routes.** A REML fit
  of the Gaussian location-scale-scale models (iid, structured, multi-component and
  sparse phylogenetic) and of the dense bivariate residual-correlation model stored
  the plain ML objective in `fit.nll`, so `gradient(fit)` (documented as the REML
  criterion's gradient) and `check_drm(fit)` read the O(1) ML score at the REML
  optimum and called a converged REML fit non-stationary; `confint(:profile)` also
  profiled the ML surface around the REML estimate. They now store the restricted
  objective: `loglik(fit) == -fit.nll(fit.theta)`. REML profile intervals on these
  routes therefore change (they are now REML profile intervals). Separately,
  `gradient()`, `hessian()`, `profile_curve()` and `parameter_surface()` on a coupled
  location-scale fit evaluated the engine-packed objective at `coef(fit)` (a
  different covariance ordering), returning a wrong value or the 1e18 barrier; they
  now map to the engine packing first. The bivariate q=2/q=4 structured REML routes
  still store the ML objective by documented policy and are unchanged.
