# Independent Gaussian random-intercept SD oracle

All code and data in this directory were created for this task without reading or calling either package. Original code is offered under the MIT license in LICENSE. NumPy is a separate dependency; no third-party source is vendored. Fixture generation was completed before any package result was examined; the fixed JSON files are inputs to comparisons, not generated at self-test time.

## Alignment and conventions

| Quantity | Generator and evaluator | Truth / extraction |
|---|---|---|
| mean beta0 + beta1 x | X beta | (1.1, 0.45); coordinates beta0,beta1 |
| log residual SD alpha + gamma x | exp(alpha + gamma x) times residual standard-normal draw | (-0.25, 0.18); coordinates alpha,gamma |
| group random intercept tau u | one standard-normal draw per group; sizes 2,3,4 | tau = 0, 0.03, 0.6; coordinate eta=log(tau) |
| Gamma SD prior | density with respect to d tau | shape 2, rate 3 by default, and shape 2.5, rate 1.7 in the additional frozen reference; no Jacobian |

The data record contains the literal x, standard-normal group and observation draws, and responses. The seed documents generation; the literal values are authoritative. The same draws are used across the three data cases. `points` are deliberately fixed interior evaluation points, not fitted estimates. No recovery claim is made.

For r=y-X beta, D=diag(exp(2(alpha+gamma*x))), V=D+exp(2 eta) ZZ', the ML negative log likelihood includes n log(2 pi)/2 and is `(log det(V)+r'V^-1 r+n log(2 pi))/2`. Add `P=3 exp(eta)-eta-2 log(3)+lgamma(2)` once, with no coordinate Jacobian. Prior SD mode is 1/3. Boundary `tau=0` evaluates the unpenalized likelihood with V=D; the prior diverges there for shape>1 and is not evaluated at eta=-infinity.

The analytical derivatives are obtained via V_i and V_ij. Set A=V^-1, u=Ar, B=A-uu'. Then `g_i=-m_i'u+tr(B V_i)/2 + P_i`, where m_i is the derivative of X beta. For coordinate j, `du_j=-A V_j u-A m_j` and `dB_j=-A V_j A-du_j u'-u du_j'`; the Hessian is `H_ij=-m_i' du_j + tr(dB_j V_i+B V_ij)/2+P_ij`. The only prior derivatives are P_eta=3 exp(eta)-1 and P_eta_eta=3 exp(eta). This includes all covariance/mean cross terms.

Independent numerical checks use a second scalar objective implemented by the groupwise determinant lemma and Sherman-Morrison formula, with central differences directly on the objective, including off-diagonal second differences. Tolerances: objective 1e-8 relative to max(1,abs(reference)); gradient absolute 1e-6; Hessian componentwise 1e-5 scaled by max(1,abs(reference)). The explicit central finite differences in the self-test use double precision; mpmath references use 60 decimal digits. Expected values are frozen derivatives computed at 60 decimal digits with mpmath and stored to 55 significant digits of the independently coded groupwise objective, not package outputs. The generator refuses to overwrite the frozen file; only the generator needs mpmath. The self-test requires only NumPy. `expected_noninteger.json` additionally freezes separate data negative log likelihood and prior penalty for both parameter sets, plus total objective and derivatives for the noninteger case; the original frozen data and reference files are unchanged. `--self-test` also tests wrong Jacobian, swapped shape/rate, missing normalizer, omitted/doubled/per-group prior, and wrong target. These mutants show discrimination at the specified points, not exhaustive correctness of either package.

Run from parent: `OPENBLAS_NUM_THREADS=1 python3 oracle/verify.py --self-test`. Require NumPy. Use `dense_oracle.evaluate(case, point)` for objective, gradient, Hessian, penalty, and `boundary(case, [beta0,beta1,alpha,gamma])` for the unpenalized boundary.

The noninteger case detects a mutant that omits only `lgamma(shape)`, which the original shape-2 case cannot distinguish because `lgamma(2)=0`. The self-test evaluates the prior and its derivatives at `eta=log((shape-1)/rate)` for both parameter sets, checking the SD-scale mode; the added-Jacobian mutant shifts its stationary point to `tau=shape/rate`.

For units rescaled by c=2.4, the test maps y and mean coefficients to c times their values, adds log(c) to alpha and eta, leaves gamma and shape fixed, and maps rate to rate/c. The normalized total negative log objective then increases by (n+1) log(c): n log(c) from the data density and log(c) from the prior density. It also checks the corresponding transformed gradient and Hessian. No fixed-effect prior is present.

## Frozen multi-start optima

`fitted_optima.json` records six dispersed numerical-derivative BFGS attempts for each combination of three cases and two priors. It retains each start, status, message, evaluation count, coordinates and objective. The lowest recorded objective was refined using the independent groupwise objective and dense analytic gradient and Hessian. The generator and receipt give versions, UTC timestamp, scores, Hessian eigenvalues, and the prespecified conditioning criterion. The self-test requires exactly six attempts in each of the six fits and at least one successful refinement meeting the numerical criteria. It re-evaluates every attempted endpoint, selects the recorded lowest attempt, checks each refined result with the separate groupwise and dense objectives, and checks analytic derivatives against finite differences.

All 36 numerical-derivative BFGS searches reported precision loss (status 2); three of six refinements also reported precision loss. The `parameter_comparison_eligible` field records numerical score and curvature eligibility only. Downstream parameter comparison additionally requires `refinement.success` to be true; a numerical eligibility flag is not a claim that the optimizer reported convergence or that the minimum is global. Three refinements reported success (Gamma(2,3) cases 0 and 1; Gamma(2.5,1.7) case 1), and all six refined scores satisfy the prespecified numerical threshold. There is no imposed SD floor. These are fixed-data comparator fits, not recovery evidence.
