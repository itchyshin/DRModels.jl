# After-task: sparse Gaussian crossed `(1 | g) + (1 | h)` (#977, lane N5)

## 1. Goal

Owner decision 27(a), 11 Oct 2026: stop the Gaussian multi-component route
(`_fit_multi_ranef_gaussian`) from scaling about as n² (n = 2000 took 331 s against
drmTMB's 0.5 s) by replacing the dense QR in `_multi_re_qr` with a sparse
factorisation of the q × q Woodbury capacitance. Estimates must not change.

## 2. Implemented

New file `src/gaussian_multi_re_sparse.jl` (included before `gaussian_ranef.jl`):

- **Value.** `_multi_re_factor` forms M = I + Z̃ᵀD⁻¹Z̃ sparsely and takes its sparse
  Cholesky (CHOLMOD, fill-reducing ordering). Gate: `max diag(M) <= 1e4`
  (`_MULTI_RE_CHOL_MAXDIAG`), otherwise — and if the factorisation fails — the sparse
  QR (SPQR, `tol = 0`) of `[D^{-1/2}Z̃; I]`, which never forms M. The gate keeps the
  #835/#837 cancellation guard: `test_cancellation_sweep.jl` (σ_e down to e⁻²⁰)
  passes unchanged.
- **Gradient.** `_multi_re_selinv` computes M⁻¹ on the factor pattern (Takahashi
  recursion on the leading columns, a dense LAPACK `potri` on the trailing block that
  is ≥ 90 % structurally full), then diag(M⁻¹) and the per-row leverage. Pattern is
  seeded with the pattern of M so every entry the leverage needs is stored.
- **Cache.** `nll(θ)` and `grad!(θ)` share one factorisation at the same θ.
- **vcov.** For q ≤ 200 the Hessian is still ForwardDiff through the dense objective
  (unchanged SEs on small fits). Above that it is a central difference of the exact
  analytic gradient (`_multi_re_fd_hessian`), under the same PD / finiteness guards.
- ForwardDiff (Dual) calls still use the dense `_multi_re_qr`.

## 3a. Decisions and rejected alternatives

- Sparse QR first (identity with the dense route by construction), then Cholesky
  behind a gate. Cholesky alone was rejected: it loses the identity to rounding as
  σ_e → 0 with one record per level (the #835/#837 class).
- The optimiser, start values and tolerances were NOT touched. At n = 20000 the fit
  needs on the order of 80 objective+gradient evaluations (44.9 s ÷ 0.57 s per
  evaluation; an estimate, not instrumented), which is data/start dependent. Changing
  that would risk the "estimates must not change" rule, so it is left as follow-up.
- SPQR column orderings were measured (DEFAULT/COLAMD/AMD/METIS/…, 0.8–2.2 s at
  n = 20000); ordering is not the lever.
- A name collision (`_fd_hessian_from_grad` already lives in `vcov_guard.jl`) caught
  by the suite in an intermediate commit; the helper was renamed `_multi_re_fd_hessian`.

## 3b. Evidence

Identity with the pre-#977 route (verbatim dense objective + gradient copied from
origin/main 740e735, same optimiser and start) in
`test/test_multi_re_sparse_kernel.jl`: objective rtol 1e-11, gradient rtol 1e-8, logLik
and every coefficient 1e-8, on crossed, three-component + varying-σ, weighted
`(0 + x | h)`, and q = 280 (> the dense-Hessian threshold) fixtures; selected inverse
and leverage vs dense inverse to 1e-12 on both routes; both routes give the same fit.

Real data (the issue's 2000-row `lme4::InstEval` extract, run on pristine main and on
this branch): logLik −843.2600014035 / −1688.1891529411 / −3374.1968374752 at
n = 500 / 1000 / 2000 on both, θ̂ equal to ~1e-12.

Timings (`bench/bench_crossed_sparse_977.jl`, InstEval-shaped synthetic design:
2972 × 1128 levels, 1 thread, Julia 1.10.4):

| n | q | before | QR-only build | final (Cholesky + cache) |
|---|---|---|---|---|
| 2 000 | 2 386 (real data: 2 163) | 331 s (issue) / 1318 s here, contended | 0.47 s | 0.26 s |
| 20 000 | 4 097 | not finished in 36 min (issue) | 195 s | 44.9 s |
| 100 000 | 4 100 | — | — | 28.4 s |

drmTMB's own times (0.5 s / 4.9 s / 14.8 s at n = 73 421) are the issue reporter's,
not re-run here. DRModels is still slower than drmTMB at n = 20 000; the gap is the
number of optimiser evaluations (above), not the per-evaluation linear algebra.
The 1318 s "before" figure ran while a test suite shared the CPUs, so it is an upper
bound; the 331 s figure is the reporter's.

q = 4 baseline (route untouched): `bench/run_sparse_tmb_nd.jl` logLik −256.5118
(drmTMB −256.52), converged. Wall time on this container (1.8 s) is not comparable to
the handover's 2.18× machine and is not claimed.

## 4. Pre-existing failure, not caused by this change

`test/test_locscale_profile_status.jl` lines 284–288, 336 fail when run inside the
sharded suite (shard 1/4 of 4) in this container, identically on pristine main
(740e735); it passes 111/111 when run alone on both. Location-scale profile route;
no overlap with this diff.

## 5. Open / follow-ups

- Fewer optimiser evaluations on weakly identified crossed designs (start values /
  warm start) — separate issue, estimates-affecting.
- The `(1 | g) + (1 | h)` non-Gaussian crossed Laplace routes are untouched.

## 6. Rose audit (claim vs evidence, scope, drift, license)

Verdict: **clean after three fixes made during the audit.**

- Claim: "estimates, logLik and BLUPs unchanged". Evidence existed for logLik,
  coefficients, objective and gradient; BLUPs were only length-checked. **Fixed:** the
  kernel test now compares `ranef` with the dense GLS conditional means to 1e-8
  (test file is 51/51, not 50/50).
- Claim: speed. Evidence is the bench rows (`bench/out/crossed_sparse_977.csv`, matches
  the PR table) on one machine, one synthetic InstEval-shaped design, 1 thread; the real
  InstEval check is identity only, on the issue's 2000 rows. The 331 s and drmTMB times are
  the reporter's, not re-run; 1318 s here was measured under CPU contention and is labelled
  an upper bound. The n = 20000 gap to drmTMB is stated, not hidden.
- Drift in code comments: the `_multi_re_qr` header said M is never formed and the vcov
  comment named `cholesky(M + I)`; both now describe the Cholesky/QR split. Comment-only.
- Drift in docs on main: none found (no page states the crossed size limit on main). The open
  docs PR #1056 (lane J5, #988) adds "crossed Gaussian size limit" notes that cite #977; they
  become stale once this merges and need updating by that lane.
- Scope: only `_fit_multi_ranef_gaussian` changed; optimiser, starts and tolerances are
  untouched; q = 4 route untouched (baseline logLik -256.5118).
- License: no drmTMB GPL source vendored; no InstEval data committed (bench design is
  synthetic); the only repository data file added is a bench timing CSV.
- Residual risk, stated not fixed: the Cholesky gate `max diag(M) <= 1e4` is a bound-based
  choice (logdet error ~ q * 1e4 * eps), supported by the cancellation sweep and the
  QR-vs-Cholesky fit agreement, not proven for every design. Follow-up: evaluation count.
