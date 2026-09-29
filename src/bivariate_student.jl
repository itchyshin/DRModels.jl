# bivariate_student.jl — bivariate Student-t residual/scatter model, the Julia
# twin of drmTMB's `biv_student()`.
#
# Two heavy-tailed real-valued responses sharing one scale-mixture:
#
#     Y = mu + diag(sigma) * Z * sqrt(nu / chisq_nu),   Z ~ N(0, R)
#
# which is exactly the multivariate-t with location `mu`, SCALE matrix
# Sigma = diag(sigma) R diag(sigma), and degrees of freedom `nu`. drmTMB's own
# `simulate()` for this family writes precisely that draw, which is what pins the
# parameterisation here (methods.R): `sigma1`/`sigma2` are **scale** parameters,
# not marginal SDs (the marginal SD is sigma * sqrt(nu/(nu-2))), and `rho12` is
# the **scatter** correlation.
#
# NU IS STRUCTURALLY SHARED. The scale mixture uses a single scalar mixing
# variable, so one `nu` governs both margins simultaneously — there is no
# per-margin `nu1`/`nu2` under the exact density. Per-margin tail heaviness would
# require a normal-variance-mixture copula, which gives up the exact density
# altogether. (Deep-research note dr19; drmTMB's own table says
# "shared-Student-t" for the same reason.)
#
# ZERO CORRELATION IS NOT INDEPENDENCE. For any finite `nu` the two components
# remain dependent even at rho12 = 0 — independence only appears in the Gaussian
# limit. That is a documentation obligation, not a numerical detail.
#
# The likelihood is CLOSED FORM: the bivariate-t density is analytic, so
# ForwardDiff + LBFGS is appropriate. dr19 documents an RQMC/derivative-free
# failure mode, but that concerns multivariate-t *probabilities* (rectangle
# integrals), not the density this likelihood evaluates — do not import that
# workaround here.
#
# WHY STRUCTURED MARKERS (phylo/relmat/animal/spatial) STAY REJECTED (#471).
# The bivariate Gaussian q=2/q=4 structured routes are exact or verified-Laplace
# because a Gaussian group-level random effect composed with a GAUSSIAN
# conditional response has a closed-form (or hand-derived, analytically
# differentiated) marginal — that is what `gaussian_bivariate.jl`,
# `coevolution_q.jl`, and the sparse augmented-state Laplace engine in
# `sparse_aug_plsm.jl`/`fit_q4_sparse_tmb.jl` are built on. None of that carries
# over to a Student-t conditional density: a Gaussian random intercept under a
# heavy-tailed, per-row scale-mixture likelihood has NO closed-form marginal, and
# there is no verified engine in this codebase (sparse or otherwise) whose
# per-leaf likelihood is bivariate-t rather than bivariate-Gaussian to reuse. The
# instruction for this family of markers is to MIRROR the established design,
# not invent a parallel one — building a bespoke, unverified joint-Laplace
# engine for this one PR would be exactly that parallel design, on the family
# this project's own history says is the least forgiving place to get a scale
# convention wrong silently. drmTMB agrees this is unsolved, not merely
# unported: `biv_student()`'s own R implementation raises "currently allows
# fixed-effect formulas only; random and structured effects are deferred" for
# the identical request (drmTMB 0.7.0, checked directly), so there is no
# reference implementation on either side of the port to mirror. The residual
# (fixed-effects-only) route below is unaffected and stays parity-verified.

"""
    drm(f::BivariateDrmFormula, ::Student; data, g_tol = 1e-8, method = :ML)

Fit a bivariate **Student-t** scatter-correlation model — drmTMB's
`biv_student()`.

`mu1`/`mu2` are locations (identity link), `sigma1`/`sigma2` are **scale**
parameters (log link) — not standard deviations; for `ν > 2` the marginal
`SD = σ·sqrt(ν/(ν−2))` — `rho12` is the **scatter** correlation (guarded
`atanh` link), and `nu` is the shared degrees of freedom on the `logm2` scale
`ν = 2 + exp(η)`, so `ν > 2` and the variance is finite.

`nu` is shared across both responses by construction, and **zero `rho12` does
not mean independent margins** at finite `ν`.

## Structured markers (`phylo`/`relmat`/`animal`/`spatial`) — NOT implemented

This route stays residual-only: no random effects, no structured markers, and
no REML. This is a deliberate rejection, not a missing port. The bivariate
Gaussian structured routes are exact (q=2) or a verified Laplace approximation
with a hand-derived, analytically differentiated per-leaf density (q=4); both
depend on the conditional response being Gaussian, which a Student-t density
is not. A Gaussian group-level random effect under a heavy-tailed,
per-row scale-mixture likelihood has no closed-form marginal, and there is no
verified non-Gaussian engine in this codebase to reuse for it — building one
for this route alone would be inventing a parallel, unverified numerical design
rather than mirroring the established one, on exactly the family (correlated
structured scale) this project has already gotten a scale convention silently
wrong once. drmTMB has the same limit, and that is the load-bearing half of this
rejection, so it is written down reproducibly rather than asserted.
**Re-verified live 2026-08-25** against the installed drmTMB 0.7.0:

```r
library(drmTMB); library(ape)
tr <- compute.brlen(stree(8, type = "balanced"), method = "Grafen")
tr\$tip.label <- paste0("s", 1:8)
d <- data.frame(y1 = rnorm(40), y2 = rnorm(40), x = rnorm(40),
                sp = factor(rep(paste0("s", 1:8), each = 5)))
# fixed effects only -> ACCEPTED
drmTMB(bf(mu1 = y1 ~ x, mu2 = y2 ~ x, sigma1 = ~1, sigma2 = ~1,
          nu = ~1, rho12 = ~1), family = biv_student(), data = d)
# add a structured marker -> REFUSED
drmTMB(bf(mu1 = y1 ~ x + phylo(1 | sp, tree = tr), mu2 = y2 ~ x, ...),
       family = biv_student(), data = d)
#> `biv_student()` currently allows fixed-effect formulas only; random and
#>  structured effects are deferred.
```

So there is no reference implementation on **either** side of the port. For a
parity goal that matters more than the numerical argument above: a parity gap
cannot be closed against a capability the reference package does not have, and
implementing one unilaterally would mean inventing the answer this port exists
to mirror.

```julia
f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
       sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
       nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1))
fit = drm(f, Student(); data = dat)
2 + exp(coef(fit, :nu)[1])   # estimated shared degrees of freedom
```
"""
function drm(f::BivariateDrmFormula, fam::Student; data, g_tol::Real = 1e-8,
             method::Symbol = :ML)
    method === :ML ||
        throw(ArgumentError("drm: the bivariate Student-t route implements ML only " *
            "(got method = :$method). drmTMB's `biv_student()` first slice has no " *
            "random effects, so there is nothing for REML to integrate out."))
    rhs = Dict(f.forms)
    _, structured_marker = _bivariate_q4_marker(rhs)
    structured_marker === nothing ||
        throw(ArgumentError("drm: the bivariate Student-t route is residual-only — " *
            "`phylo`/`relmat`/`animal`/`spatial` markers are not implemented for the " *
            "bivariate `Student`. This is a deliberate rejection, not a missing port: " *
            "the bivariate Gaussian structured routes rely on a Gaussian conditional " *
            "response (closed-form q=2, or a hand-derived Laplace q=4), and neither " *
            "carries over to a Student-t density — there is no verified non-Gaussian " *
            "engine in this codebase to mirror, and drmTMB's own `biv_student()` " *
            "defers random/structured effects too (checked directly against drmTMB " *
            "0.7.0). Use `Gaussian()` for the structured bivariate engines, or " *
            "`LogNormal()` if the response is lognormal (it delegates to the same " *
            "verified Gaussian machinery on log(y))."))
    return _fit_bivariate_residual(f, fam, data, rhs, g_tol)
end

function _fit_bivariate_residual(f::BivariateDrmFormula, fam::Student, data, rhs,
                                 g_tol::Real)
    y1, X1, nm1 = _design(f.response1, rhs[:mu1], data)
    y2, X2, nm2 = _design(f.response2, rhs[:mu2], data)
    _, Xs1, nms1 = _design(f.response1, rhs[:sigma1], data)
    _, Xs2, nms2 = _design(f.response1, rhs[:sigma2], data)
    _, Xν, nmν = _design(f.response1, get(rhs, :nu, ConstantTerm(1)), data)
    _, Xr, nmr = _design(f.response1, rhs[:rho12], data)

    n = length(y1)
    obs1 = _observed_response_mask(y1)
    obs2 = _observed_response_mask(y2)
    n_like = count(obs1 .| obs2)
    n_like > 0 ||
        throw(ArgumentError("drm: at least one bivariate Student-t response cell must be observed"))
    count(obs1 .& obs2) > 0 ||
        throw(ArgumentError("drm: at least one row must observe both bivariate Student-t " *
            "responses to estimate rho12"))

    # Parameter order mirrors drmTMB's dpar order for `biv_student()`:
    # mu1, mu2, sigma1, sigma2, NU, rho12 — nu before rho12.
    ps = (size(X1, 2), size(X2, 2), size(Xs1, 2), size(Xs2, 2), size(Xν, 2), size(Xr, 2))
    offs = cumsum([0, ps...])
    rng(k) = (offs[k]+1):offs[k+1]

    function nll(θ)
        b1 = θ[rng(1)]; b2 = θ[rng(2)]; bs1 = θ[rng(3)]
        bs2 = θ[rng(4)]; bν = θ[rng(5)]; br = θ[rng(6)]
        η1 = X1 * b1; η2 = X2 * b2
        ls1 = Xs1 * bs1; ls2 = Xs2 * bs2
        ην = Xν * bν; ηr = Xr * br
        s = zero(eltype(θ))
        @inbounds for i in 1:n
            ν = 2 + exp(ην[i])                 # logm2 link ⇒ ν > 2, finite variance
            if obs1[i] && obs2[i]
                ρ = RHO_GUARD * tanh(ηr[i])
                om = 1 - ρ * ρ
                z1 = (y1[i] - η1[i]) * exp(-ls1[i])
                z2 = (y2[i] - η2[i]) * exp(-ls2[i])
                d2 = (z1 * z1 - 2ρ * z1 * z2 + z2 * z2) / om   # Mahalanobis on the scatter
                # −log f₂(y) for the bivariate (p = 2) Student-t. The normalising
                # constant loggamma((ν+2)/2) − loggamma(ν/2) is Γ(ν/2+1)/Γ(ν/2) in
                # log space, an EXACT Gamma recursion identity (Γ(x+1) = xΓ(x)) for
                # every ν > 0 — not an asymptotic approximation — so it collapses to
                # log(ν/2) with no cancellation. Substituting that identity here
                # (−log(ν/2) + log ν + log π = log(2π)) removes the catastrophic
                # cancellation `loggamma((ν+2)/2) - loggamma(ν/2)` suffered for large
                # ν (two ~ν·log ν-sized terms subtracted to leave an O(log ν) result,
                # same failure mode as #721/#820): at ν = 1e16 the naive difference
                # was off by ~36 nats, so the bivariate loglik silently diverged from
                # the correct bivariate-Normal limit. Reproduced and fixed on branch
                # claude/twin-gap-bivstudent (test/test_bivariate_student_large_nu.jl).
                s += log(2π) + ls1[i] + ls2[i] + 0.5 * log(om) +
                     ((ν + 2) / 2) * log1p(d2 / ν)
            elseif obs1[i]
                # A margin of a bivariate-t is a univariate t with the SAME ν. This
                # loggamma((ν+1)/2) − loggamma(ν/2) is NOT an exact-recursion case
                # (arguments differ by 1/2, not 1) — it is exactly the univariate
                # Student cancellation from #721/#820, so reuse the stable large-ν
                # evaluator `_student_logpdf_std` from student.jl instead of
                # re-deriving it by hand.
                z1 = (y1[i] - η1[i]) * exp(-ls1[i])
                s += -_student_logpdf_std(z1, ην[i]) + ls1[i]
            elseif obs2[i]
                z2 = (y2[i] - η2[i]) * exp(-ls2[i])
                s += -_student_logpdf_std(z2, ην[i]) + ls2[i]
            end
        end
        return s
    end

    θ0 = zeros(offs[end])
    X1_obs = Matrix{Float64}(X1[obs1, :]); y1_obs = Vector{Float64}(y1[obs1])
    X2_obs = Matrix{Float64}(X2[obs2, :]); y2_obs = Vector{Float64}(y2[obs2])
    β1 = X1_obs \ y1_obs
    β2 = X2_obs \ y2_obs
    θ0[rng(1)] .= β1
    θ0[rng(2)] .= β2
    θ0[rng(3)][1] = _seed_ls(y1_obs - X1_obs * β1, y1_obs)
    θ0[rng(4)][1] = _seed_ls(y2_obs - X2_obs * β2, y2_obs)
    θ0[rng(5)][1] = log(6.0)     # ν ≈ 8: heavy-tailed but comfortably finite-variance
    # rho12 block stays at 0 ⇒ ρ = 0.

    res = Optim.optimize(nll, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol);
                         autodiff = :forward)
    θ̂ = Optim.minimizer(res)
    V = _vcov_from_hessian(ForwardDiff.hessian(nll, θ̂); context = "bivariate Student-t")

    blocks = [:mu1 => rng(1), :mu2 => rng(2), :sigma1 => rng(3), :sigma2 => rng(4),
              :nu => rng(5), :rho12 => rng(6)]
    names = [:mu1 => nm1, :mu2 => nm2, :sigma1 => nms1, :sigma2 => nms2,
             :nu => nmν, :rho12 => nmr]
    means = Dict(:mu1 => X1 * θ̂[rng(1)], :mu2 => X2 * θ̂[rng(2)])
    obs = Dict(:mu1 => Vector{Float64}(y1), :mu2 => Vector{Float64}(y2))
    scales = Dict(:sigma1 => exp.(Xs1 * θ̂[rng(3)]),
                  :sigma2 => exp.(Xs2 * θ̂[rng(4)]),
                  :nu => 2 .+ exp.(Xν * θ̂[rng(5)]),
                  :rho12 => RHO_GUARD .* tanh.(Xr * θ̂[rng(6)]))
    return _withiterations(
        _withformula(
            _withnll(DrmFit(fam, blocks, names, θ̂, V, -nll(θ̂), n_like,
                            Optim.converged(res), means, obs, scales), nll), f),
        Optim.iterations(res))
end

# --- Parametric-bootstrap replicate draw (#766) ------------------------------
#
# `gaussian_core.jl`'s generic `_simulate_once` special-cases bivariate
# GAUSSIAN fits (`fam isa Gaussian && haskey(fit.scales, :sigma1)`) before
# falling through to `μ = fit.means[:mu]` for every other family. A bivariate
# Student-t fit (`biv_student()`) stores `:mu1`/`:mu2` and `:sigma1`/`:sigma2`
# instead — the same shape as bivariate Gaussian, never `:mu`/`:sigma` — so
# that fallthrough throws `KeyError: key :mu not found` on the very first
# `simulate(fit)` call. Because the parametric bootstrap (`bootstrap_result` /
# `bootstrap_ci`) draws its replicate response via `simulate(fit0; rng)` before
# any refit is attempted, this failure recurs immediately on EVERY replicate —
# exactly the "0 s abort, no fit work attempted" symptom in #766. (Profile
# intervals do not go through `simulate` at all, and were already fixed by
# claude/twin-gap-bivstudent's large-ν density correction; this is the
# separate bootstrap-side gap #766 also names.)
#
# This method is dispatched by `fit::DrmFit{Student}`, more specific than the
# generic `fit::DrmFit` method in gaussian_core.jl, so it does not disturb the
# univariate `Student` branch there for any other fit — it reimplements that
# one case (identical formula) alongside the new bivariate one, exactly as
# gaussian_core.jl itself distinguishes bivariate vs. univariate Gaussian by
# the presence of `:sigma1`.
#
# Bivariate draw follows this file's own docstring: Y = mu + diag(sigma) * Z *
# sqrt(nu / chisq_nu), Z ~ N(0, R). `nu` is a single scalar mixing variable
# shared across both margins PER ROW (see "NU IS STRUCTURALLY SHARED" above),
# so one `Chisq(nu[i])` draw scales both components of row i.
function _simulate_once(fit::DrmFit{Student}, rng; mu = nothing, sigma = nothing)
    if haskey(fit.scales, :sigma1)   # bivariate biv_student() fit
        μ1, μ2 = fit.means[:mu1], fit.means[:mu2]
        σ1, σ2 = fit.scales[:sigma1], fit.scales[:sigma2]
        ρ, ν = fit.scales[:rho12], fit.scales[:nu]
        n = length(μ1)
        z1 = randn(rng, n)
        z2 = randn(rng, n)
        sh = [sqrt(ν[i] / rand(rng, Distributions.Chisq(ν[i]))) for i in 1:n]
        return Dict(:mu1 => μ1 .+ σ1 .* z1 .* sh,
                    :mu2 => μ2 .+ σ2 .* (ρ .* z1 .+ sqrt.(1 .- ρ .^ 2) .* z2) .* sh)
    end
    # Univariate Student-t: identical to gaussian_core.jl's generic branch.
    μ = mu === nothing ? fit.means[:mu] : mu
    σ = sigma === nothing ? _scale_vector(fit, :sigma) : sigma
    ν = _scale_vector(fit, :nu)
    n = length(μ)
    return Float64[μ[i] + σ[i] * rand(rng, Distributions.TDist(ν[i])) for i in 1:n]
end
