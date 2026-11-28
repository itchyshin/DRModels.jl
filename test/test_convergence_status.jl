# FX3b twin contract: `convergence_status` is :converged, :boundary or
# :degenerate by the numeric rules of drmTMB `R/check.R`
# (`drm_twin_convergence_status_from_parts`, drmTMB PR #1564) and
# `docs/design/278-twin-convergence-status.md`, extended by the current
# optimizer, relative-gradient, boundary and positive-definite-Hessian contract.
# `is_converged` / `converged` are true only for :converged.
using DRModels
using Test, Logging, Random, LinearAlgebra
import Distributions

using DRModels: _twin_inflated_se, _twin_vcov_positive_definite,
    _twin_standard_errors, _withuncertainty, _withgeometry, _converged_nondegenerate

# A hand-built fit, so each rule can be pinned at its exact cut. Poisson, so the
# Gaussian scale-collapse screen (`_nondegenerate_fit`) stays out of the way
# unless a test asks for it.
function _cs_fit(; V = Matrix(0.01I, 2, 2), loglik = -10.0, converged = true,
                 resd = nothing, rho12 = nothing, family = Poisson(),
                 scales = Dict{Symbol,Vector{Float64}}())
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:2]
    names = Pair{Symbol,Vector{String}}[:mu => ["(Intercept)", "x"]]
    theta = [0.1, 0.2]
    if resd !== nothing
        push!(blocks, :resd => 3:3); push!(names, :resd => ["g"]); push!(theta, log(resd))
        Vfull = fill(0.0, 3, 3); Vfull[1:2, 1:2] .= V[1:2, 1:2]
        Vfull[3, 3] = all(isfinite, V) ? 0.04 : NaN
        V = Vfull
    end
    sc = copy(scales)
    rho12 === nothing || (sc[:rho12] = Float64.(rho12))
    fit = DrmFit(family, blocks, names, theta, Matrix{Float64}(V), loglik, 10, converged,
                 Dict(:mu => fill(1.0, 10)), Dict(:mu => Float64.(0:9)), sc)
    objective = θ -> -Float64(loglik) + sum(abs2, θ .- theta) / 2
    objective_gradient = (g, θ) -> (g .= θ .- theta)
    return DRModels._withnll(fit, objective, objective_gradient)
end

@testset "twin numeric constants (drmTMB R/check.R)" begin
    @test DRModels._TWIN_INFLATED_SE_FLOOR == 50
    @test DRModels._TWIN_INFLATED_SE_RATIO == 1000
    @test DRModels._TWIN_RE_SD_BOUNDARY == 1e-4
    @test DRModels._TWIN_RHO12_BOUNDARY == 0.98
    @test DRModels._TWIN_SENTINEL_LOGLIK == -1e15
end

@testset "standard errors from vcov" begin
    se = _twin_standard_errors([4.0 0 0; 0 -1.0 0; 0 0 NaN])
    @test se[1] == 2.0
    @test isnan(se[2]) && isnan(se[3])   # negative or non-finite variance -> NaN
    @test isempty(_twin_standard_errors(zeros(0, 0)))
end

@testset "positive-definite covariance" begin
    @test _twin_vcov_positive_definite([1.0 0.2; 0.2 1.0])
    @test !_twin_vcov_positive_definite(zeros(0, 0))
    @test !_twin_vcov_positive_definite(ones(2, 3))
    @test !_twin_vcov_positive_definite([1.0 NaN; NaN 1.0])
    @test !_twin_vcov_positive_definite([1.0 1.0; 1.0 1.0])   # singular: min eigenvalue 0
    # R's eigen(symmetric = TRUE) reads the LOWER triangle: [1 5; 5 1] is
    # indefinite, although the upper triangle alone would give the identity.
    @test !_twin_vcov_positive_definite([1.0 0.0; 5.0 1.0])
end

@testset "inflated SE: SE >= 50 and SE >= 1000 x median(SEs < 50)" begin
    @test !_twin_inflated_se([0.1, 0.2, 0.15])
    @test _twin_inflated_se([0.1, 100.0])        # both cuts met exactly (>=)
    @test !_twin_inflated_se([0.1, 99.9])        # ratio cut missed
    @test !_twin_inflated_se([0.01, 49.9])       # floor cut missed
    @test _twin_inflated_se([0.1, 0.3, 200.0])   # median of the sub-50 SEs is 0.2
    # every finite SE >= 50: the reference falls back to the smallest SE
    @test _twin_inflated_se([80.0, 90.0, 8.0e4])
    @test !_twin_inflated_se([80.0, 90.0, 7.9e4])
    @test !_twin_inflated_se([NaN, Inf])         # no finite SE: not this rule
    @test _twin_inflated_se([NaN, 0.1, 500.0])   # non-finite entries are ignored
end

@testset "rule order and cuts on hand-built fits" begin
    base = _cs_fit()
    @test convergence_status(base) === :converged
    @test is_converged(base) && converged(base)

    # 1. optimiser gate and the -1e15 sentinel (strictly above)
    @test convergence_status(_cs_fit(converged = false)) === :not_converged
    @test convergence_status(_cs_fit(loglik = -1e15)) === :not_converged
    @test convergence_status(_cs_fit(loglik = -9.99e14)) === :converged
    @test convergence_status(_cs_fit(loglik = NaN)) === :not_converged
    @test convergence_status(_cs_fit(loglik = Inf)) === :not_converged

    # 2. geometry: a Poisson / NB2 MLE that does not exist
    @test convergence_status(_withgeometry(base, :count_mle_nonexistent)) === :degenerate

    # 3-5. covariance, finite SEs, inflated SE
    @test convergence_status(_cs_fit(V = fill(NaN, 2, 2))) === :degenerate
    @test convergence_status(_cs_fit(V = [1.0 1.0; 1.0 1.0])) === :degenerate
    @test convergence_status(_cs_fit(V = [0.01 0.0; 0.0 1.0e6])) === :degenerate   # SE 1000 vs 0.1
    @test convergence_status(_cs_fit(V = [0.01 0.0; 0.0 9.0e3])) === :converged    # SE 94.9 < 1000 x 0.1

    # 6. boundary: random-effect SD strictly below 1e-4, |rho12| strictly above 0.98
    @test convergence_status(_cs_fit(resd = 5e-5)) === :boundary
    # A direct `:resd` log-SD below -8 is already at the optimizer clamp, so
    # use the scale helper to test the separate strict 1e-4 cut without
    # confusing it with the more restrictive optimizer boundary.
    @test !DRModels._twin_re_sd_below_boundary(_cs_fit(resd = 1e-4))
    @test DRModels._twin_re_sd_below_boundary(_cs_fit(resd = prevfloat(1e-4)))
    @test !DRModels._twin_re_sd_below_boundary(_cs_fit(resd = nextfloat(1e-4)))
    @test convergence_status(_cs_fit(resd = 1e-4)) === :boundary
    @test convergence_status(_cs_fit(resd = 0.5)) === :converged
    @test convergence_status(_cs_fit(rho12 = [0.5, 0.981])) === :boundary
    @test convergence_status(_cs_fit(rho12 = [-0.99, 0.0])) === :boundary
    @test convergence_status(_cs_fit(rho12 = [0.98, -0.98])) === :converged
    @test convergence_status(_cs_fit(rho12 = [NaN, 0.5])) === :converged       # non-finite ignored

    # An explicitly detected boundary retains its status even when the
    # covariance is unavailable; it is never classified as converged.
    @test convergence_status(_cs_fit(V = fill(NaN, 2, 2), resd = 5e-5)) === :boundary
    # An unmapped inner conditional-mode clamp cannot be certified with an
    # outer-coordinate KKT score, so it fails closed even when `se = false`.
    clamped = _withuncertainty(
        _cs_fit(V = fill(NaN, 2, 2), resd = exp(-8),
                scales = Dict(:cond_logsd_wall => [-8.0])), false)
    @test convergence_status(clamped) === :degenerate
    @test convergence_status(_cs_fit(scales = Dict(:cond_logsd_wall => [3.0]))) === :degenerate

    # is_converged / converged are `status === :converged`; the internal
    # optimiser-and-degeneracy gate keeps boundary fits
    b = _cs_fit(resd = 5e-5)
    @test !is_converged(b) && !converged(b)
    @test _converged_nondegenerate(b)
    @test b.converged
end

@testset "unverified uncertainty cannot pass the Hessian convergence gate" begin
    nan_fit = _cs_fit(V = fill(NaN, 2, 2))
    @test nan_fit.uncertainty === :ok
    @test convergence_status(nan_fit) === :degenerate          # a FAILED covariance
    skipped = _withuncertainty(nan_fit, false)
    @test skipped.uncertainty === :skipped
    @test convergence_status(skipped) === :degenerate           # Hessian was not verified
    @test !is_converged(skipped)
    @test _withuncertainty(nan_fit, true) === nan_fit           # se = true leaves the fit alone
    # A skipped covariance still cannot pass the positive-definite Hessian gate.
    @test convergence_status(_withuncertainty(_cs_fit(V = [0.01 0.0; 0.0 1.0e6]), false)) === :degenerate
    @test convergence_status(_withuncertainty(_cs_fit(V = fill(NaN, 2, 2), resd = 5e-5), false)) === :boundary
    @test convergence_status(_withuncertainty(_cs_fit(V = fill(NaN, 2, 2), converged = false), false)) === :not_converged
    @test convergence_status(_withgeometry(skipped, :count_mle_nonexistent)) === :degenerate
    # the record survives the copy helpers the fitters chain
    @test DRModels._withiterations(skipped, 7).uncertainty === :skipped
    g = _withgeometry(_cs_fit(), :count_mle_nonexistent)
    @test DRModels._withformula(_withuncertainty(g, false), nothing).geometry === :count_mle_nonexistent
end

@testset "Gaussian scale collapse is degenerate (geometry rule)" begin
    collapsed = _cs_fit(family = Gaussian(),
                        scales = Dict(:sigma => fill(1e-12, 10)))
    @test convergence_status(collapsed) === :degenerate
    @test convergence_status(_withuncertainty(collapsed, false)) === :degenerate
end

@testset "real fits" begin
    rng = MersenneTwister(20261018)
    n = 60
    x = randn(rng, n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(rng, n)
    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
    @test convergence_status(fit) === :converged
    @test is_converged(fit) && converged(fit)
    @test occursin("convergence_status = converged",
                   sprint(show, MIME("text/plain"), fit))
    chk = with_logger(NullLogger()) do
        check_drm(fit)
    end
    @test chk.convergence_status === :converged && chk.converged && chk.ok

    # `se = false` on a Poisson relmat route returns an all-NaN covariance.
    # Optimisation may finish, but positive-definite curvature is unverified,
    # so convergence is not established under the shared rule.
    G = 20
    C = [0.5^abs(i - j) for i in 1:G, j in 1:G]
    id = repeat(1:G, inner = 5)
    xc = randn(rng, length(id))
    u = 0.8 .* (cholesky(C).L * randn(rng, G))
    yc = Float64.(rand.(rng, Distributions.Poisson.(exp.(0.3 .+ 0.3 .* xc .+ u[id]))))
    form = bf(@formula(y ~ x + relmat(1 | id)))
    off = drm(form, Poisson(); data = (; y = yc, x = xc, id), K = C, se = false)
    @test off.uncertainty === :skipped
    @test all(isnan, off.vcov)
    @test off.converged
    @test convergence_status(off) === :degenerate
    @test !is_converged(off)
    on = drm(form, Poisson(); data = (; y = yc, x = xc, id), K = C)
    @test on.uncertainty === :ok
    @test convergence_status(on) === :converged
    # the same fit with its covariance thrown away AFTER the fact, uncertainty
    # still `:ok`, is a failed covariance
    failed = DrmFit(on.family, on.blocks, on.coefnames, on.theta, fill(NaN, size(on.vcov)),
                    on.loglik, on.nobs, on.converged, on.means, on.obs, on.scales)
    failed = DRModels._withnll(failed, on.nll, on.nllgrad)
    @test convergence_status(failed) === :degenerate
end

# #1186: binomial (quasi-)complete separation is a geometry flag, as drmTMB's
# `binomial_separation` reason in `drm_mle_nonexistent()` (R/mle-existence.R), so
# it is `:degenerate` even when the SE steps are skipped (`se = false`). The
# non-separated control can pass after its objective curvature is checked.
@testset "binomial separation is a geometry flag, with or without SEs (#1186)" begin
    x = collect(range(-2, 2; length = 40))
    ysep = Float64.(x .> 0)                       # completely separated by x
    yctl = [0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,1,1,0,1,0, 0,1,1,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1]
    fit_binom(y; kw...) = with_logger(NullLogger()) do
        drm(bf(@formula(y ~ 1 + x)), Binomial(); data = (; y = Float64.(y), x = x), kw...)
    end
    off = fit_binom(ysep; se = false)
    @test off.uncertainty === :skipped
    @test off.geometry === :binomial_separation
    @test convergence_status(off) === :degenerate
    @test !is_converged(off)
    on = fit_binom(ysep)
    @test on.geometry === :binomial_separation
    @test convergence_status(on) === :degenerate
    # a non-separated control is untouched
    ctl_off = fit_binom(yctl; se = false)
    @test ctl_off.uncertainty === :skipped
    @test ctl_off.geometry === :ok
    @test convergence_status(ctl_off) === :converged
    ctl_on = fit_binom(yctl)
    @test ctl_on.geometry === :ok
    @test convergence_status(ctl_on) === :converged
    # the guard still sets the Inf SE that it set before
    @test any(isinf, diag(on.vcov))
end
