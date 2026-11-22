# FX3b twin contract: `convergence_status` is :converged, :boundary or
# :degenerate by the numeric rules of drmTMB `R/check.R`
# (`drm_twin_convergence_status_from_parts`, drmTMB PR #1564) and
# `docs/design/278-twin-convergence-status.md`. `is_converged` / `converged`
# are true only for :converged. A fit whose standard errors were turned off
# (`se = false`) skips the covariance rules: skipped is not failed (#1109).
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
    return DrmFit(family, blocks, names, theta, Matrix{Float64}(V), loglik, 10, converged,
                  Dict(:mu => fill(1.0, 10)), Dict(:mu => Float64.(0:9)), sc)
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
    @test convergence_status(_cs_fit(resd = 1e-4)) === :converged
    @test convergence_status(_cs_fit(resd = 0.5)) === :converged
    @test convergence_status(_cs_fit(rho12 = [0.5, 0.981])) === :boundary
    @test convergence_status(_cs_fit(rho12 = [-0.99, 0.0])) === :boundary
    @test convergence_status(_cs_fit(rho12 = [0.98, -0.98])) === :converged
    @test convergence_status(_cs_fit(rho12 = [NaN, 0.5])) === :converged       # non-finite ignored

    # boundary is tested BEFORE the Hessian (SHINICHI-1753-1118): a boundary fit with an
    # unusable covariance is :boundary, never :degenerate
    @test convergence_status(_cs_fit(V = fill(NaN, 2, 2), resd = 5e-5)) === :boundary
    # an unfinished optimiser wins over everything
    @test convergence_status(_cs_fit(converged = false, resd = 5e-5)) === :not_converged

    # is_converged / converged are `status === :converged`; the internal
    # optimiser-and-degeneracy gate keeps boundary fits
    b = _cs_fit(resd = 5e-5)
    @test !is_converged(b) && !converged(b)
    @test _converged_nondegenerate(b)
    @test b.converged
end

@testset "skipped uncertainty is not degenerate" begin
    nan_fit = _cs_fit(V = fill(NaN, 2, 2))
    @test nan_fit.uncertainty === :ok
    @test convergence_status(nan_fit) === :degenerate          # a FAILED covariance
    skipped = _withuncertainty(nan_fit, false)
    @test skipped.uncertainty === :skipped
    @test convergence_status(skipped) === :converged            # a SKIPPED one
    @test is_converged(skipped)
    @test _withuncertainty(nan_fit, true) === nan_fit           # se = true leaves the fit alone
    # rules 3-5 are skipped, rules 1, 2 and 6 are not
    @test convergence_status(_withuncertainty(_cs_fit(V = [0.01 0.0; 0.0 1.0e6]), false)) === :converged
    @test convergence_status(_withuncertainty(_cs_fit(V = fill(NaN, 2, 2), resd = 5e-5), false)) === :boundary
    @test convergence_status(_withuncertainty(_cs_fit(V = fill(NaN, 2, 2), converged = false), false)) === :not_converged
    @test convergence_status(_withgeometry(skipped, :count_mle_nonexistent)) === :degenerate
    # the record survives the copy helpers the fitters chain
    @test DRModels._withiterations(skipped, 7).uncertainty === :skipped
    g = _withgeometry(_cs_fit(), :count_mle_nonexistent)
    @test DRModels._withformula(_withuncertainty(g, false), nothing).geometry === :count_mle_nonexistent
end

@testset "Gaussian scale collapse is a Heywood boundary (SHINICHI-1753-1118)" begin
    collapsed = _cs_fit(family = Gaussian(),
                        scales = Dict(:sigma => fill(1e-12, 10)))
    @test convergence_status(collapsed) === :boundary
    @test convergence_status(_withuncertainty(collapsed, false)) === :boundary
    @test DRModels.convergence_info(collapsed).boundary_params ==
          [(name = "sigma", kind = :heywood)]
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

    # `se = false` on a Poisson relmat route returns an all-NaN covariance. That
    # is the caller's choice, not a failure: drmTMB reads `drm_control(se =
    # FALSE)` the same way.
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
    @test convergence_status(off) === :converged
    @test is_converged(off)
    on = drm(form, Poisson(); data = (; y = yc, x = xc, id), K = C)
    @test on.uncertainty === :ok
    @test convergence_status(on) === :converged
    # the same fit with its covariance thrown away AFTER the fact, uncertainty
    # still `:ok`, is a failed covariance
    failed = DrmFit(on.family, on.blocks, on.coefnames, on.theta, fill(NaN, size(on.vcov)),
                    on.loglik, on.nobs, on.converged, on.means, on.obs, on.scales)
    @test convergence_status(failed) === :degenerate
end

# #1186: binomial (quasi-)complete separation is a geometry flag, as drmTMB's
# `binomial_separation` reason in `drm_mle_nonexistent()` (R/mle-existence.R), so
# it is `:degenerate` even when the SE steps are skipped (`se = false`).
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

@testset "convergence_info record (SHINICHI-1753-1118)" begin
    info = convergence_info(_cs_fit())
    @test keys(info) == (:converged, :convergence_status, :boundary_params, :pd_hessian,
                         :max_gradient, :optimiser_code, :optimiser_message)
    @test info.converged && info.convergence_status === :converged
    @test isempty(info.boundary_params) && info.pd_hessian === true
    @test isnan(info.max_gradient)            # a hand-built fit stores no objective
    @test info.optimiser_code == 0 && info.optimiser_message isa String

    nc = convergence_info(_cs_fit(converged = false))
    @test !nc.converged && nc.convergence_status === :not_converged && nc.optimiser_code == 1
    @test convergence_info(_cs_fit(loglik = NaN)).optimiser_code == 1
    @test occursin("not finite", convergence_info(_cs_fit(loglik = NaN)).optimiser_message)

    b = convergence_info(_cs_fit(resd = 5e-5, rho12 = [0.99]))
    @test b.convergence_status === :boundary
    @test Set(b.boundary_params) == Set([(name = "sd:g", kind = :sd), (name = "rho12", kind = :cor)])

    d = convergence_info(_cs_fit(V = [1.0 1.0; 1.0 1.0]))
    @test d.convergence_status === :degenerate && d.pd_hessian === false
    @test ismissing(convergence_info(_withuncertainty(_cs_fit(), false)).pd_hessian)

    # relative gradient: a stored objective with a clearly non-stationary point is
    # :not_converged, a stationary one is :converged
    fitg(c) = begin
        f = _cs_fit(loglik = -100.0)
        DRModels._withnll(f, th -> 100.0 + c * th[1] + (th[2] - 0.2)^2)
    end
    @test convergence_status(fitg(0.0)) === :converged
    @test convergence_info(fitg(0.0)).max_gradient < 1e-6
    @test convergence_status(fitg(50.0)) === :not_converged
    @test convergence_info(fitg(50.0)).max_gradient > 0.1
    @test convergence_info(fitg(50.0)).max_gradient ≈ 0.5 rtol = 0.05   # |g| / max(1, |loglik|) = 50 / 100
end

@testset "REML fit: stored ML objective is not a stationarity oracle (b_sxz regression)" begin
    i = 1:40; x = sin.(i); z = cos.(1.3 .* i); u = mod.(0.7548776662 .* i, 1)
    d = (; y = 1 .+ 0.6 .* x .- 0.4 .* z .+ exp.(0.5 .* x .- 0.3 .* z) .* 0.8 .* (2 .* u .- 1), x, z)
    fit = drm(bf(@formula(y ~ x + z), @formula(sigma ~ x + z)), Gaussian(); data = d, method = :REML)
    @test convergence_status(fit) === :converged
    @test all(isfinite, DRModels._display_se(fit))
end
