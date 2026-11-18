# Robust-entry leftovers (2026-11-08):
#   0. Gamma `y ~ x + (1 | id)`, `sigma ~ 1 + (1 | site)` with SD formulas (the Gaussian / NB2 / Gamma
#      sigma-SD Laplace engine, `gaussian_sigma_sd_laplace.jl`): a non-PD Hessian in the Newton
#      step raised PosDefException out of `drm()`; it now gives a fit with converged = false.
#   1. `ml_loglik(fit)` / `reml_loglik(fit)` follow `loglik(fit)`: NaN with a warning for a failed
#      fit (sentinel, non-finite or runaway stored value); converged fits and the documented
#      NaN of `reml_loglik` on an ML fit are unchanged; raw values stay in the fields.
using DRModels, Test, Logging, StableRNGs, DelimitedFiles
import Distributions

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

function _with_fields(f; kw...)
    names = fieldnames(typeof(f))
    vals = Any[getfield(f, k) for k in names]
    for (k, v) in kw
        vals[findfirst(==(k), names)] = v
    end
    return typeof(f).name.wrapper(vals...)
end

@testset "ml_loglik / reml_loglik: NaN with a warning on a failed fit, unchanged otherwise" begin
    rng = StableRNG(11)
    x = randn(rng, 60)
    d = (; y = 1 .+ 0.5 .* x .+ randn(rng, 60), x)
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    ml = drm(f, Gaussian(); data = d)
    reml = drm(f, Gaussian(); data = d, method = :REML)
    # converged fits: no warning, same value as the field
    @test @test_logs(ml_loglik(ml)) == ml.ml_loglik
    @test @test_logs(ml_loglik(reml)) == reml.ml_loglik
    @test @test_logs(reml_loglik(reml)) == reml.reml_loglik
    @test isfinite(reml_loglik(reml)) && isfinite(ml_loglik(ml))
    # an ML fit: reml_loglik is the documented NaN, silently
    @test isnan(@test_logs(reml_loglik(ml)))
    # failed fits
    for bad in (-1e18, -Inf, 3.6e76)
        fm = _with_fields(ml; ml_loglik = bad, converged = false)
        v = @test_logs (:warn, r"ml_loglik\(fit\) is NaN") match_mode = :any ml_loglik(fm)
        @test isnan(v)
        @test fm.ml_loglik == bad                                  # raw value stays on the object
        fr = _with_fields(reml; reml_loglik = bad, converged = false)
        v = @test_logs (:warn, r"reml_loglik\(fit\) is NaN") match_mode = :any reml_loglik(fr)
        @test isnan(v)
        @test fr.reml_loglik == bad
    end
    # a non-converged fit with a plausible value is not touched
    plausible = _with_fields(ml; converged = false)
    @test @test_logs(ml_loglik(plausible)) == ml.ml_loglik
    # a converged fit with a large value is not touched
    big = _with_fields(ml; ml_loglik = 3.6e76, converged = true)
    @test @test_logs(ml_loglik(big)) == 3.6e76
end

@testset "Gamma sigma-SD Laplace: non-PD Newton Hessian gives converged = false, not PosDefException" begin
    raw = readdlm(joinpath(@__DIR__, "fixtures", "entry_robust", "gamma_sigma_sd_nonpd_data.csv"), ','; header = true)[1]
    data = (; y = Float64.(raw[:, 1]), x = Float64.(raw[:, 2]), id = string.(raw[:, 3]),
            site = string.(raw[:, 4]), v = Float64.(raw[:, 5]))
    f = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | site)),
           @formula(sd(id) ~ 1), @formula(sd(site, sigma) ~ v))
    # fixture: 5 sites x 1 obs, one outlier (y[1] ~ 1e4) among values ~ 1e-3. Before the guard this
    # raised `PosDefException: matrix is not positive definite` from the ForwardDiff Newton step.
    fit = _quiet(() -> drm(f, Gamma(); data, se = false))
    @test fit isa DRModels.DrmFit
    @test fit.converged == false
    @test !DRModels.is_converged(fit)
    ll = _quiet(() -> loglikelihood(fit))
    @test isnan(ll) || ll > -1e15
end
