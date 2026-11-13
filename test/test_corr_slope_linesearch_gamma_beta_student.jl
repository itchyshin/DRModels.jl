# Gamma, Beta and Student `(1 + x1 | g)`: the default HagerZhang line search used to throw
# `AssertionError: B > A` out of `drm()` on a dataset whose slope SD sits at the boundary. These routes
# now share the retry that LogNormal already had (`_optimize_corr_ranef`): the first attempt is the
# unchanged default call; only a fit that threw is retried from the start with a finite sentinel and
# BackTracking, then polished with the default line search. `converged` stays honest.
#
# Data (fixtures/corr_slope_retry/data.csv, 120 rows each, 12 groups of 10, formula
# `y ~ x1 + (1 + x1 | g)`, `sigma ~ 1`): gamma3, beta7 and student8. Julia 1.10.12 threw `B > A` on all
# three before the fix. Julia 1.12.7 and 1.13.3 do not throw on these datasets, so this test also passes
# on the old code there.
#
# There is no R reference: drmTMB 0.8.1 does not implement `(1 + x1 | g)` for Gamma, Beta or Student and
# refuses these models. The expected log-likelihoods below are the values this package gives on Julia
# 1.13.3 (where the first attempt succeeds). The retry on 1.10.12 reaches the same optimum, so the
# bound is loose (1e-5 on the logLik; the surface is flat along the slope SD).
using DRModels, Test, DelimitedFiles, Logging

const _CSRD = joinpath(@__DIR__, "fixtures", "corr_slope_retry")

function _csr_data(ds)
    raw, _ = readdlm(joinpath(_CSRD, "data.csv"), ','; header = true)
    rows = String.(raw[:, 1]) .== ds
    return (; y = Float64.(raw[rows, 2]), x1 = Float64.(raw[rows, 3]), g = String.(raw[rows, 4]))
end

@testset "Gamma, Beta, Student (1 + x1 | g): no line-search AssertionError" begin
    f = bf(@formula(y ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1))
    for (ds, fam, ll13) in (("gamma3", Gamma(), -176.89859444798117),
                            ("beta7", Beta(), 51.464125409720985),
                            ("student8", Student(), -115.67850809571075))
        data = _csr_data(ds)
        @test length(data.y) == 120
        fit = Logging.with_logger(Logging.NullLogger()) do
            try
                drm(f, fam; data)
            catch e
                e
            end
        end
        @test !(fit isa AssertionError)
        fit isa DRModels.DrmFit || continue
        @test isfinite(loglikelihood(fit))
        @test abs(loglikelihood(fit) - ll13) <= 1e-5
        @test fit.converged isa Bool
        @test all(isfinite, coef(fit))
    end
end

# The polish step of `_optimize_corr_ranef`. After the retry (finite sentinel + BackTracking) has not met
# the gradient tolerance, the fit is polished with the default line search and the better of the two
# results is kept. This ill-conditioned objective is one where BackTracking L-BFGS stops at its iteration
# limit far from the minimum, while the default line search reaches it, so a version without the polish
# step returns a clearly worse, unconverged result. The objective throws the line-search AssertionError
# on its first call only, to force the retry path.
@testset "_optimize_corr_ranef polishes an unconverged retry" begin
    Optim = DRModels.Optim
    illc(θ) = sum(10.0^i * θ[i]^2 for i in eachindex(θ)) + sum(θ .^ 4)
    θ0 = ones(12); g_tol = 1e-8
    ncalls = Ref(0)
    nll = θ -> (ncalls[] += 1; ncalls[] == 1 && throw(AssertionError("B > A")); illc(θ))
    res = DRModels._optimize_corr_ranef(nll, θ0, g_tol, zeros(2))
    # the retry alone (what the code would return without the polish step)
    retry = Optim.optimize(DRModels._safe_objective(illc), θ0,
        Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()), Optim.Options(g_tol = g_tol); autodiff = :forward)
    @test !DRModels.drm_optim_converged(retry)
    @test Optim.minimum(retry) > 1.0
    @test DRModels.drm_optim_converged(res)
    @test Optim.minimum(res) < 1e-10
    # no AssertionError: the first attempt is returned unchanged
    plain = Optim.optimize(illc, θ0, Optim.LBFGS(), Optim.Options(g_tol = g_tol); autodiff = :forward)
    @test Optim.minimizer(DRModels._optimize_corr_ranef(illc, θ0, g_tol, zeros(2))) == Optim.minimizer(plain)
end
