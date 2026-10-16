# NB2 at the Poisson limit (#1072, #1069, #1039).
#
# When the data carry no overdispersion the NB2 likelihood is increasing in the
# size r = 1/sigma^2 all the way to infinity, so the MLE sits at the Poisson limit
# (log sigma -> -Inf). Three things went wrong there:
#   #1072  the fit came back with no flag at all (a clean, converged-looking fit
#          whose sigma is an optimiser-stopping artefact);
#   #1069  on Poisson-generated data with a random intercept the optimiser ran onto
#          that plateau although an interior sigma had a HIGHER likelihood;
#   #1039  the LBFGS / HagerZhang line search threw an uncaught AssertionError.
# Not covered here: location SEs with phi held at the limit (#1072 request 3) -- that
# is the SE path and is tracked separately.

using DRModels
using Test
using Random
import Distributions

# Underdispersed counts (Binomial variance < mean): the NB2 likelihood has no
# interior maximum in sigma, so the MLE is on the Poisson boundary by construction.
function _nb2_underdispersed(seed; n = 200)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    y = [Float64(rand(rng, Distributions.Binomial(20, 0.3))) for _ in 1:n]
    return (; y = y, x = x)
end

# The generator of the #1069 / #1039 reports (50 groups x 20, group SD 0.3).
function _nb2_poisson_glmm(seed; G = 50, m = 20)
    Random.seed!(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    b = 0.3 .* randn(G)
    y = [rand(Distributions.Poisson(exp(log(150.0) + b[g[i]]))) for i in 1:n]
    return (; y = Float64.(y), g = g)
end

@testset "NB2 at the Poisson limit" begin

    @testset "#1072 the boundary is flagged, not silent" begin
        d = _nb2_underdispersed(7)
        fit = @test_logs (:warn, r"no overdispersion") match_mode = :any drm(
            bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        @test DRModels._nb2_at_poisson_limit(fit)
        # the location estimates are the Poisson ones, so the estimate itself is fine
        pfit = drm(bf(@formula(y ~ 1 + x)), Poisson(); data = d)
        @test isapprox(coef(fit)[1:2], coef(pfit)[1:2]; atol = 1e-3)
    end

    @testset "a genuinely overdispersed fit is not flagged" begin
        rng = MersenneTwister(4)
        n = 600
        x = randn(rng, n)
        r = 1 / 0.6^2
        mu = exp.(1.4 .- 0.2 .* x)
        y = [Float64(rand(rng, Distributions.NegativeBinomial(r, r / (r + mu[i])))) for i in 1:n]
        fit = @test_logs min_level = Base.CoreLogging.Warn drm(
            bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y = y, x = x))
        @test !DRModels._nb2_at_poisson_limit(fit)
    end

    @testset "#1039 an AssertionError from one line search falls through to the next" begin
        calls = Symbol[]
        out = DRModels._with_optimizer_fallback([
            () -> (push!(calls, :first); throw(AssertionError("isfinite(phi_c)"))),
            () -> (push!(calls, :second); :ok),
        ])
        @test out === :ok
        @test calls == [:first, :second]
        # anything that is not an AssertionError is not swallowed
        @test_throws ErrorException DRModels._with_optimizer_fallback([
            () -> error("a real bug"), () -> :unreached])
        # and when every attempt fails the last error surfaces
        @test_throws AssertionError DRModels._with_optimizer_fallback([
            () -> throw(AssertionError("a")), () -> throw(AssertionError("b"))])
    end

    @testset "#1039 seed 20282004 returns a fit" begin
        d = _nb2_poisson_glmm(20282004)
        fit = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        @test fit isa DrmFit
        @test isfinite(loglik(fit))
    end

    @testset "#1069 seed 20282003: NB2 is not left on the plateau below an interior optimum" begin
        d = _nb2_poisson_glmm(20282003)
        nb = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        po = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = d)
        # NB2 nests Poisson, so its maximised logLik cannot be below the Poisson one;
        # an interior sigma (log sigma ~ -3.8) gains ~1.46 nats on this seed.
        @test loglik(nb) > loglik(po) + 1.0
        @test nb.theta[2] > -10      # sigma is interior, not on the Poisson plateau
    end
end
