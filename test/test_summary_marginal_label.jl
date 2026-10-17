# #1071: the printed fit names the marginal approximation behind its logLik, and
# #949 regression guard: `show` prints the trusted `is_converged`. The label must
# not change any logLik value. Reviewers asked for the default `:LA` label on
# both `summary` and `coeftable` as well as `show` — Poisson `(1|g)` never
# stores BLUPs, so the check cannot use `fit.ranef === nothing`.

using DRModels
using Test
using Random
import Distributions

@testset "show names the marginal approximation (#1071)" begin
    rng = MersenneTwister(1071)
    G, m = 30, 8
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    u = 0.6 .* randn(rng, G)
    y = Float64[rand(rng, Distributions.Poisson(exp(0.3 + 0.4 * x[i] + u[g[i]]))) for i in eachindex(g)]
    data = (; y, x, g = string.(g))
    f = bf(@formula(y ~ 1 + x + (1 | g)))

    fit_la = drm(f, Poisson(); data = data)
    fit_lp = drm(f, Poisson(); data = data, marginal = :Laplace)
    shown(fit) = sprint(show, MIME"text/plain"(), fit)
    # `summary(fit)` is `coeftable(fit)`; both print paths must name `:LA`.
    printed_summary(fit) = sprint(show, MIME"text/plain"(), summary(fit))
    printed_coeftable(fit) = sprint(show, MIME"text/plain"(), coeftable(fit))

    @test occursin("marginal :LA", shown(fit_la))
    @test occursin("marginal :LA", printed_summary(fit_la))
    @test occursin("marginal :LA", printed_coeftable(fit_la))
    @test occursin("marginal :Laplace", shown(fit_lp))
    @test occursin("marginal :Laplace", printed_summary(fit_lp))
    @test occursin("marginal :Laplace", printed_coeftable(fit_lp))
    @test !occursin("marginal :Laplace", shown(fit_la))
    @test !occursin("marginal :Laplace", printed_summary(fit_la))
    @test !occursin("marginal :Laplace", printed_coeftable(fit_la))
    @test !occursin("marginal :LA,", shown(fit_lp))
    @test !occursin("marginal :LA,", printed_summary(fit_lp))
    @test !occursin("marginal :LA,", printed_coeftable(fit_lp))

    # Values are untouched and the two approximations genuinely differ.
    @test loglik(fit_la) == fit_la.loglik
    @test loglik(fit_la) != loglik(fit_lp)

    # #949: show reports the trusted accessor.
    @test occursin("converged = " * string(is_converged(fit_la)), shown(fit_la))

    # A fixed-effect fit has no marginal approximation to name: output unchanged.
    fit_fe = drm(bf(@formula(y ~ 1 + x)), Poisson(); data = data)
    @test !occursin("marginal :", shown(fit_fe))
    @test !occursin("marginal :", printed_summary(fit_fe))
    @test !occursin("marginal :", printed_coeftable(fit_fe))
end
