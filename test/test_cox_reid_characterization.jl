# test_cox_reid_characterization.jl — standalone #441 probe tripwire.
#
# NOT registered in test/runtests.jl. Documents the punch-through hook and the
# routes that remain ML-only.
#
# The tripwire FIRED as designed: this file used to assert that
# `drm(::Poisson; method = :REML)` was rejected family-wide, and #443 admitted it on the
# scalar `(1 | g)` GHQ route. That assertion has moved to
# `test/test_cox_reid_poisson_ranef.jl`, which owns the shipped cell; what stays here is
# the ML default and the routes still awaiting certification.
#
# Not a recovery campaign. Not a drmTMB / GLLVM numeric import.
#
# Run:  julia --project=. -e 'include("test/test_cox_reid_characterization.jl")'
#
# Worked example (ML is still the default):
#
#   using DRModels
#   G, m = 8, 4
#   g = repeat(1:G, inner = m); x = randn(G * m)
#   y = Float64.(rand.(Ref(Random.default_rng()),
#                      Poisson.(exp.(0.3 .+ 0.2 .* x))))
#   fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y, x, g))
#   estimation_method(fit) === :ML

using DRModels
using Test, Random, LinearAlgebra
import Distributions

const D = DRModels

@testset "Cox–Reid characterization (#441) — hook + uncertified routes" begin
    @testset "ML is still the Poisson default; (1|g) REML is opt-in (#443)" begin
        rng = MersenneTwister(441)
        G, m = 8, 4
        n = G * m
        g = repeat(1:G, inner = m)
        x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.2 * xi))) for xi in x])
        data = (; y, x, g)
        form = D.bf(D.@formula(y ~ x + (1 | g)))
        fit = D.drm(form, D.Poisson(); data = data)
        @test D.estimation_method(fit) === :ML
        # #443 admitted `:REML` HERE and only here. Asking for it must not silently
        # return an ML fit — the tag has to follow the request.
        fit_reml = D.drm(form, D.Poisson(); data = data, method = :REML)
        @test D.estimation_method(fit_reml) === :REML
    end

    @testset "the uncertified Poisson routes still refuse :REML" begin
        rng = MersenneTwister(4412)
        G, m = 8, 4
        n = G * m
        g = repeat(1:G, inner = m)
        x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.2 * xi))) for xi in x])
        data = (; y, x, g)
        # Crossed intercepts: the hook exists, the bias is uncharacterised.
        h = repeat(1:4, outer = div(n, 4))
        err = try
            D.drm(D.bf(D.@formula(y ~ x + (1 | g) + (1 | h))), D.Poisson();
                  data = (; y, x, g, h), method = :REML)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("(1 | g)", sprint(showerror, err))
    end

    @testset "sparse-Laplace hook: nllgrad + generic penalty" begin
        rng = MersenneTwister(4411)
        ntip, per = 12, 4
        tree = D.random_balanced_tree(ntip; branch_length = 0.25)
        species = repeat(1:ntip, inner = per)
        n = length(species)
        x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.25 + 0.2 * x[i]))) for i in 1:n])
        fit = D.drm(D.bf(D.@formula(y ~ x + phylo(1 | species))), D.Poisson();
                    data = (; y, x, species), tree = tree, se = false)
        @test D.estimation_method(fit) === :ML
        @test fit.nllgrad !== nothing
        pμ = length(D.coef(fit, :mu))
        θ̂ = D.coef(fit)
        @test pμ == 2
        @test length(θ̂) == 3
        grad_fn = θ -> (g = zeros(length(θ)); fit.nllgrad(g, θ); copy(g))
        pen = D._glsp_reml_penalty(grad_fn, θ̂, pμ)
        @test isfinite(pen) && pen < 1e16
        @test pen > 0
    end

    @testset "Gaussian reduction: generic CR ≈ #440 REML (oracle, read-only)" begin
        rng = MersenneTwister(4242)
        G, n_each = 12, 5
        n = G * n_each
        g = repeat(1:G, inner = n_each)
        x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ 0.8 .* randn(rng, G)[g] .+ 0.7 .* randn(rng, n)
        data = (; y, x, g)
        form = D.bf(D.@formula(y ~ x + (1 | g)))
        fit_ml = D.drm(form, D.Gaussian(); data = data)
        fit_reml = D.drm(form, D.Gaussian(); data = data, method = :REML)
        @test D.estimation_method(fit_ml) === :ML
        @test D.estimation_method(fit_reml) === :REML
        θ̂ml = D.coef(fit_ml)
        pμ = length(D.coef(fit_ml, :mu))
        # #440 oracle exists and ML ≠ REML on the VC. Full |θ_CR − θ_REML|
        # = 2.871e-06 lives in bench/out (Cell B), not this tripwire.
        @test maximum(abs.(θ̂ml .- D.coef(fit_reml))) > 1e-4
        if fit_ml.nllgrad !== nothing
            grad_fn = θ -> (g = zeros(length(θ)); fit_ml.nllgrad(g, θ); copy(g))
            pen = D._glsp_reml_penalty(grad_fn, θ̂ml, pμ)
            @test isfinite(pen) && 0 < pen < 1e16
        end
    end
end
