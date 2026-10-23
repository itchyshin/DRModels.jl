# test_cox_reid_poisson_ranef.jl — opt-in Cox–Reid REML on Poisson `(1 | g)` (#443).
#
# NOT registered in test/runtests.jl (that waits on the Option A sibling), so run it
# directly:
#
#   julia --project=. -e 'include("test/test_cox_reid_poisson_ranef.jl")'
#
# Scope: the ONE cell certified by the #441 scoping probe — a scalar random intercept
# integrated by 32-node Gauss–Hermite quadrature (`_fit_poisson_ranef`, src/poisson.jl).
# ML remains the default everywhere. Cox–Reid over-corrects at larger G (probe Cell A:
# +4.38% at G=40), which is why this is opt-in and why the tests below assert a
# DIRECTION and a mechanism rather than a recovery target.
#
# Worked example:
#
#   using DRModels, Random, Distributions
#   rng = MersenneTwister(443)
#   G, m = 10, 6
#   g = repeat(1:G, inner = m)
#   x = randn(rng, G * m)
#   b = 0.6 .* randn(rng, G)
#   y = Float64.([rand(rng, Poisson(exp(0.3 + 0.2 * x[i] + b[g[i]]))) for i in eachindex(x)])
#   form = bf(@formula(y ~ x + (1 | g)))
#   fit_ml   = drm(form, Poisson(); data = (; y, x, g))                   # ML (default)
#   fit_reml = drm(form, Poisson(); data = (; y, x, g), method = :REML)   # opt-in Cox–Reid
#   estimation_method(fit_reml) === :REML

using DRModels
using Test, Random, LinearAlgebra
import Distributions

const D = DRModels

# One Poisson `(1 | g)` draw with a genuine scalar cluster effect, so σ̂_b is identified
# (a σ→0 boundary fit has nothing for a restricted objective to correct).
function _cr_poisson_draw(seed::Int; G::Int = 10, m::Int = 6, σb::Float64 = 0.6)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    n = G * m
    x = randn(rng, n)
    b = σb .* randn(rng, G)
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.2 * x[i] + b[g[i]])))
                  for i in 1:n])
    return (; y, x, g)
end

_sigma_b(fit) = exp(D.coef(fit)[end])

@testset "Cox–Reid REML on Poisson (1|g) (#443)" begin
    form = D.bf(D.@formula(y ~ x + (1 | g)))

    @testset "opt-in :REML is admitted and tagged" begin
        data = _cr_poisson_draw(443)
        fit = D.drm(form, D.Poisson(); data = data, method = :REML)
        @test D.estimation_method(fit) === :REML
        @test D.is_converged(fit)
        # `_withreml` is a tag: it records BOTH likelihoods and puts the restricted one
        # in the public slot. Both must be real numbers, and they must differ — an
        # identical pair would mean the penalty never entered the objective.
        @test isfinite(fit.reml_loglik)
        @test isfinite(fit.ml_loglik)
        @test fit.loglik == fit.reml_loglik
        @test fit.reml_loglik != fit.ml_loglik
    end

    @testset "ML stays the default and is unchanged" begin
        data = _cr_poisson_draw(443)
        fit_default = D.drm(form, D.Poisson(); data = data)
        fit_ml = D.drm(form, D.Poisson(); data = data, method = :ML)
        @test D.estimation_method(fit_default) === :ML
        @test D.estimation_method(fit_ml) === :ML
        # `method = :ML` is a no-op, so it must reproduce the default fit exactly.
        @test D.coef(fit_default) == D.coef(fit_ml)
        @test fit_default.loglik == fit_ml.loglik
        @test fit_default.loglik == fit_default.ml_loglik
    end

    @testset "the penalty moves σ̂_b upward (probe Cell A direction)" begin
        # ½·logdet(I_ββ) falls as σ_b grows (more cluster variance ⇒ less information
        # about β), so the restricted optimum sits at a LARGER σ̂_b than ML. Cell A saw
        # this on every G it measured. Per-seed, not averaged: this is the mechanism,
        # not a bias estimate.
        for seed in (443, 4431, 4432)
            data = _cr_poisson_draw(seed)
            fit_ml = D.drm(form, D.Poisson(); data = data)
            fit_reml = D.drm(form, D.Poisson(); data = data, method = :REML)
            @test _sigma_b(fit_reml) > _sigma_b(fit_ml)
            # The mean block should barely move — the correction targets the variance
            # component, not β.
            @test maximum(abs.(D.coef(fit_reml, :mu) .- D.coef(fit_ml, :mu))) < 0.2
        end
    end

    @testset "REML vcov is the restricted observed information (#310)" begin
        data = _cr_poisson_draw(443)
        fit_ml = D.drm(form, D.Poisson(); data = data)
        fit_reml = D.drm(form, D.Poisson(); data = data, method = :REML)
        Vr = D.vcov(fit_reml)
        @test size(Vr) == size(D.vcov(fit_ml))
        @test all(isfinite, Vr)
        @test issymmetric(Symmetric(Vr))
        @test all(>(0), diag(Vr))
        # It must NOT be the ML observed information: the restricted objective carries
        # the penalty curvature too.
        @test maximum(abs.(Vr .- D.vcov(fit_ml))) > 1e-8
    end

    @testset "scalar-per-cluster only — other Poisson routes still reject" begin
        # Poisson phylo Laplace: this used to assert a family-wide reject here, but
        # #450 admitted opt-in :REML on the phylo/relmat Laplace route. That
        # assertion has moved to test/test_cox_reid_poisson_phylo.jl, which owns the
        # shipped cell (same pattern as the #441 -> #443 handoff documented in
        # test/test_cox_reid_characterization.jl).

        # Correlated random slope `(1 + x | g)` is a 2-D RE, not scalar-per-cluster.
        data = _cr_poisson_draw(4434)
        err2 = try
            D.drm(D.bf(D.@formula(y ~ x + (1 + x | g))), D.Poisson();
                  data = data, method = :REML)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("(1 | g)", sprint(showerror, err2))

        # Fixed-effects-only Poisson has no variance component to restrict.
        err3 = try
            D.drm(D.bf(D.@formula(y ~ x)), D.Poisson(); data = data, method = :REML)
            nothing
        catch e
            e
        end
        @test err3 isa ArgumentError
    end

    @testset "other families are untouched by the hole" begin
        data = _cr_poisson_draw(4435)
        nt = (; y = Float64.(data.y .> 0), x = data.x, g = data.g)
        err = try
            D.drm(form, D.Binomial(); data = nt, method = :REML)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ML-only", sprint(showerror, err))
    end

    @testset "the marginal/method split is preserved" begin
        data = _cr_poisson_draw(4436)
        for bad in (:VA, :LA)
            err = try
                D.drm(form, D.Poisson(); data = data, method = bad)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("marginal = :$bad", sprint(showerror, err))
        end
        err = try
            D.drm(form, D.Poisson(); data = data, method = :NOPE)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("unknown", sprint(showerror, err))
    end

    @testset "Gaussian REML (#440) is not disturbed" begin
        rng = MersenneTwister(4437)
        G, n_each = 12, 5
        n = G * n_each
        g = repeat(1:G, inner = n_each)
        x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ 0.8 .* randn(rng, G)[g] .+ 0.7 .* randn(rng, n)
        gdata = (; y, x, g)
        @test D.estimation_method(D.drm(form, D.Gaussian(); data = gdata)) === :ML
        @test D.estimation_method(D.drm(form, D.Gaussian(); data = gdata,
                                       method = :REML)) === :REML
    end
end
