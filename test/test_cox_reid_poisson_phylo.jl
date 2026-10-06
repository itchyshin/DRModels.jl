# test_cox_reid_poisson_phylo.jl — opt-in Cox–Reid REML on Poisson phylo/relmat
# Laplace (`_fit_poisson_general_laplace`, #450).
#
# NOT registered in test/runtests.jl (AGHQ #448 / PR #449 dirties that file),
# so run it directly:
#
#   julia --project=. -e 'include("test/test_cox_reid_poisson_phylo.jl")'
#
# Scope: the callers of `_fit_poisson_general_laplace` — `phylo(1 | species)` and
# `relmat`/`animal`/precomputed spatial. Public `(1 | g)` is GHQ-32 via
# `_fit_poisson_ranef` and is already wired (#443); this file does not re-punch
# it. ML remains the default. Probe Cell D is NOT a recovery headline (ntip=16 /
# 12 seeds: ML +8.18%, CR +17.41%) — tests assert DIRECTION and mechanism only.
#
# Worked example:
#
#   using DRModels, StableRNGs, Distributions
#   rng = StableRNG(450)
#   ntip, per = 12, 4
#   tree = random_balanced_tree(ntip; branch_length = 0.25)
#   species = repeat(1:ntip, inner = per)
#   x = randn(rng, length(species))
#   C = sigma_phy_dense(tree; σ²_phy = 0.45^2)
#   u = cholesky(Symmetric(C)).L * randn(rng, ntip)
#   y = Float64.([rand(rng, Poisson(exp(0.25 + 0.2 * x[i] + u[species[i]])))
#                 for i in eachindex(x)])
#   form = bf(@formula(y ~ x + phylo(1 | species)))
#   fit_ml   = drm(form, Poisson(); data = (; y, x, species), tree = tree, se = false)
#   fit_reml = drm(form, Poisson(); data = (; y, x, species), tree = tree, se = false,
#                  method = :REML)
#   estimation_method(fit_reml) === :REML

using DRModels
using Test, Random, LinearAlgebra
using StableRNGs
import Distributions

const D = DRModels

function _cr_phylo_draw(seed::Int; ntip::Int = 12, per::Int = 4, σphy::Float64 = 0.45)
    rng = StableRNG(seed)
    tree = D.random_balanced_tree(ntip; branch_length = 0.25)
    species = repeat(1:ntip, inner = per)
    n = length(species)
    x = randn(rng, n)
    C = D.sigma_phy_dense(tree; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(rng, ntip)
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.25 + 0.2 * x[i] + u[species[i]])))
                  for i in 1:n])
    return (; data = (; y, x, species), tree)
end

function _cr_relmat_draw(seed::Int; G::Int = 16, m::Int = 4, σb::Float64 = 0.50)
    rng = StableRNG(seed)
    pos = rand(rng, G, 2) .* 6.0
    Dist = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
    K = Symmetric(exp.(-Dist ./ 0.8) + 1e-8 * I)
    d = sqrt.(diag(K))
    K = Symmetric(K ./ (d * d'))
    id = repeat(1:G, inner = m)
    n = length(id)
    x = randn(rng, n)
    u = σb .* (cholesky(K).L * randn(rng, G))
    y = Float64.([rand(rng, Distributions.Poisson(exp(0.20 + 0.3 * x[i] + u[id[i]])))
                  for i in 1:n])
    return (; data = (; y, x, id), K = Matrix(K))
end

_sigma(fit) = exp(D.coef(fit)[end])

@testset "Cox–Reid REML on Poisson phylo/relmat Laplace (#450)" begin
    form_phy = D.bf(D.@formula(y ~ x + phylo(1 | species)))
    form_rel = D.bf(D.@formula(y ~ x + relmat(1 | id)))

    @testset "opt-in :REML is admitted and tagged on phylo Laplace" begin
        draw = _cr_phylo_draw(450)
        fit = D.drm(form_phy, D.Poisson(); data = draw.data, tree = draw.tree,
                    se = false, method = :REML)
        @test D.estimation_method(fit) === :REML
        @test D.is_converged(fit)
        @test isfinite(fit.reml_loglik)
        @test isfinite(fit.ml_loglik)
        @test fit.loglik == fit.reml_loglik
        @test fit.reml_loglik != fit.ml_loglik
    end

    @testset "ML stays the default on phylo Laplace" begin
        draw = _cr_phylo_draw(450)
        fit_default = D.drm(form_phy, D.Poisson(); data = draw.data, tree = draw.tree, se = false)
        fit_ml = D.drm(form_phy, D.Poisson(); data = draw.data, tree = draw.tree,
                       se = false, method = :ML)
        @test D.estimation_method(fit_default) === :ML
        @test D.estimation_method(fit_ml) === :ML
        @test D.coef(fit_default) == D.coef(fit_ml)
        @test fit_default.loglik == fit_ml.loglik
    end

    @testset "the penalty moves σ̂ upward (direction, not recovery)" begin
        # ½·logdet(I_ββ) falls as the variance component grows, so the restricted
        # optimum sits at a LARGER σ̂ than ML. Per-seed, not averaged. Cell D is
        # not a recovery target and is not asserted here.
        #
        # GUARDED for DRModels.jl#498. The direction claim above only has content when
        # the optimum is INTERIOR. On Julia 1.12 (but not 1.10) this route drives
        # the variance component to the boundary on 2 of these 3 seeds — measured
        # sigma_hat 3.09e-04 and 2.40e-04 against a true sigma_phy of 0.45, i.e.
        # three orders of magnitude low. There ML and REML agree to ten significant
        # figures and `sigma_REML > sigma_ML` decides on noise (diff = -2.4e-11),
        # so the assertion flips with the Julia version rather than with anything
        # about Cox-Reid.
        #
        # That collapse is a REAL, PRE-EXISTING defect (verified against
        # origin/main's src on Julia 1.12: identical to ten decimals) and is
        # tracked in #498. It is NOT what this testset is about, so collapsed
        # seeds are reported and skipped rather than silently absorbed — and at
        # least one seed must be interior, so this can never degrade into a
        # testset that asserts nothing.
        interior = 0
        for seed in (450, 4501, 4502)
            draw = _cr_phylo_draw(seed)
            fit_ml = D.drm(form_phy, D.Poisson(); data = draw.data, tree = draw.tree, se = false)
            fit_reml = D.drm(form_phy, D.Poisson(); data = draw.data, tree = draw.tree,
                             se = false, method = :REML)
            if _sigma(fit_ml) < 0.01                      # boundary: truth is 0.45
                @info "seed skipped — variance component collapsed to the boundary (DRModels.jl#498)" seed sigma_ml=_sigma(fit_ml) sigma_reml=_sigma(fit_reml) julia=VERSION
                continue
            end
            interior += 1
            @test _sigma(fit_reml) > _sigma(fit_ml)
        end
        @test interior >= 1                                # never vacuous
    end

    @testset "relmat/animal share the same spine" begin
        draw = _cr_relmat_draw(4503)
        fit_rel = D.drm(form_rel, D.Poisson(); data = draw.data, K = draw.K,
                        se = false, method = :REML)
        @test D.estimation_method(fit_rel) === :REML
        @test fit_rel.reml_loglik != fit_rel.ml_loglik
        fit_an = D.drm(D.bf(D.@formula(y ~ x + animal(1 | id))), D.Poisson();
                       data = draw.data, A = draw.K, se = false, method = :REML)
        @test D.estimation_method(fit_an) === :REML
        fit_ml = D.drm(form_rel, D.Poisson(); data = draw.data, K = draw.K, se = false)
        @test _sigma(fit_rel) > _sigma(fit_ml)
    end

    @testset "uncertified routes still error" begin
        rng = StableRNG(4504)
        G, m = 8, 4
        n = G * m
        g = repeat(1:G, inner = m)
        x = randn(rng, n)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.2 * xi))) for xi in x])
        data = (; y, x, g)

        # Coordinate-spatial with jointly estimated ρ is a different fitter.
        coords = rand(rng, G, 2)
        site = g
        err_sp = try
            D.drm(D.bf(D.@formula(y ~ x + spatial(1 | site))), D.Poisson();
                  data = (; y, x, site), coords = coords, se = false, method = :REML)
            nothing
        catch e
            e
        end
        @test err_sp isa ArgumentError
        @test occursin("REML", sprint(showerror, err_sp))

        # Crossed intercepts, slopes, VA, FE-only stay rejected.
        h = repeat(1:4, outer = div(n, 4))
        err_x = try
            D.drm(D.bf(D.@formula(y ~ x + (1 | g) + (1 | h))), D.Poisson();
                  data = (; y, x, g, h), method = :REML)
            nothing
        catch e
            e
        end
        @test err_x isa ArgumentError

        err_sl = try
            D.drm(D.bf(D.@formula(y ~ x + (1 + x | g))), D.Poisson();
                  data = data, method = :REML)
            nothing
        catch e
            e
        end
        @test err_sl isa ArgumentError

        err_va = try
            D.drm(D.bf(D.@formula(y ~ x + (1 | g))), D.Poisson();
                  data = data, marginal = :VA, method = :REML)
            nothing
        catch e
            e
        end
        @test err_va isa ArgumentError

        err_fe = try
            D.drm(D.bf(D.@formula(y ~ x)), D.Poisson(); data = data, method = :REML)
            nothing
        catch e
            e
        end
        @test err_fe isa ArgumentError
    end

    @testset "Binomial still rejects :REML" begin
        draw = _cr_phylo_draw(4505)
        nt = (; y = Float64.(draw.data.y .> 0), x = draw.data.x, species = draw.data.species)
        err = try
            D.drm(form_phy, D.Binomial(); data = nt, tree = draw.tree, method = :REML)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("ML-only", sprint(showerror, err))
    end

    @testset "GHQ (1|g) cell is untouched" begin
        rng = StableRNG(4506)
        G, m = 10, 6
        g = repeat(1:G, inner = m)
        x = randn(rng, G * m)
        b = 0.6 .* randn(rng, G)
        y = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.2 * x[i] + b[g[i]])))
                      for i in eachindex(x)])
        form = D.bf(D.@formula(y ~ x + (1 | g)))
        @test D.estimation_method(D.drm(form, D.Poisson(); data = (; y, x, g))) === :ML
        @test D.estimation_method(D.drm(form, D.Poisson(); data = (; y, x, g),
                                       method = :REML)) === :REML
    end
end
