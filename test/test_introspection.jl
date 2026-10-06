# test_introspection.jl — A4d-2: `profile_targets` and `structured_effects`,
# the Julia twins of drmTMB's post-fit inventories.
#
# WHY THIS FILE EXISTS. Both functions are *claims about what is available*, and
# a claim is exactly the thing that rots silently. `profile_targets` is only
# worth having if its `profile_ready` column tracks what `profile_result` will
# ACTUALLY do — a readiness flag that always says "ready" is worse than no flag,
# because it converts a clear downstream error into a broken promise. So the
# tests below pin readiness against the real dispatch on routes that differ:
# a fixed-effect location-scale fit (profilable) and a σ-phylo fit (not, unless
# it was fitted with `profile_ci = true`).

using DRModels
using Test
using Random
using LinearAlgebra

# σ-phylo on BOTH axes: the route with no re-optimisable objective, which is what
# makes the `profile_ready` column non-trivial.
function _introspect_phylo_fixture(seed::Int; G::Int = 12, m::Int = 6)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(G; branch_length = 0.4)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d'); L = cholesky(Symmetric(K)).L
    species = repeat(1:G, inner = m); n = G * m
    u1 = 0.8 .* (L * randn(rng, G)); u2 = 0.5 .* (L * randn(rng, G))
    y = [1.0 + u1[species[i]] + exp(log(0.5) + u2[species[i]]) * randn(rng) for i in 1:n]
    return (data = (; y, species), tree = phy,
            form = bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ phylo(1 | species))))
end

@testset "profile_targets / structured_effects (A4d-2)" begin

    @testset "profile_targets on a profilable fixed-effect fit" begin
        rng = MersenneTwister(9); n = 200; x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ 0.6 .* randn(rng, n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

        tg = profile_targets(fit)
        @test length(tg) == length(coef(fit))
        @test all(r -> r.profile_ready, tg)
        @test [r.param for r in tg] == [:mu, :mu, :sigma, :sigma]
        @test [r.index for r in tg] == 1:4
        # the estimate column is the estimation-scale theta, not a back-transform
        @test [r.estimate for r in tg] == fit.theta
        # scale column: mean coefficients identity, scale coefficients log
        @test [r.scale for r in tg] == [:identity, :identity, :log, :log]
        @test length(profile_targets(fit; ready_only = true)) == length(tg)

        # The claim must match the dispatch: everything reported ready must
        # actually come back from profile_result.
        pr = profile_result(fit)
        @test length(pr.ci) == count(r -> r.profile_ready, tg)
    end

    @testset "profile_targets is honest about a route that cannot profile" begin
        fx = _introspect_phylo_fixture(3)
        fit = drm(fx.form, Gaussian(); data = fx.data, tree = fx.tree)
        tg = profile_targets(fit)
        @test !isempty(tg)
        @test all(r -> !r.profile_ready, tg)
        # the note has to be actionable, not merely true
        @test all(r -> occursin("profile_ci", r.profile_note), tg)
        @test isempty(profile_targets(fit; ready_only = true))
        # and the refusal it predicts is the refusal that actually happens
        @test_throws ArgumentError profile_result(fit)
    end

    @testset "profile_ci = true flips the σ-phylo rows to ready" begin
        fx = _introspect_phylo_fixture(3)
        fit = drm(fx.form, Gaussian(); data = fx.data, tree = fx.tree, profile_ci = true)
        tg = profile_targets(fit)
        ready = filter(r -> r.profile_ready, tg)
        @test !isempty(ready)
        # only the SD blocks carry precomputed rows; the mean/scale ones do not,
        # and must say so rather than claiming a profile they cannot produce
        @test all(r -> r.param in (:resd_mu, :resd_sigma), ready)
        @test all(r -> occursin("precomputed", r.profile_note), ready)
    end

    @testset "structured_effects reports every marker, per dpar" begin
        fx = _introspect_phylo_fixture(3)
        fit = drm(fx.form, Gaussian(); data = fx.data, tree = fx.tree)
        se = structured_effects(fit)
        @test length(se) == 2
        @test all(r -> r.kind === :phylo, se)
        @test all(r -> r.grouping === :species, se)
        @test Set(r.dpar for r in se) == Set([:mu, :sigma])
    end

    @testset "structured_effects is empty when there is no structure" begin
        rng = MersenneTwister(4); n = 120; x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ 0.6 .* randn(rng, n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test isempty(structured_effects(fit))
    end
end
