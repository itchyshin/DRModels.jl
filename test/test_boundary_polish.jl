# test_boundary_polish.jl — issue #422: a collapsed variance component used to
# leave the other coordinates short of their conditional optimum.
#
# THE MECHANISM. When the structured SD collapses, the outer objective goes FLAT
# in `log_sd` — on the recorded fixture, nll at log_sd = -12.3 and at -81.6 agree
# to twelve digits. LBFGS chased that flat direction to -81 and terminated there
# on the gradient norm, leaving the remaining coordinates ~1e-3 short and costing
# ~7e-05 of nll. A parity harness saw that as a near-tolerance failure against
# native drmTMB; it was not a likelihood difference, it was under-convergence.
#
# THE TEST THAT MATTERS is self-contained: DRModels.jl's own objective must not be
# lower anywhere near the reported optimum. That cannot pass by agreeing with a
# second buggy implementation, and it fails on the pre-fix engine.

using DRModels
using Test
using Random
using LinearAlgebra
using Statistics

function _collapse_fixture(seed::Int; n_tip = 12, n_each = 6)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(n_tip; branch_length = 1.0 / 4)  # unit height
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C)); K = C ./ (d * transpose(d))
    L = cholesky(Symmetric(K)).L
    species = repeat(1:n_tip, inner = n_each)
    n = n_tip * n_each
    x = randn(rng, n)
    # sd_phy = 0 on purpose: the phylo SD SHOULD collapse, which is exactly the
    # regime where the old optimiser wandered onto the flat shelf.
    eta = 0.3 .+ 0.4 .* x
    y = [exp(eta[i]) * (0.8 + 0.4 * rand(rng)) for i in 1:n]
    return (data = (; y, x, species), tree = phy)
end

@testset "boundary polish for a collapsed variance component (#422)" begin

    @testset "the reported optimum is not beatable along its own objective" begin
        for seed in (411, 412, 413)
            fx = _collapse_fixture(seed)
            fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      DRModels.Gamma(); data = fx.data, tree = fx.tree)
            @test fit.nll !== nothing
            base = fit.nll(fit.theta)
            # Perturb the NON-variance coordinates: if the fit under-converged
            # them (the #422 symptom) some nearby point is strictly better.
            rng = MersenneTwister(seed)
            for _ in 1:12
                θp = copy(fit.theta)
                for k in 1:(length(θp) - 1)
                    θp[k] += 1e-3 * randn(rng)
                end
                @test fit.nll(θp) >= base - 1e-9
            end
        end
    end

    @testset "a collapsed SD is never chased into denormal territory" begin
        # Chasing log_sd down the flat shelf produced absurd reported SDs
        # (3.7e-36 on the issue's gamma fixture). The point is not that the floor
        # always engages -- on many datasets the optimiser stops well above it,
        # and then there is nothing to fix -- but that the reported SD is never
        # an artefact of an unbounded walk.
        for seed in (411, 412, 413)
            fx = _collapse_fixture(seed)
            fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      DRModels.Gamma(); data = fx.data, tree = fx.tree)
            sd = re_sd(fit)[:species]
            @test sd > 0
            @test sd < 1e-2         # collapsed, as this DGP (no phylo signal) intends
            @test sd >= 1e-9        # and not walked into denormal territory
        end
    end

    @testset "an INTERIOR optimum is untouched" begin
        # The polish must only engage on the boundary. With a real phylo signal
        # the fitted SD is interior and the fit must be unchanged in kind.
        rng = MersenneTwister(99)
        n_tip = 16; n_each = 8
        phy = random_balanced_tree(n_tip; branch_length = 0.25)
        C = sigma_phy_dense(phy; σ²_phy = 1.0)
        d = sqrt.(diag(C)); K = C ./ (d * transpose(d))
        L = cholesky(Symmetric(K)).L
        species = repeat(1:n_tip, inner = n_each); n = n_tip * n_each
        x = randn(rng, n)
        u = 0.8 .* (L * randn(rng, n_tip))
        eta = 0.3 .+ 0.4 .* x .+ u[species]
        y = [exp(eta[i]) * (0.8 + 0.4 * rand(rng)) for i in 1:n]
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                  DRModels.Gamma(); data = (; y, x, species), tree = phy)
        sd = re_sd(fit)[:species]
        @test sd > 1e-3             # genuinely interior: the floor never engaged
        @test is_converged(fit)
    end
end

@testset "Binomial refuses unsupported structured markers with a MESSAGE (A9)" begin
    # Found during the A9 general-covariance audit: `relmat(1|g)` with `K = K`
    # produced a bare `MethodError: no method matching drm(::DrmFormula,
    # ::Binomial; K=…)` — a DISPATCH failure, so the explanation this method
    # already carried was unreachable. A refusal should say what is supported.
    rng = MersenneTwister(5); G = 14; m = 6
    K = [i == j ? 1.0 : 0.3 for i in 1:G, j in 1:G]
    n = G * m; g = repeat(1:G, inner = m); x = randn(rng, n)
    y = Float64.(rand(rng, n) .< 0.5)

    err = try
        drm(bf(@formula(y ~ x + relmat(1 | g))), DRModels.Binomial(); data = (; y, x, g), K = K)
        nothing
    catch e
        e
    end
    @test err !== nothing
    @test !(err isa MethodError)                     # the actual regression
    @test occursin("phylo", sprint(showerror, err))  # and it names what IS supported

    # the supported provider still fits
    phy = random_balanced_tree(G; branch_length = 0.25)
    fit = drm(bf(@formula(y ~ x + phylo(1 | g))), DRModels.Binomial(); data = (; y, x, g), tree = phy)
    @test is_converged(fit)
end
