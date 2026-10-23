# test_bootstrap_nongaussian_structured.jl — issue #479.
#
# The generic (non-Gaussian) method of `bootstrap_result` built its refit
# closure as `drm(formula, fit.family; data=datab)` -- formula, family, data,
# and NOTHING ELSE. Every bootstrap replicate of a non-Gaussian STRUCTURED fit
# (phylo/relmat/animal) therefore refit without the covariance structure the
# original fit had and failed on every replicate, hidden behind a guard whose
# message ("K/A/tree are only valid for Gaussian fits") read as "pass them
# differently" when the truth was "this path cannot use them at all".
#
# Fixed by threading K/A/tree into both the refit closure and the marginal
# simulator, exactly as the Gaussian method already does -- the caller
# re-supplies the same K/A/tree used to produce the fit, and a mismatch is
# now caught by `drm(...)`'s own per-family checks instead of the generic
# guard's misleading message.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "#479 non-Gaussian structured bootstrap threads K/A/tree" begin
    @testset "Poisson phylo -- multi-tree-height round-trip" begin
        # The trap this project has hit before: `re_sd` for a phylo term is
        # defined against the RAW covariance `sigma_phy_dense(phy)` (diagonal =
        # tree height), NOT the normalised correlation matrix. A mistake that
        # substitutes the correlation matrix is INVISIBLE on a height-1 tree.
        # Round-trip across several heights so such a mistake could not hide.
        for (seed, branch_length) in ((20260901, 0.2), (20260902, 1.0), (20260903, 3.0))
            Random.seed!(seed)
            p = 16
            m = 4
            phy = random_balanced_tree(p; branch_length = branch_length)
            species = repeat(1:p, inner = m)
            n = length(species)
            x = randn(n)
            β = [0.15, 0.35]
            σphy = 0.45
            C = sigma_phy_dense(phy; σ²_phy = σphy^2)
            u = cholesky(Symmetric(C)).L * randn(p)
            λ = exp.(β[1] .+ β[2] .* x .+ u[species])
            y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
            dat = (; y, x, species)

            fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                      data = dat, tree = phy, se = false)
            @test fit.converged

            # Before the fix: every replicate failed regardless of `branch_length`
            # (the closure dropped `tree` no matter what the caller passed).
            res = bootstrap_result(fit; data = dat, tree = phy, B = 12,
                                    rng = MersenneTwister(seed + 1),
                                    failures = :skip, check_converged = false)
            @test res.attempted == 12
            @test res.used >= 9   # allow a few genuinely hard resampled refits

            xr = first(r for r in res.summary if r.param === :mu && r.coef == "x")
            @test xr.lower < xr.estimate < xr.upper

            sd_rows = [r for r in res.summary if r.param === :resd]
            @test !isempty(sd_rows)
            sdr = first(sd_rows)
            @test isfinite(sdr.lower) && isfinite(sdr.upper)
            @test sdr.lower < sdr.estimate < sdr.upper
            # Not the #459 degeneracy signature (a near-zero-width CI), at any height.
            @test (sdr.upper - sdr.lower) > 0.05
        end
    end

    @testset "Poisson relmat -- K threaded through refit and simulator" begin
        rng = MersenneTwister(20260910)
        G = 24
        m = 4
        pos = rand(rng, G, 2) .* 6.0
        D = [sqrt(sum(abs2, pos[k, :] .- pos[l, :])) for k in 1:G, l in 1:G]
        Craw = exp.(-D ./ 0.8) + 1e-8 * I
        d = sqrt.(diag(Craw))
        C = Symmetric(Craw ./ (d * d'))
        id = repeat(1:G, inner = m)
        n = length(id)
        x = randn(rng, n)
        β = [0.2, 0.4]
        σb = 0.5
        u = σb .* (cholesky(C).L * randn(rng, G))
        λ = exp.(β[1] .+ β[2] .* x .+ u[id])
        y = Float64.([rand(rng, Distributions.Poisson(λi)) for λi in λ])
        dat = (; y, x, id)

        fit = drm(bf(@formula(y ~ x + relmat(1 | id))), Poisson();
                  data = dat, K = Matrix(C), se = false)
        @test fit.converged

        # Before the fix: `bootstrap_result(fit; ..., K = Matrix(C))` raised
        # "bootstrap_result: K/A/tree are only valid for Gaussian fits" outright.
        res = bootstrap_result(fit; data = dat, K = Matrix(C), B = 12,
                                rng = MersenneTwister(20260911),
                                failures = :skip, check_converged = false)
        @test res.attempted == 12
        @test res.used >= 9

        xr = first(r for r in res.summary if r.param === :mu && r.coef == "x")
        @test xr.lower < xr.estimate < xr.upper
    end

    @testset "omitting tree no longer trips the misleading Gaussian-only guard" begin
        Random.seed!(20260920)
        p = 12
        m = 4
        phy = random_balanced_tree(p; branch_length = 0.3)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(n)
        u = 0.4 .* (cholesky(Symmetric(sigma_phy_dense(phy; σ²_phy = 1.0))).L * randn(p))
        λ = exp.(0.2 .+ 0.3 .* x .+ u[species])
        y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
        dat = (; y, x, species)
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                  data = dat, tree = phy, se = false)

        # `tree` genuinely required and not supplied: every replicate now fails
        # inside `drm(...)` with the family's own honest message, not the old
        # "K/A/tree are only valid for Gaussian fits" guard.
        err = nothing
        try
            bootstrap_result(fit; data = dat, B = 3, rng = MersenneTwister(1))
        catch e
            err = e
        end
        @test err isa ErrorException
        msg = sprint(showerror, err)
        @test occursin("phylo(1 | species) needs", msg)
        @test !occursin("only valid for Gaussian fits", msg)
    end
end
