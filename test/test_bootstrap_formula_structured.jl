# test_bootstrap_formula_structured.jl — issue #480, sibling of #479.
#
# #479 fixed the FIT-based `bootstrap_result(fit::DrmFit; ...)`, whose refit
# closure discarded `tree`/`K`/`A` so every replicate of a non-Gaussian
# STRUCTURED fit refit the wrong (unstructured) model. The keywords were
# already declared and forwarded by `bootstrap_summary`/`bootstrap_ci` -- only
# the bottom method threw them away.
#
# The FORMULA-based surface carried the same belief in two comments
# ("structured-matrix keywords (those are Gaussian-only)" / "matrix keywords;
# those are Gaussian-only") but never even DECLARED K/A/tree for non-Gaussian
# families, so a structured formula there raised a Julia `MethodError` --
# loud, not silently wrong, but the comments were false: #479 established the
# non-Gaussian bootstrap CAN thread a structured covariance.
#
# Fixed here by mirroring #479 exactly: `bootstrap_result(formula::DrmFormula,
# family; ...)` now declares K/A/tree, forwards a keyword to `drm(...)` only
# when the caller actually supplied it (most non-Gaussian `drm` methods do not
# declare K/A/tree at all, so forwarding `nothing` unconditionally would break
# every ordinary unstructured fit), and builds the same #459 marginal
# simulator the fit-based method uses so a variance-component CI is not
# degenerate. `bootstrap_ci`/`bootstrap_summary` thread the same keywords
# through. Both "Gaussian-only" comments are deleted.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "#480 formula-based non-Gaussian bootstrap threads K/A/tree" begin
    @testset "Poisson phylo -- multi-tree-height round-trip" begin
        # Same trap as #479: `re_sd` for a phylo term is defined against the RAW
        # covariance `sigma_phy_dense(phy)` (diagonal = tree height), NOT the
        # normalised correlation matrix. A mistake that substitutes the
        # correlation matrix is INVISIBLE on a height-1 tree, so round-trip
        # across several heights.
        for (seed, branch_length) in ((20260930, 0.2), (20260931, 1.0), (20260932, 3.0))
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
            form = bf(@formula(y ~ x + phylo(1 | species)))

            # Before the fix: this threw `MethodError: no keyword argument tree` --
            # the formula-based non-Gaussian method declared no K/A/tree at all.
            res = bootstrap_result(form, Poisson(); data = dat, tree = phy, B = 12,
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
            # Not the #459 degeneracy signature (a near-zero-width CI): the
            # marginal simulator must be wired here too, not only on the
            # fit-based route.
            @test (sdr.upper - sdr.lower) > 0.05
        end
    end

    @testset "Poisson relmat -- K threaded through bootstrap_ci/summary/result" begin
        rng = MersenneTwister(20260940)
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
        form = bf(@formula(y ~ x + relmat(1 | id)))

        # Before the fix: raised a bare MethodError, not a domain message.
        rows_ci = bootstrap_ci(form, Poisson(); data = dat, K = Matrix(C), B = 12,
                                rng = MersenneTwister(20260941), failures = :skip,
                                check_converged = false)
        rows_sum = bootstrap_summary(form, Poisson(); data = dat, K = Matrix(C), B = 12,
                                      rng = MersenneTwister(20260941), failures = :skip,
                                      check_converged = false)
        res = bootstrap_result(form, Poisson(); data = dat, K = Matrix(C), B = 12,
                                rng = MersenneTwister(20260941), failures = :skip,
                                check_converged = false)
        @test res.attempted == 12
        @test res.used >= 9
        xr = first(r for r in res.summary if r.param === :mu && r.coef == "x")
        @test xr.lower < xr.estimate < xr.upper
        @test [(r.param, r.coef, r.lower, r.upper) for r in rows_ci] ==
              [(r.param, r.coef, r.lower, r.upper) for r in res.summary]
        @test [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in rows_sum] ==
              [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in res.summary]

        # Cross-check against the #479 fit-based route: same K, same seeds ->
        # same refit sequence, since both now thread K identically into `drm(...)`
        # and build the same marginal simulator.
        fit = drm(form, Poisson(); data = dat, K = Matrix(C), se = false)
        res_fit = bootstrap_result(fit; data = dat, K = Matrix(C), B = 12,
                                    rng = MersenneTwister(20260941), failures = :skip,
                                    check_converged = false)
        @test res_fit.seeds == res.seeds
    end

    @testset "omitting tree fails with drm's own honest message, not a bare MethodError" begin
        Random.seed!(20260950)
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
        form = bf(@formula(y ~ x + phylo(1 | species)))

        err = nothing
        try
            bootstrap_result(form, Poisson(); data = dat, B = 3, rng = MersenneTwister(1))
        catch e
            err = e
        end
        @test err !== nothing
        @test !(err isa MethodError)
        msg = sprint(showerror, err)
        @test occursin("phylo(1 | species) needs", msg)
        @test !occursin("only valid for Gaussian fits", msg)
    end

    @testset "unstructured families without K/A/tree in their drm() signature still work" begin
        # Regression guard for the "forward only if supplied" fix: LogNormal,
        # Tweedie, Student, SkewNormal, ZeroOneBeta, CumulativeLogit and
        # TruncatedNegBinomial2 do not declare K/A/tree at all in `drm(...)`.
        # Forwarding them unconditionally (even as `nothing`) would throw a
        # MethodError on every ordinary fit through these families.
        Random.seed!(20260960)
        n = 200
        x = randn(n)
        yln = Float64[exp(0.1 + 0.3 * x[i] + 0.25 * randn()) for i in 1:n]
        form = bf(@formula(yln ~ x), @formula(sigma ~ 1))
        dat = (; yln, x)
        res = bootstrap_result(form, LogNormal(); data = dat, B = 8,
                                rng = MersenneTwister(1))
        @test res.attempted == res.used == 8
        xr = first(r for r in res.summary if r.param === :mu && r.coef == "x")
        @test xr.lower < xr.estimate < xr.upper
    end
end
