# test_phylo_count_largep_gate.jl — a Julia-side standing gate for the
# `phylo_count_large_p` capability row.
#
# WHY THIS EXISTS
# ---------------
# The row's own claim_boundary named this file's absence as its limitation:
# "the harness is R-side (tools/parity_classc*.R) with NO wired Julia test, so
# this is not a standing gate". Everything backing the row lived in an R script
# that must be run by hand, which means a regression in the large-p phylogenetic
# count route could not be caught by `Pkg.test()`.
#
# This is NOT a parity test — it needs no R, no drmTMB, and no fixtures. Parity
# stays with the R harness. What this gate protects is that the route keeps
# FITTING and keeps RECOVERING at scale.
#
# THE LOAD-BEARING TEST IS THE TREE-HEIGHT ROUND-TRIP
# ---------------------------------------------------
# `re_sd` for a phylo term is defined against the RAW covariance
# `sigma_phy_dense(phy)`, whose diagonal is the tree height — NOT against a
# normalised correlation matrix. The two agree exactly on a height-1 tree, so
# the mistake is invisible on the trees people usually write down, and a fixture
# built at height 1 cannot detect it.
#
# So the simulation draws the phylo effect from the RAW covariance with a known
# raw-scale sigma, and refits across heights 0.8, 1.6 and 4.0 (a 5x range). If
# `re_sd` were being interpreted on the normalised scale, the recovered value
# would scale as sqrt(height) and spread by sqrt(4.0/0.8) ~ 2.24x across those
# three trees. Measured spread is ~1.08x. That gap is what the assertion below
# is sized against.
#
# FIXED: `converged` at scale — DRModels.jl#491
# -----------------------------------------------
# `fit.converged` used to be true at p=128 and false for every p >= 192, while
# the estimates got BETTER. `_laplace_outer_converged` compared a flat limit,
# 1e-4*(1 + norm(theta, Inf)) ~ 1.6e-4, against a gradient that grows linearly
# with n; the relative gradient is flat (1.21e-7, 1.33e-7, 1.39e-7 at
# n = 512, 1024, 2048). The `g_tol * n` term written to handle this was ~30x too
# small to ever win the max() and first bound near n = 16,000. The fix
# normalises the gradient by n before comparing it to a flat floor
# (max(g_tol, 1e-4)), matching the convention already used by the q4 routes.
#
# This used to be recorded with `@test_broken` rather than by asserting
# `false`, so that a fix would show up as "Unexpectedly Passed" instead of
# looking like a regression. Now that #491 is fixed, it is a plain `@test`.

using DRModels
using Test
using Random
using LinearAlgebra
import Distributions

# Simulate a phylogenetic Poisson count dataset and fit it.
# `sig` is the phylo SD on the RAW covariance scale, matching `re_sd`'s definition.
function _largep_sim_fit(p::Int, branch_length::Real, sig::Real;
                         seed::Int = 4242, m::Int = 4, b0 = 0.2, b1 = 0.3, g_tol = 1e-8)
    rng = MersenneTwister(seed)
    phy = DRModels.random_balanced_tree(p; branch_length = branch_length)
    Sraw = DRModels.sigma_phy_dense(phy)              # diagonal = tree height
    L = cholesky(Symmetric(Sraw) + 1e-10I).L
    a = sig .* (L * randn(rng, p))
    species = repeat(1:p, inner = m)
    n = p * m
    x = randn(rng, n)
    eta = b0 .+ b1 .* x .+ a[species]
    y = Float64.([rand(rng, Distributions.Poisson(exp(clamp(eta[i], -20, 20)))) for i in 1:n])
    fit = drm(bf(@formula(y ~ x + phylo(1 | species))), DRModels.Poisson();
              data = (; y, x, species), tree = phy, se = false, g_tol = g_tol)
    return (; fit, height = Sraw[1, 1], n)
end

@testset "phylo_count_large_p — Julia-side standing gate" begin
    B1_TRUE, SIG_TRUE = 0.3, 0.6

    @testset "fits and recovers as p grows" begin
        # Tolerances sized from measurement (2026-08-25), not chosen a priori.
        # Observed |b1 - 0.3|: 0.082 at p=128, then <= 0.023 for p >= 256.
        # Observed |re_sd - 0.6| <= 0.068 throughout.
        for (p, b1_atol) in ((128, 0.15), (256, 0.08), (512, 0.08))
            r = _largep_sim_fit(p, 0.2, SIG_TRUE)
            @test r.n == 4p
            @test isfinite(r.fit.loglik)
            @test r.fit.loglik < 0
            @test isapprox(r.fit.theta[2], B1_TRUE; atol = b1_atol)
            @test isapprox(exp(r.fit.theta[3]), SIG_TRUE; atol = 0.15)
        end
    end

    @testset "re_sd is on the RAW covariance scale, not the normalised one" begin
        # Same p, same raw-scale sigma, three tree heights spanning 5x.
        heights, sds = Float64[], Float64[]
        for bl in (0.1, 0.2, 0.5)
            r = _largep_sim_fit(256, bl, SIG_TRUE)
            push!(heights, r.height)
            push!(sds, exp(r.fit.theta[3]))
            @test isapprox(exp(r.fit.theta[3]), SIG_TRUE; atol = 0.15)
        end
        # The trees really do differ in height, or the test proves nothing.
        @test maximum(heights) / minimum(heights) >= 4.0
        # Raw scale => re_sd is height-invariant. Normalised scale => it would
        # spread by sqrt(4.0/0.8) ~ 2.24. Measured spread ~1.08; 1.5 separates
        # the two cleanly with margin on both sides.
        @test maximum(sds) / minimum(sds) < 1.5
    end

    @testset "converged flag at scale (DRModels.jl#491)" begin
        small = _largep_sim_fit(128, 0.2, SIG_TRUE)
        large = _largep_sim_fit(512, 0.2, SIG_TRUE)
        @test small.fit.converged
        # The p=512 fit is BETTER than p=128 by every relative measure, and is
        # now also flagged true (the acceptance limit is normalised by n).
        @test large.fit.converged
        # Whatever the flag says, the fit itself must stay usable.
        @test isfinite(large.fit.loglik)
        @test isapprox(large.fit.theta[2], B1_TRUE; atol = 0.08)
    end

    @testset "converged is not for sale (DRModels.jl#491, D-179 #1)" begin
        # The sharpest finding in #491: with the old `Optim.converged(res) &&
        # return true` short-circuit, ASKING for a sloppier fit made the flag
        # EASIER to earn -- g_tol = 10.0 reported converged at a relative
        # gradient of 1.18e-03, four orders of magnitude worse than the default
        # fit that reported NOT converged. The flag now answers a fixed,
        # scale-invariant question (mean per-observation gradient <= 1e-6), so a
        # deliberately loose optimiser run must NOT be reported as converged.
        sloppy = _largep_sim_fit(128, 0.2, SIG_TRUE; g_tol = 10.0)
        @test !sloppy.fit.converged
        # ... while the honestly-converged default fit still is (asserted above,
        # re-asserted here so this testset stands alone).
        @test _largep_sim_fit(128, 0.2, SIG_TRUE).fit.converged
    end

    @testset "FD vcov step grows with n (large-p SE gap, D-179 roadmap A1)" begin
        # The p >= 1000 SE-parity gap (~1.5e-3 at p=1000, ~3.6e-3 at p=3000 vs
        # native TMB on identical data) was the FD NOISE FLOOR: the outer vcov
        # Hessian differentiated the summed, inner-solved objective at a fixed
        # h = 1e-4, and the objective's evaluation noise grows with n while the
        # /h^2 amplification does not. The step is now n-aware
        # (h ~ 2.5e-7 * n, clamped to [1e-4, 1e-2]); with it the measured gaps
        # fall to 2.0e-5 (p=1000) and 8.1e-6 (p=3000). Cross-engine numbers
        # live in the R-side harness (tools/parity_classc_largep.R); this lock
        # just keeps the step's shape from silently reverting.
        @test DRModels._fd_hessian_step(48) == 1e-4        # small fixtures: unchanged
        @test DRModels._fd_hessian_step(400) == 1e-4       # clamp edge
        @test DRModels._fd_hessian_step(4000) ≈ 1e-3       # p=1000 cell
        @test DRModels._fd_hessian_step(12000) ≈ 3e-3      # p=3000 cell
        @test DRModels._fd_hessian_step(10^6) == 1e-2      # truncation guard
    end
end
