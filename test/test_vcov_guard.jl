# Guarded Hessian→covariance step (`src/vcov_guard.jl`).
#
# Regression for the 2026-08-14 CI failure: `anchor (c)` in test_variational.jl
# threw SingularException(3) on Julia 1.10.11 and passed on 1.10.0 / 1.11 with
# the same seeded data. The NB2 fit sits at r → ∞, so the σ coordinate is
# unidentified and the Hessian is singular by construction. The outcome must now
# be the same on every platform: a warning plus a pseudo-inverse, never a crash
# and never a silent finite-but-meaningless inverse.

using DRModels
using Test
using Random
using LinearAlgebra
using Logging
import Distributions

const DVG = DRModels

@testset "vcov guard (boundary-singular Hessian)" begin

    @testset "well-conditioned Hessian inverts exactly" begin
        H = [4.0 1.0; 1.0 3.0]
        V = DVG._vcov_from_hessian(H)
        @test V ≈ inv(H)
        @test issymmetric(V)
    end

    @testset "singular Hessian warns and pseudo-inverts instead of throwing" begin
        # Third coordinate flat — the shape seen at an NB2 size boundary.
        H = [39.8 -14.5 0.0 3.8
             -14.5 85.6 0.0 0.4
             0.0 0.0 0.0 0.0
             3.8 0.4 0.0 12.1]
        V = @test_logs (:warn,) DVG._vcov_from_hessian(H)
        @test all(isfinite, V)
        @test issymmetric(V)
        # The flat coordinate must not acquire a spurious finite variance.
        @test V[3, 3] == 0
    end

    @testset "near-singular Hessian is treated as singular on every platform" begin
        # cond ~ 1e16: the knife edge where LAPACK may or may not throw.
        H = [39.8 -14.5 0.0 3.8
             -14.5 85.6 0.0 0.4
             0.0 0.0 9.85e-16 0.0
             3.8 0.4 0.0 12.1]
        V = @test_logs (:warn,) DVG._vcov_from_hessian(H)
        @test all(isfinite, V)
        @test V[3, 3] < 1.0   # not the 1e15 a raw inverse would produce
    end

    @testset "regression: NB2 at r → ∞ on Poisson data does not throw" begin
        # The exact fixture from test_variational.jl:105.
        rng = MersenneTwister(1362)
        ng, per = 8, 6
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        Xσ = ones(n, 1)
        β = [0.4, -0.3]; logσb = log(0.5)
        bg = exp(logσb) .* randn(rng, ng)
        y = Float64.([rand(rng, Distributions.Poisson(exp((Xμ * β)[i] + bg[gidx[i]])))
                      for i in 1:n])

        fit_nb = DRModels._fit_nb2_ranef_va(DRModels.NegBinomial2(), y, Xμ, Xσ, gidx, ng,
                                       ["(Intercept)", "x"], ["(Intercept)"], :g, 1e-8)
        @test all(isfinite, fit_nb.theta)
        @test all(isfinite, fit_nb.vcov)
    end
end

@testset "sparse-Laplace route consults the shared guard (#488)" begin
    # `src/sparse_laplace_glmm.jl` used to compute `vcov` via a bare
    # `try inv(Symmetric(H)) catch identity end`, bypassing `_vcov_from_hessian`
    # entirely: no `rcond`, no guard, so a singular Hessian on this route could
    # never produce the warning above. Both cells now route through the same
    # guard on the same threshold as every other family.

    @testset "deliberately ill-conditioned fit fires the existing guard" begin
        # Pure-Poisson data (no overdispersion, no random-intercept signal at
        # all) fit through NB2 phylo(1|species) with a constant sigma formula
        # — this is the general-mean-nuisance sparse-Laplace cell. Both the
        # log-size (sigma) and the phylo-SD (resd) coordinates are driven to
        # their unidentified boundary simultaneously.
        Random.seed!(90001)
        p = 20; m = 6
        phy = random_balanced_tree(p; branch_length = 0.20)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(n)
        β = [0.3, 0.4]
        μ = exp.(β[1] .+ β[2] .* x)                                    # no RE term
        y = Float64.([rand(Distributions.Poisson(μi)) for μi in μ])    # no overdispersion

        io = IOBuffer()
        local fit
        Logging.with_logger(Logging.SimpleLogger(io, Logging.Warn)) do
            fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      NegBinomial2(); data = (; y, x, species), tree = phy, se = true)
        end
        captured = String(take!(io))

        # The identical warning text `_vcov_from_hessian` emits everywhere else —
        # not a second, differently-worded message for this route.
        @test occursin("Hessian is numerically singular at the optimum", captured)
        @test occursin("pseudo-inverse", captured)
        @test occursin("NOT trustworthy", captured)
        @test occursin("sparse-Laplace GLMM (general-mean nuisance)", captured)

        # The pseudo-inverse fallback, not the old silent identity matrix.
        @test all(isfinite, fit.vcov)
        @test issymmetric(fit.vcov)
    end

    @testset "well-conditioned fit on the same route stays silent" begin
        # Same cell (NB2 phylo, constant sigma), but with real overdispersion
        # and real phylo signal — the healthy case from test_nb2_phylo_laplace.jl.
        Random.seed!(20260605)
        p = 32; m = 8
        phy = random_balanced_tree(p; branch_length = 0.20)
        species = repeat(1:p, inner = m)
        n = length(species)
        x = randn(n)
        β = [0.25, 0.35]; sz = 4.0; σphy = 0.45
        C = sigma_phy_dense(phy; σ²_phy = σphy^2)
        u = cholesky(Symmetric(C)).L * randn(p)
        μ = exp.(β[1] .+ β[2] .* x .+ u[species])
        y = Float64.([rand(Distributions.NegativeBinomial(sz, sz / (sz + μi))) for μi in μ])

        io = IOBuffer()
        local fit
        Logging.with_logger(Logging.SimpleLogger(io, Logging.Warn)) do
            fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                      NegBinomial2(); data = (; y, x, species), tree = phy, se = true)
        end
        captured = String(take!(io))

        @test isempty(strip(captured))   # no new noise on a healthy fit
        @test fit.converged
        @test all(isfinite, fit.vcov)
    end
end
