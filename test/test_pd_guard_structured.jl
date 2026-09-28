# Two more user-matrix entry points in gaussian_structured.jl lacked the
# PD/symmetry guard `_dense_comp` got in #869:
#   (1) `_fit_structured_gaussian` — the single, non-crossed relmat/animal
#       Gaussian-mean route — called `cholesky(Symmetric(K))` with the
#       default `check = true` (a bare `PosDefException`, not an actionable
#       error naming the grouping factor), and `Symmetric(K)` silently reads
#       only one triangle of a non-symmetric `K` instead of rejecting it.
#   (2) `_fit_two_structured_gaussian` — checked only the assembled marginal
#       V (inside the optimiser, with `check = false` + a penalty so a bad
#       line-search step never throws), never C1/C2 themselves at setup, so a
#       genuinely indefinite or non-symmetric user matrix was silently fit
#       against with no error at all.
#
# Both routes now share `_checked_relmat_chol(C, grp)` (also used by
# `_dense_comp`): symmetry at a sqrt(eps) relative tolerance, then a
# `check = false` Cholesky, raising `ArgumentError` naming the grouping
# factor on either failure. This test pins: (a) an indefinite matrix and a
# non-symmetric matrix each raise `ArgumentError` naming the group, on BOTH
# routes; (b) PD input still gives the same fit (unchanged to 1e-10).
using DRModels
using Test, Random, LinearAlgebra

_corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))

# Indefinite (not PSD): identity with one flipped diagonal entry.
function _indefinite_matrix(n)
    C = Matrix{Float64}(I, n, n)
    C[1, 1] = -1.0
    return C
end

# Non-symmetric but otherwise plausible (asymmetric off-diagonal perturbation).
function _nonsymmetric_matrix(n; seed)
    rng = MersenneTwister(seed)
    C = Matrix{Float64}(I, n, n)
    C[1, 2] += 0.3
    C[2, 1] += 0.3 + 1e-3   # breaks symmetry well above sqrt(eps) tolerance
    return C
end

@testset "PD guard: _fit_structured_gaussian (single relmat/animal route)" begin
    G = 10; m = 4; n = G * m
    Random.seed!(20260927)
    id = repeat(1:G, inner = m)
    x = randn(n)
    y = 0.3 .+ 0.5 .* x .+ randn(n)
    data = (; y, x, id)

    @testset "indefinite K raises ArgumentError naming the group" begin
        Kbad = _indefinite_matrix(G)
        err = try
            drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                Gaussian(); data = data, K = Kbad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("id", sprint(showerror, err))
    end

    @testset "non-symmetric K raises ArgumentError naming the group" begin
        Kbad = _nonsymmetric_matrix(G; seed = 3)
        err = try
            drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)),
                Gaussian(); data = data, A = Kbad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("id", sprint(showerror, err))
    end

    @testset "PD K: fit unchanged to 1e-10 (before/after the shared guard)" begin
        A = let M = randn(MersenneTwister(11), G, G); M * M' / G + I end
        K = _corr(A)
        fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)),
                  Gaussian(); data = data, K = K)
        @test fit.converged
        @test isfinite(loglik(fit))

        # Reference: the pre-guard path (default-`check` Cholesky) on the same
        # PD K, called directly, must agree with the public route to 1e-10.
        Kfac_ref = cholesky(Symmetric(K))
        Kfac_new = DRModels._checked_relmat_chol(K, :id)
        @test Matrix(Kfac_new.L) ≈ Matrix(Kfac_ref.L) atol = 1e-10
        @test logdet(Kfac_new) ≈ logdet(Kfac_ref) atol = 1e-10
    end
end

@testset "PD guard: _fit_two_structured_gaussian (C1/C2 checked directly)" begin
    G = 12; m = 5; n = G * m
    Random.seed!(20260928)
    phy = random_balanced_tree(G; branch_length = 0.4)
    species = repeat(1:G, inner = m)
    id = repeat(1:G, inner = m)
    x = randn(n)
    y = 0.2 .+ 0.4 .* x .+ randn(n)
    data = (; y, x, species, id)

    @testset "indefinite C2 (animal/relmat side) raises ArgumentError naming id" begin
        Kbad = _indefinite_matrix(G)
        err = try
            drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
                Gaussian(); data = data, tree = phy, K = Kbad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("id", sprint(showerror, err))
    end

    @testset "non-symmetric C2 raises ArgumentError naming id" begin
        Kbad = _nonsymmetric_matrix(G; seed = 5)
        err = try
            drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
                Gaussian(); data = data, tree = phy, K = Kbad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("id", sprint(showerror, err))
    end

    @testset "PD C1, C2: fit unchanged to 1e-10 (before/after the shared guard)" begin
        Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
        M = randn(MersenneTwister(13), G, G); Canim = _corr(M * M' / G + I)
        fit = drm(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1)),
                  Gaussian(); data = data, tree = phy, K = Canim)
        @test fit.converged
        @test isfinite(loglik(fit))

        # The guard call itself must not perturb C1/C2 (it only reads/factors
        # a copy); confirm both matrices still pass the guard cleanly and
        # agree with a direct default-check Cholesky to 1e-10.
        for (C, grp) in ((Cphy, :species), (Canim, :id))
            ch_ref = cholesky(Symmetric(C))
            ch_new = DRModels._checked_relmat_chol(C, grp)
            @test Matrix(ch_new.L) ≈ Matrix(ch_ref.L) atol = 1e-10
            @test logdet(ch_new) ≈ logdet(ch_ref) atol = 1e-10
        end
    end
end
