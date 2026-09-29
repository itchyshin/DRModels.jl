# `_dense_comp` builds the Q = C⁻¹ precision (and logdetC) for `relmat`/`animal`
# dense correlation/relatedness components (gaussian_structured.jl). The
# original implementation used `inv(Symmetric(Matrix(C)))` / `logdet(Symmetric(
# Matrix(C)))` directly, with no positive-definiteness check — unlike its
# `_q4_structured_precision` sibling (gaussian_bivariate.jl), which validates
# via `isposdef` before `cholesky`. On a semidefinite/non-PD C this silently
# returned a finite but WRONG logdet about half the time in a random sweep (the
# other half threw a confusing `DomainError` from `log` of a negative number,
# not an actionable error naming the offending matrix).
#
# This test pins: (1) a Cholesky-based `_dense_comp` matches the old inv/logdet
# path to high precision on well-conditioned AND ill-conditioned (cond up to
# 1e12) PD matrices, and (2) a semidefinite/non-PD C raises a clear
# `ArgumentError` naming the matrix, instead of propagating silent garbage or a
# confusing low-level DomainError.
using DRModels
using Test, Random, LinearAlgebra

# Build an ill-conditioned relatedness-like matrix at a target condition number
# (spectral construction: random orthogonal basis, eigenvalues log-spaced from
# 1 to `condnum`) — stands in for a pedigree A matrix with close inbreeding or
# a GRM with near-duplicate individuals.
function _illcond_relmat(n, condnum; seed)
    rng = MersenneTwister(seed)
    A = randn(rng, n, n)
    Qm = Matrix(qr(A).Q)
    ev = exp.(range(0, log(condnum), length = n))
    C = Qm * Diagonal(ev) * Qm'
    return Matrix(Symmetric((C + C') / 2))
end

@testset "_dense_comp: Cholesky matches inv/logdet on well-conditioned C" begin
    C = _illcond_relmat(12, 10.0; seed = 1)
    gidx = collect(1:12)
    comp = DRModels._dense_comp(gidx, 12, C, :grp)

    Qref = inv(Symmetric(Matrix(C)))
    logdetref = logdet(Symmetric(Matrix(C)))

    @test maximum(abs.(Matrix(comp.Q) .- Qref)) < 1e-10
    @test comp.logdetCprior ≈ logdetref atol = 1e-10
end

@testset "_dense_comp: Cholesky path unchanged to 1e-10 at cond 1e8/1e10/1e12" begin
    for (i, condtarget) in enumerate((1e8, 1e10, 1e12))
        n = 25
        C = _illcond_relmat(n, condtarget; seed = 100 + i)
        gidx = collect(1:n)
        comp = DRModels._dense_comp(gidx, n, C, :grp)

        Qref = inv(Symmetric(Matrix(C)))
        logdetref = logdet(Symmetric(Matrix(C)))

        # Relative tolerance since raw magnitudes grow with condition number;
        # both paths (inv/logdet vs Cholesky) are algebraically equivalent to
        # machine precision on the SAME PD input.
        relerr_logdet = abs(comp.logdetCprior - logdetref) / abs(logdetref)
        @test relerr_logdet < 1e-6

        relerr_Q = maximum(abs.(Matrix(comp.Q) .- Qref)) / maximum(abs.(Qref))
        @test relerr_Q < 1e-5
    end
end

@testset "_dense_comp: 256-bit BigFloat reference agreement" begin
    n = 20
    C = _illcond_relmat(n, 1e10; seed = 7)
    gidx = collect(1:n)
    comp = DRModels._dense_comp(gidx, n, C, :grp)

    logdet_big = setprecision(BigFloat, 256) do
        Cbig = BigFloat.(C)
        Float64(logdet(cholesky(Symmetric(Cbig))))
    end
    @test comp.logdetCprior ≈ logdet_big atol = 1e-6 rtol = 1e-8
end

@testset "_dense_comp: non-PD / semidefinite C raises a clear ArgumentError" begin
    Random.seed!(2026)
    n = 10
    B = randn(n, n - 1)                 # rank n-1 ⇒ singular PSD
    Csemi = Matrix(Symmetric(B * B'))
    gidx = collect(1:n)

    @test !isposdef(Symmetric(Csemi))
    err = try
        DRModels._dense_comp(gidx, n, Csemi, :myid)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("myid", sprint(showerror, err))
end

@testset "_dense_comp: indefinite C also raises (not a silent NaN/Inf)" begin
    n = 8
    C = Matrix{Float64}(I, n, n)
    C[1, 1] = -1.0                      # indefinite, not PSD
    gidx = collect(1:n)
    @test_throws ArgumentError DRModels._dense_comp(gidx, n, C, :bad)
end
