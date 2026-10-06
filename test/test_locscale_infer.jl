# Wald inference + group-level summaries for the location–scale model (#202).
# The correctness gate cross-checks the observed information — built as the
# finite-difference Jacobian of the EXACT outer gradient — against an independent
# second finite difference of the marginal `_ls_fit_nll`. Agreement confirms the
# Hessian assembly (no transpose/index slip). Then a fit-level test checks the
# attached vcov / SEs / components are valid.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_nb2_draw_i(η, ψ) = (r = exp(ψ); μ = exp(η);
                     Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))

# Dense Hessian by central second differences of a scalar θ-function.
function _fd_hess(f, θ; h = 1e-4)
    p = length(θ); H = zeros(p, p)
    for j in 1:p, k in 1:p
        a = copy(θ); a[j] += h; a[k] += h
        b = copy(θ); b[j] += h; b[k] -= h
        c = copy(θ); c[j] -= h; c[k] += h
        d = copy(θ); d[j] -= h; d[k] -= h
        H[j, k] = (f(a) - f(b) - f(c) + f(d)) / (4h^2)
    end
    return H
end

@testset "location–scale observed information vs nll second differences" begin
    Random.seed!(909)
    G = 5; m = 8; n = G * m
    gidx = repeat(1:G, inner = m)
    x = randn(n); z = randn(n)
    Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), z)
    Λt = DRModels._ls_lc_to_Λ([log(0.4), 0.1, log(0.5)])
    Lt = cholesky(Symmetric(Λt)).L
    A = [Lt * randn(2) for _ in 1:G]
    y = [_nb2_draw_i(0.3 + 0.4x[i] + A[gidx[i]][1], 0.2 + A[gidx[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)
    θ = [0.25, 0.35, 0.1, -0.05, log(0.45), 0.08, log(0.55)]   # generic, not the optimum

    Hg = Matrix(DRModels._ls_obs_information(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ))
    f = θ -> DRModels._ls_fit_nll(Val(:nb2), y, Xμ, Xψ, gidx, G, Q, θ)
    Hn = _fd_hess(f, θ)
    @test maximum(abs.(Hg .- Hn)) < 1e-2 * (1 + maximum(abs.(Hn)))
    @test Hg ≈ Hg'                                   # symmetric by construction
end

@testset "location–scale fit attaches valid vcov / SEs / components" begin
    Random.seed!(7171)
    G = 25; m = 25; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    Λt = [0.25 0.05; 0.05 0.16]
    LΛ = cholesky(Symmetric(Λt)).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nb2_draw_i(0.5 + 0.4x[i] + A[species[i]][1], 0.3 + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)

    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, species, G, Q; se = true)
    p = length(fit.θ)
    @test fit.vcov !== nothing
    @test isposdef(Symmetric(fit.vcov))              # PD at the optimum
    @test length(fit.se) == p
    @test all(isfinite, fit.se) && all(fit.se .> 0)

    # components are consistent with Λ, and present even without SEs.
    @test fit.components.sd_mu ≈ sqrt(fit.Lambda[1, 1])
    @test fit.components.sd_psi ≈ sqrt(fit.Lambda[2, 2])
    @test fit.components.cor_mu_psi ≈
          fit.Lambda[1, 2] / sqrt(fit.Lambda[1, 1] * fit.Lambda[2, 2])
    fit0 = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, species, G, Q; se = false)
    @test fit0.vcov === nothing && fit0.se === nothing
    @test fit0.components.sd_mu > 0
end

# #870 review finding: at a true mean–scale correlation of 0.999 the
# finite-difference Hessian in `_ls_obs_information` can go non-finite
# ("Terminated early due to NaN in Hessian"), and `inv(Matrix(H))` in
# `_ls_vcov` then throws `ArgumentError: matrix contains Infs or NaNs` --
# NOT a `SingularException`, so the bare `catch err; err isa
# SingularException ...` guard did not catch it and the fit crashed instead
# of degrading gracefully. Reproduces the review's probe (scratchpad
# probe870.jl "near-singular Λ (ρ=0.999)" case) with the exact seed/shape.
@testset "location–scale fit at a near-singular mean-scale Λ does not throw (#870 review)" begin
    _nbdraw(η, ψ) = (μ = exp(η); r = exp(-2ψ);
                     Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))
    # Match the review's probe (scratchpad probe870.jl) exactly, including its
    # single seed shared across an "ordinary Λ" case run first and the
    # "near-singular Λ (ρ=0.999)" case that actually crashed -- only the
    # second case is asserted on, but the RNG stream must be consumed in the
    # same order to land on the same near-singular fit.
    Random.seed!(2026)
    fit = nothing
    for Λt in ([0.25 0.05; 0.05 0.16], [0.25 0.1998; 0.1998 0.16])   # ordinary, then ρ = 0.999
        G = 25; m = 25; n = G * m
        sp = repeat(1:G, inner = m); x = randn(n)
        LΛ = cholesky(Symmetric(Λt + 1e-12I)).L
        A = [LΛ * randn(2) for _ in 1:G]
        y = [_nbdraw(0.5 + 0.4x[i] + A[sp[i]][1], 0.3 + 0.2x[i] + A[sp[i]][2]) for i in 1:n]
        Xμ = hcat(ones(n), x)
        gidx, Gd = DRModels._group_index(sp)
        Q = sparse(1.0 * I, Gd, Gd)
        fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xμ, gidx, Gd, Q; g_tol = 1e-8, se = true)
    end
    @test fit !== nothing
    @test isfinite(fit.nll)
    @test all(isfinite, fit.θ)
    @test all(isfinite, fit.Lambda)
    # The Hessian is (near-)singular in this regime -- the fit must either flag
    # it (nothing vcov / se, per the existing `_ls_vcov`/`_ls_whitened_vcov`
    # convention) or report a FINITE vcov with finite, positive SEs; never
    # garbage (NaN/Inf) and never a crash. With the other fixes landed in
    # batch 2 (most likely the whitened Λ⁻¹ callers #862/#865/#868; not
    # bisected) the finite-difference Hessian on this exact seed is finite,
    # so the fit now takes the second branch (measured on the batch-2
    # branch: all SEs finite, the log L22 SE ≈ 108, i.e. weakly identified
    # near ρ = 0.999, as expected); before them it took the first.
    if fit.vcov === nothing
        @test fit.se === nothing
    else
        @test all(isfinite, fit.vcov)
        @test fit.se !== nothing && all(isfinite, fit.se) && all(>(0), fit.se)
    end
end
