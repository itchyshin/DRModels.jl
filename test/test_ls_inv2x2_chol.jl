# `_ls_inv2x2` (locscale_inner.jl) inverts a 2×2 Λ = LLᵀ via the naive
# `a*d - b*c` determinant AFTER Λ has already been formed as a matrix from its
# log-Cholesky factor L = [l11 0; l21 l22]. Forming Λ first computes
# `d = l21² + l22²` as a Float64 SUM, which already destroys l22 whenever
# |l21| ≫ l22 — no formula operating on the formed matrix (the naive
# determinant here, or a fresh `cholesky` re-factorisation, cf. PR #865's
# identical `coevo_marginal_cov` failure) can recover what `d` already lost.
# The stable route inverts straight from the log-Cholesky factor:
#   Λ⁻¹ = L⁻ᵀL⁻¹, log det Λ = 2(log l11 + log l22).
# This gates the fix (`_ls_lc_inv2x2`/`_ls_lc_logdetΛ`) against a 256-bit
# BigFloat reference at extreme log-Cholesky diagonals, checks it agrees with
# the existing matrix-formed path away from the singular boundary, and
# documents that the matrix-formed path itself fails at the same diagonals
# (present on origin/main both before and after this fix — `_ls_inv2x2` and
# `_ls_lc_to_Λ` are untouched; only call sites were rerouted).
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Independent BigFloat reference: builds L and Λ = LLᵀ entirely in 256-bit
# arithmetic from the raw log-Cholesky components, then inverts/logdets that.
function _bigfloat_lc_reference(v)
    l11 = exp(BigFloat(v[1])); l21 = BigFloat(v[2]); l22 = exp(BigFloat(v[3]))
    L = BigFloat[l11 big"0.0"; l21 l22]
    Λ = L * transpose(L)
    return inv(Λ), logdet(Λ)
end

const _EXTREME_DIAGONALS = (-12.0, -18.0, -25.0)

@testset "_ls_lc_inv2x2 / _ls_lc_logdetΛ vs 256-bit BigFloat reference" begin
    setprecision(BigFloat, 256) do
        for l22diag in _EXTREME_DIAGONALS
            # l21 = 1.0 ≫ l22 = exp(l22diag): the regime that makes `d = l21² +
            # l22²` swallow l22 once Λ is formed in Float64.
            v = [0.3, 1.0, l22diag]
            Λinv_ref, logdetΛ_ref = _bigfloat_lc_reference(v)

            Λinv = DRModels._ls_lc_inv2x2(v)
            logdetΛ = DRModels._ls_lc_logdetΛ(v)

            @test all(isfinite, Λinv)
            @test isfinite(logdetΛ)
            rel_inv = maximum(abs, BigFloat.(Λinv) .- Λinv_ref) / maximum(abs, Λinv_ref)
            rel_logdet = abs(BigFloat(logdetΛ) - logdetΛ_ref) / abs(logdetΛ_ref)
            @test rel_inv <= 1e-10
            @test rel_logdet <= 1e-10
        end
    end
end

@testset "_ls_lc_inv2x2 agrees with the matrix-formed path in the well-conditioned regime" begin
    Random.seed!(20260927)
    for _ in 1:20
        # l21 and l22 comparable in magnitude: no cancellation in either path.
        v = [0.6 * randn(), 0.6 * randn(), 0.6 * randn()]
        Λ = DRModels._ls_lc_to_Λ(v)
        Λinv_old = DRModels._ls_inv2x2(Λ)
        Λinv_new = DRModels._ls_lc_inv2x2(v)
        @test all(isfinite, Λinv_old)
        @test all(isfinite, Λinv_new)
        rel = maximum(abs, Λinv_new .- Λinv_old) / maximum(abs, Λinv_old)
        @test rel <= 1e-12
    end
end

@testset "matrix-formed path (_ls_inv2x2 ∘ _ls_lc_to_Λ) loses the claimed digits at the same diagonals" begin
    setprecision(BigFloat, 256) do
        for l22diag in (-18.0, -25.0)
            v = [0.3, 1.0, l22diag]
            Λinv_ref, _ = _bigfloat_lc_reference(v)
            Λ = DRModels._ls_lc_to_Λ(v)
            Λinv_old = DRModels._ls_inv2x2(Λ)
            rel_old = maximum(abs, BigFloat.(Λinv_old) .- Λinv_ref) / maximum(abs, Λinv_ref)
            # Documents the pre-existing catastrophic cancellation: the
            # matrix-formed path is unreliable here (`_ls_inv2x2` itself is
            # untouched by the fix — call sites were rerouted around it).
            @test rel_old > 1e-10
        end
    end
end

# Fit-level regression: the unit tests above exercise `_ls_lc_inv2x2` in
# isolation, but no test previously ran a FIT through the four rerouted call
# sites (`locscale_fit.jl`, `locscale_grad.jl`, `locscale_profile.jl`,
# `gaussian_locscale_phylo.jl`). A coupled NB2 location-scale fit at a
# moderately high mean-scale correlation (ρ = 0.95, well short of the
# separate near-singular-Hessian crash at ρ = 0.999 fixed in #883) exercises
# `prior_precision(Q, _ls_lc_inv2x2(...))` on every Newton/gradient step.
# Reference nll/θ/Λ were measured on the parent commit (6fb5172a0, before
# #870 rerouted these callers to `_ls_lc_inv2x2`) with the identical seed and
# data — the two paths are mathematically equivalent in this well-conditioned
# regime (already shown to agree to 1e-12 above), so a fit through the
# rerouted callers must reproduce the pre-#870 optimum to high precision.
@testset "coupled NB2 location-scale fit through the rerouted _ls_lc_inv2x2 callers (ρ=0.95)" begin
    _nbdraw(η, ψ) = (μ = exp(η); r = exp(-2ψ);
                     Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))
    Random.seed!(4242)
    Λt = [0.25 0.19; 0.19 0.16]                      # cor = 0.19 / (0.5*0.4) = 0.95
    G = 20; m = 20; n = G * m
    sp = repeat(1:G, inner = m); x = randn(n)
    LΛ = cholesky(Symmetric(Λt + 1e-12I)).L
    A = [LΛ * randn(2) for _ in 1:G]
    y = [_nbdraw(0.5 + 0.4x[i] + A[sp[i]][1], 0.3 + 0.2x[i] + A[sp[i]][2]) for i in 1:n]
    Xμ = hcat(ones(n), x)
    gidx, Gd = DRModels._group_index(sp)
    Q = sparse(1.0 * I, Gd, Gd)

    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xμ, gidx, Gd, Q; g_tol = 1e-8, se = false)

    # Pre-#870 reference, measured on parent commit 6fb5172a0 with this exact
    # seed/data (script: `_fit_locscale(Val(:nb2), y, Xμ, Xμ, gidx, Gd, Q;
    # g_tol=1e-8, se=false)`).
    nll_pre870 = 625.5385855683999
    theta_pre870 = [0.4784938788075241, 0.4122900997490708, 0.41155200257791813,
                    0.3132473687459513, -0.8049941668703925, 0.42955211622351375,
                    -1.292571271971299]
    Lambda_pre870 = [0.19988994602946675 0.19204868523315913;
                      0.19204868523315913 0.2599003536647257]

    @test all(isfinite, fit.θ)
    @test isapprox(fit.nll, nll_pre870; atol = 1e-8)
    @test isapprox(fit.θ, theta_pre870; atol = 1e-8)
    @test isapprox(fit.Lambda, Lambda_pre870; atol = 1e-8)
end
