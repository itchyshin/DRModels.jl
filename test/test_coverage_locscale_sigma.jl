# DJ1: src/locscale_sigma.jl. The existing σ-axis test fits with se = false and
# never calls `_sigma_re_nll`. This file checks that helper against the q=2
# marginal, checks its gradient against a central difference, and runs the
# se = true observed-information path on a small seeded Gamma fit.
using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

_gamma_draw(rng, η, ψ) = (α = exp(ψ); μ = exp(η); rand(rng, Distributions.Gamma(α, μ / α)))

# File-scope so the inner-mode restart is not a `@testset` local closure.
function _dj1_sigma_nll(t, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
    warm[] = copy(seed_mode)
    return DRModels._sigma_re_nll(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, t, Zη, Zψ; warm = warm)
end

function _sigma_design(; G = 8, nper = 6, seed = 11, τ = 0.5, βμ = log(3.0), βψ = log(1.5))
    rng = MersenneTwister(seed)
    n = G * nper
    gidx = repeat(1:G, inner = nper)
    u = τ .* randn(rng, G)
    y = [_gamma_draw(rng, βμ, βψ + u[gidx[i]]) for i in 1:n]
    Xμ = ones(n, 1)
    Xψ = ones(n, 1)
    Q = sparse(1.0 * I, G, G)
    Zη, Zψ = DRModels._sigma_re_loadings(n)
    return (; y, Xμ, Xψ, gidx, G, Q, Zη, Zψ, βμ, βψ, τ)
end

@testset "σ-axis marginal NLL matches the q=2 Laplace marginal" begin
    d = _sigma_design()
    θ = vcat(d.βμ, d.βψ, log(d.τ))
    val = DRModels._sigma_re_nll(Val(:gamma), d.y, d.Xμ, d.Xψ, d.gidx, d.G, d.Q, θ, d.Zη, d.Zψ)
    Λ = DRModels._sigma_re_Lambda(log(d.τ))
    P = prior_precision(d.Q, DRModels._ls_inv2x2(Λ))
    direct, _, ok = DRModels._ls_marginal_nll(Val(:gamma), d.y, d.Xμ * d.βμ, d.Xψ * d.βψ,
                                               d.gidx, d.G, P, d.Zη, d.Zψ)
    @test ok
    @test isfinite(val)
    @test val ≈ direct atol = 1e-8 rtol = 1e-8
    # Λ is diagonal with the free scale on axis 1 and the pinned ε on axis 2.
    @test Λ[1, 2] == 0 && Λ[2, 1] == 0
    @test Λ[1, 1] ≈ d.τ^2 atol = 1e-12
    @test Λ[2, 2] == DRModels._SIGMA_RE_EPS^2

    warm = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    val_w = DRModels._sigma_re_nll(Val(:gamma), d.y, d.Xμ, d.Xψ, d.gidx, d.G, d.Q, θ,
                                    d.Zη, d.Zψ; warm = warm)
    @test val_w ≈ val atol = 1e-8 rtol = 1e-8
    @test warm[] isa Vector{Float64}
    @test length(warm[]) == 2 * d.G
    @test all(isfinite, warm[])
end

@testset "σ-axis gradient matches a central difference of the marginal NLL" begin
    d = _sigma_design(; G = 6, nper = 5, seed = 19)
    θ = vcat(d.βμ, d.βψ, log(0.4))
    nll(t) = DRModels._sigma_re_nll(Val(:gamma), d.y, d.Xμ, d.Xψ, d.gidx, d.G, d.Q, t, d.Zη, d.Zψ)
    g = DRModels._sigma_re_grad(Val(:gamma), d.y, d.Xμ, d.Xψ, d.gidx, d.G, d.Q, θ, d.Zη, d.Zψ)
    # The marginal is Float64-only (the inner mode rejects Dual), so the
    # oracle is a central difference at h = 1e-5.
    h = 1e-5
    g_fd = similar(θ)
    for j in eachindex(θ)
        tp = copy(θ); tm = copy(θ)
        tp[j] += h; tm[j] -= h
        g_fd[j] = (nll(tp) - nll(tm)) / (2h)
    end
    @test g ≈ g_fd atol = 1e-5 rtol = 1e-5
end

@testset "σ-axis se=true covariance is the inverse Hessian of the marginal" begin
    Random.seed!(7)
    G = 24
    nper = 10
    n = G * nper
    rng = MersenneTwister(7)
    gidx = repeat(1:G, inner = nper)
    τ = 0.5
    u = τ .* randn(rng, G)
    βμ = log(4.0)
    βψ = log(2.0)
    y = [_gamma_draw(rng, βμ, βψ + u[gidx[i]]) for i in 1:n]
    Xμ = ones(n, 1)
    Xψ = ones(n, 1)
    fit = DRModels._fit_sigma_axis_re(DRModels.Gamma(), Val(:gamma), y, Xμ, Xψ, gidx, G,
                                       ["(Intercept)"], ["(Intercept)"], "g";
                                       link = :log, se = true, g_tol = 1e-6)
    @test is_converged(fit)
    @test isfinite(loglik(fit))
    τ̂ = re_sd(fit)[:g_logsigma]
    @test isfinite(τ̂) && τ̂ > 0
    # True τ = 0.5. The band is the sampling range of this design, not a
    # tolerance on a published parity number.
    @test 0.2 < τ̂ < 1.2
    V = vcov(fit)
    @test size(V) == (3, 3)
    @test issymmetric(V)
    # Interior fit: every diagonal of the observed information is a variance.
    @test all(isfinite, V)
    @test all(>(0), diag(V))
    # Curvature of `_sigma_re_nll` with the inner mode restarted from the
    # optimum on every probe. A cold start jumps to a worse mode (the NLL
    # rises by a step-independent amount), which is not the Hessian. The
    # fitter instead differences `_sigma_re_grad`; this differences the NLL.
    # Measured max |V - V_nll| on this seed was 5.9e-6.
    θ̂ = coef(fit)
    Q = sparse(1.0 * I, G, G)
    Zη, Zψ = DRModels._sigma_re_loadings(n)
    warm = Ref{Union{Nothing,Vector{Float64}}}(nothing)
    f0 = DRModels._sigma_re_nll(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ; warm = warm)
    seed_mode = copy(warm[])
    h = 1e-4
    np = length(θ̂)
    H = zeros(np, np)
    for j in 1:np
        tp = copy(θ̂); tp[j] += h
        tm = copy(θ̂); tm[j] -= h
        fp = _dj1_sigma_nll(tp, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        fm = _dj1_sigma_nll(tm, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        # `2 * f0`, not `2f0`: Julia parses `2f0` as the Float32 literal 2.
        H[j, j] = (fp - 2 * f0 + fm) / h^2
    end
    for j in 1:np, i in 1:(j - 1)
        pp = copy(θ̂); pp[i] += h; pp[j] += h
        pm = copy(θ̂); pm[i] += h; pm[j] -= h
        mp = copy(θ̂); mp[i] -= h; mp[j] += h
        mm = copy(θ̂); mm[i] -= h; mm[j] -= h
        hpp = _dj1_sigma_nll(pp, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        hpm = _dj1_sigma_nll(pm, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        hmp = _dj1_sigma_nll(mp, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        hmm = _dj1_sigma_nll(mm, warm, seed_mode, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ)
        H[i, j] = (hpp - hpm - hmp + hmm) / (4h^2)
        H[j, i] = H[i, j]
    end
    V_nll = inv(Symmetric(H))
    @test V ≈ V_nll atol = 2e-5 rtol = 1e-4
    gopt = DRModels._sigma_re_grad(Val(:gamma), y, Xμ, Xψ, gidx, G, Q, θ̂, Zη, Zψ)
    @test norm(gopt) < 1e-4
end
