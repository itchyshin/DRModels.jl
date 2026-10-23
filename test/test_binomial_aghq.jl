# #712/#713: the Binomial 1-D `(1 | g)` route used a non-adaptive fixed 32-node
# Gauss–Hermite grid on the PRIOR scale (b = √2 σ_b z). Bernoulli 0/1 responses
# were fine, but GROUPED trials (n trials/row > 1) with an informative group SD
# give a narrow per-group posterior the grid does not resolve: a measurement on
# the DGP below found −9.73 nat of logLik error at the fitted optimum, and a
# 4.6–20 nat error evaluated at the AGHQ-accurate θ̂ across RE SD 0.5–0.8 (see
# the after-task report). `_fit_binomial_ranef` now goes through the shared
# adaptive-quadrature helper (`_aghq_marginal_loglik`, #834) with q = 1; K = 1
# is exactly Laplace (the drmTMB/TMB integrator), and the default
# `_BINOMIAL_RANEF_AGHQ_K = 3` is swept against an INDEPENDENT AGHQ-40
# implementation below.
using DRModels
using Test, Random, ForwardDiff
using LinearAlgebra: SymTridiagonal, eigen
import Distributions

# --- independent 1-D AGHQ-K reference (own Newton mode finder + K-node rule) --
function _gh1d(K)
    β = [sqrt(k / 2) for k in 1:(K-1)]
    E = eigen(SymTridiagonal(zeros(K), β))
    return E.values, sqrt(π) .* E.vectors[1, :] .^ 2
end
function _ref_group_1d(h; K = 40)
    b = 0.0
    for _ in 1:200
        gr = ForwardDiff.derivative(h, b)
        Hm = -ForwardDiff.derivative(x -> ForwardDiff.derivative(h, x), b)
        δ = gr / Hm; t = 1.0
        while h(b + t * δ) < h(b) - 1e-12 && t > 1e-10
            t /= 2
        end
        b += t * δ
        abs(t * δ) < 1e-12 && break
    end
    Hm = -ForwardDiff.derivative(x -> ForwardDiff.derivative(h, x), b)
    K == 1 && return h(b) + 0.5 * log(2π) - 0.5 * log(Hm)   # exact Laplace formula
    C = sqrt(1 / Hm)
    z, w = _gh1d(K); lw = log.(w)
    terms = [lw[k] + h(b + sqrt(2) * C * z[k]) + z[k]^2 for k in 1:K]
    mx = maximum(terms)
    return 0.5 * log(2) + log(C) + mx + log(sum(exp.(terms .- mx)))
end
function _ref_loglik_1d(llf, g, η0, σb; K = 40)
    tot = 0.0
    for j in 1:maximum(g)
        idx = findall(==(j), g)
        h(b) = sum(llf(i, η0[i] + b) for i in idx) - 0.5 * (b / σb)^2 - 0.5 * log(2π) - log(σb)
        tot += _ref_group_1d(h; K)
    end
    return tot
end

# Grouped-trial DGP: G groups, m obs/group, `ntrials` trials/row, RE SD `sdb`.
function _dgp_1d(seed; G = 100, m = 30, ntrials = 20, sdb = 0.7, β0 = 0.1)
    Random.seed!(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    b = sdb .* randn(G)
    μ = 1 ./ (1 .+ exp.(-(β0 .+ b[g])))
    s = [rand(Distributions.Binomial(ntrials, μ[i])) for i in 1:n]
    fail = ntrials .- s
    return (; s = Float64.(s), fail = Float64.(fail), g), n, g, ntrials
end

@testset "Binomial (1 | g) grouped-trial AGHQ: K sweep vs independent AGHQ-40 (#712/#713)" begin
    data, n, g, ntrials = _dgp_1d(20260927)
    ntr = fill(Float64(ntrials), n)
    Xμ = ones(n, 1)

    errs = Dict{Int, Float64}()
    for K in (1, 2, 3, 4, 5, 8)
        fit = DRModels._fit_binomial_ranef(DRModels.Binomial(), data.s, ntr, Xμ, g, maximum(g),
                                            ["(Intercept)"], :g, 1e-8; nq = K)
        θ̂ = fit.theta
        η0 = fill(θ̂[1], n)
        llf = (i, η) -> Distributions.logpdf(Distributions.Binomial(ntrials, 1 / (1 + exp(-η))), Int(data.s[i]))
        ref40 = _ref_loglik_1d(llf, g, η0, exp(θ̂[2]))
        errs[K] = DRModels.loglik(fit) - ref40
        @test fit.converged
    end

    @test abs(errs[1]) > 0.01                        # K = 1 (Laplace) is NOT within tolerance here
    @test abs(errs[2]) > 0.01                        # K = 2 still misses (measured ≈ 0.033 nat)
    @test abs(errs[DRModels._BINOMIAL_RANEF_AGHQ_K]) < 0.01   # default K = 3: measured ≈ 0.0046 nat
    @test abs(errs[8]) < 1e-3                          # larger K keeps tightening

    # K = 1 equals the exact Laplace formula (independent implementation) at the
    # SAME θ̂ — this is drmTMB's own integrator for this route.
    fit1 = DRModels._fit_binomial_ranef(DRModels.Binomial(), data.s, ntr, Xμ, g, maximum(g),
                                         ["(Intercept)"], :g, 1e-8; nq = 1)
    θ̂1 = fit1.theta
    η0 = fill(θ̂1[1], n)
    llf = (i, η) -> Distributions.logpdf(Distributions.Binomial(ntrials, 1 / (1 + exp(-η))), Int(data.s[i]))
    ref_laplace = _ref_loglik_1d(llf, g, η0, exp(θ̂1[2]); K = 1)
    @test DRModels.loglik(fit1) ≈ ref_laplace atol = 1e-8

    # --- Regression guard: the OLD fixed 32-node PRIOR-SCALE grid (pre-#834/#712),
    # reconstructed here (not `include`d — it no longer exists in src/binomial.jl),
    # is off by > 1 nat on this same DGP evaluated at the AGHQ-accurate θ̂.
    function _old_grid_nll(θ, s, ntr, members, K = 32)
        n = length(s)
        sint = round.(Int, s); nint = round.(Int, ntr)
        z, w = DRModels._gauss_hermite(K); logw = log.(w); rt2 = sqrt(2.0); lπ = log(π)
        βμ = θ[1]; σb = exp(θ[2]); η0 = fill(βμ, n)
        s_ = 0.0
        for idx in members
            isempty(idx) && continue
            terms = Vector{Float64}(undef, K)
            for k in 1:K
                δ = rt2 * σb * z[k]; gll = logw[k]
                for i in idx
                    μ = 1 / (1 + exp(-clamp(η0[i] + δ, -15.0, 15.0)))
                    gll += Distributions.logpdf(Distributions.Binomial(nint[i], μ), sint[i])
                end
                terms[k] = gll
            end
            mx = maximum(terms)
            s_ -= (-0.5 * lπ + mx + log(sum(exp.(terms .- mx))))
        end
        return s_
    end
    fit3 = DRModels._fit_binomial_ranef(DRModels.Binomial(), data.s, ntr, Xμ, g, maximum(g),
                                         ["(Intercept)"], :g, 1e-8; nq = DRModels._BINOMIAL_RANEF_AGHQ_K)
    θ̂3 = fit3.theta
    members = [Int[] for _ in 1:maximum(g)]
    for i in 1:n
        push!(members[g[i]], i)
    end
    old_reported = -_old_grid_nll(θ̂3, data.s, ntr, members)
    η0 = fill(θ̂3[1], n)
    llf = (i, η) -> Distributions.logpdf(Distributions.Binomial(ntrials, 1 / (1 + exp(-η))), Int(data.s[i]))
    ref_at_new_theta = _ref_loglik_1d(llf, g, η0, exp(θ̂3[2]))
    @test abs(old_reported - ref_at_new_theta) > 1.0   # old grid: fails (>1 nat error)
    @test abs(DRModels.loglik(fit3) - ref_at_new_theta) < 0.01   # new AGHQ: passes
end
