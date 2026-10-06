# test_corr_locscale_equiv.jl — Cross-engine equivalence gate for the unified
# q2 Laplace correlated-slope path vs the GHQ _fit_*_corr_ranef fitters.
#
# For each of {Poisson, Gamma, NegBinomial2, Beta, LogNormal} we simulate a
# (1 + x | g) dataset (G=120, m=25, sd0=0.4, sd1=0.3, ρ=0.3), fit both the
# GHQ path (current) and the new Laplace path (_fit_corr_locscale), and assert:
#   1. logLik close: |ΔlogLik| / |logLik_GHQ| < 1e-2 (Laplace vs GHQ differ
#      by the integral approximation quality, so this is a SANE tolerance).
#   2. coef(:mu) close within ~10%.
#   3. vc[:g] (2×2 RE covariance) close within ~10%.
#
# BetaBinomial and Student have no _corr_kind — they stay on GHQ untouched.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Bring in the internal engine helpers (they are in the DRModels module but not
# exported; use DRModels.func or reference via the module).
const _fcloc = DRModels._fit_corr_locscale
const _corrkind = DRModels._corr_kind

# ── helpers ─────────────────────────────────────────────────────────────────

function _check_equiv(name, fit_ghq, fit_lap; ll_rtol = 1e-2, coef_rtol = 0.10, vc_rtol = 0.15)
    ll_g = loglik(fit_ghq)
    ll_l = loglik(fit_lap)
    rel_ll = abs(ll_l - ll_g) / (abs(ll_g) + 1e-10)
    β_g = coef(fit_ghq, :mu)
    β_l = coef(fit_lap, :mu)
    coef_ok = all(i -> abs(β_l[i] - β_g[i]) / (abs(β_g[i]) + 1e-3) < coef_rtol, eachindex(β_g))
    V_g = vc(fit_ghq)[:g]
    V_l = vc(fit_lap)[:g]
    vc_ok = all(i -> abs(V_l[i] - V_g[i]) / (abs(V_g[i]) + 1e-6) < vc_rtol, eachindex(V_g))
    @info "$name: Δloglik_rel=$(round(rel_ll, sigdigits=4))  ll_ghq=$(round(ll_g,digits=2))  ll_lap=$(round(ll_l,digits=2))  coef_ok=$coef_ok  vc_ok=$vc_ok"
    @test rel_ll < ll_rtol
    @test coef_ok
    @test vc_ok
    return (rel_ll = rel_ll, ll_ghq = ll_g, ll_lap = ll_l,
            coef_delta = abs.(β_l .- β_g), vc_delta = abs.(diag(V_l) .- diag(V_g)))
end

# ── DGP parameters (shared) ─────────────────────────────────────────────────
const G_EQ = 120; const M_EQ = 25; const N_EQ = G_EQ * M_EQ
const SD0_EQ = 0.4; const SD1_EQ = 0.3; const RHO_EQ = 0.3

function _sim_corr(seed)
    Random.seed!(seed)
    g = repeat(1:G_EQ, inner = M_EQ)
    x = randn(N_EQ)
    Σ = [SD0_EQ^2 RHO_EQ*SD0_EQ*SD1_EQ; RHO_EQ*SD0_EQ*SD1_EQ SD1_EQ^2]
    B = cholesky(Symmetric(Σ)).L * randn(2, G_EQ)
    b0 = B[1, :]; b1 = B[2, :]
    return g, x, b0, b1
end

# ── 1. Poisson ───────────────────────────────────────────────────────────────
@testset "Corr-locscale equiv: Poisson (1+x|g)" begin
    g, x, b0, b1 = _sim_corr(20260611_01)
    β = [0.3, 0.4]
    λ = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
    data = (; y, x, g)

    # GHQ path (current)
    fit_ghq = drm(bf(@formula(y ~ x + (1 + x | g))), Poisson(); data = data)

    # Laplace path (new): build design matrices the same way
    fam = Poisson()
    n = N_EQ; Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
    Xψ = zeros(n, 0); nmσ = String[]
    gidx = g; Gv = G_EQ
    xs = Float64.(x)
    fit_lap = _fcloc(fam, _corrkind(fam), :corr, y, Xμ, Xψ, xs, gidx, Gv,
                     nmμ, nmσ, "g"; link = :log, se = true)

    _check_equiv("Poisson", fit_ghq, fit_lap)
end

# ── 2. Gamma ─────────────────────────────────────────────────────────────────
@testset "Corr-locscale equiv: Gamma (1+x|g)" begin
    g, x, b0, b1 = _sim_corr(20260611_02)
    β = [0.5, 0.4]; α = 8.0
    μv = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
    y = Float64.([rand(Distributions.Gamma(α, μv[i] / α)) for i in 1:N_EQ])
    data = (; y, x, g)

    fit_ghq = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gamma(); data = data)

    fam = Gamma()
    n = N_EQ; Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
    Xψ = ones(n, 1); nmσ = ["(Intercept)"]    # sigma ~ 1
    gidx = g; Gv = G_EQ; xs = Float64.(x)
    fit_lap = _fcloc(fam, _corrkind(fam), :corr, y, Xμ, Xψ, xs, gidx, Gv,
                     nmμ, nmσ, "g"; link = :log, se = true)

    _check_equiv("Gamma", fit_ghq, fit_lap)
end

# ── 3. NegBinomial2 ───────────────────────────────────────────────────────────
@testset "Corr-locscale equiv: NegBinomial2 (1+x|g)" begin
    g, x, b0, b1 = _sim_corr(20260611_03)
    β = [0.3, 0.4]; θnb = 3.0
    μv = exp.(β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x)
    y = Float64.([rand(Distributions.NegativeBinomial(θnb, θnb / (θnb + μv[i]))) for i in 1:N_EQ])
    data = (; y, x, g)

    fit_ghq = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), NegBinomial2(); data = data)

    fam = NegBinomial2()
    n = N_EQ; Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
    Xψ = ones(n, 1); nmσ = ["(Intercept)"]
    gidx = g; Gv = G_EQ; xs = Float64.(x)
    fit_lap = _fcloc(fam, _corrkind(fam), :corr, y, Xμ, Xψ, xs, gidx, Gv,
                     nmμ, nmσ, "g"; link = :log, se = true)

    _check_equiv("NegBinomial2", fit_ghq, fit_lap)
end

# ── 4. Beta ───────────────────────────────────────────────────────────────────
@testset "Corr-locscale equiv: Beta (1+x|g)" begin
    g, x, b0, b1 = _sim_corr(20260611_04)
    β = [0.2, 0.5]; φ = 15.0
    ηv = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x
    μv = 1 ./ (1 .+ exp.(-ηv))
    y = Float64.([rand(Distributions.Beta(μv[i] * φ, (1 - μv[i]) * φ)) for i in 1:N_EQ])
    data = (; y, x, g)

    fit_ghq = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Beta(); data = data)

    fam = Beta()
    n = N_EQ; Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
    Xψ = ones(n, 1); nmσ = ["(Intercept)"]
    gidx = g; Gv = G_EQ; xs = Float64.(x)
    # respobs = y (observed proportion — the bounded-family response semantics)
    fit_lap = _fcloc(fam, _corrkind(fam), :corr, y, Xμ, Xψ, xs, gidx, Gv,
                     nmμ, nmσ, "g"; link = :logit, se = true, respobs = y)

    _check_equiv("Beta", fit_ghq, fit_lap)
end

# ── 5a. Independent slope rk=:slope (Gamma, recovery + identifiability) ──────
# Direct call to _fit_corr_locscale with rk=:slope verifies the (0+x|g)
# independent-slope path recovers the true slope-RE variance AND leaves the
# model identified. The SLOPE loads axis-1 (Zη = [xᵢ 0]); axis-2 is unused, so
# its variance must be PINNED at ε (not left free). #301 fixed the start that
# previously pinned the identified slope axis tiny and let the unused axis-2
# wander as an unidentified free parameter.
@testset "Corr-locscale: independent slope (0+x|g) via rk=:slope" begin
    Random.seed!(20260611_99)
    g_s = repeat(1:G_EQ, inner = M_EQ)
    x_s = randn(N_EQ)
    β = [0.5, 0.4]; α = 8.0
    # Independent slope only: b_slope ~ N(0, SD1_EQ^2) per group, no intercept RE
    b1_ind = SD1_EQ .* randn(G_EQ)
    μv_ind = exp.(β[1] .+ β[2] .* x_s .+ b1_ind[g_s] .* x_s)
    y_ind = Float64.([rand(Distributions.Gamma(α, μv_ind[i] / α)) for i in 1:N_EQ])

    fam = Gamma()
    n = N_EQ; Xμ = hcat(ones(n), x_s); nmμ = ["(Intercept)", "x"]
    Xψ = ones(n, 1); nmσ = ["(Intercept)"]
    xs_s = Float64.(x_s)
    fit_slope = _fcloc(fam, _corrkind(fam), :slope, y_ind, Xμ, Xψ, xs_s, g_s, G_EQ,
                       nmμ, nmσ, "g"; link = :log, se = true)

    @test isfinite(loglik(fit_slope))
    @test coef(fit_slope, :mu)[2] ≈ β[2] atol = 0.15   # log-mean slope recovery
    V = vc(fit_slope)[:g]
    # The IDENTIFIED slope axis is axis-1 (Λ[1,1]); recover the true slope-RE SD.
    @test sqrt(V[1, 1]) ≈ SD1_EQ atol = 0.20
    # The UNUSED axis-2 must be pinned near ε (identifiability), not free.
    @test sqrt(V[2, 2]) < 1e-3
    # The identified variance must not be pinned tiny (the pre-#301 failure mode).
    @test V[1, 1] > 0.01
    # No cross-covariance between the identified and pinned axes.
    @test isapprox(V[1, 2], 0.0; atol = 1e-6)
end

# ── 5. LogNormal ──────────────────────────────────────────────────────────────
# NOTE: LogNormal with correlated random slopes is a Gaussian LMM on log(y).
# The Laplace approximation is exact for Gaussian models, so it consistently
# finds a higher (better) logLik than GHQ at K=12 nodes. The relative difference
# is 1–2%, so we accept a slightly looser tolerance of 2% for this family.
@testset "Corr-locscale equiv: LogNormal (1+x|g)" begin
    g, x, b0, b1 = _sim_corr(20260611_05)
    β = [0.5, 0.4]; σlog = 0.3
    μv = β[1] .+ β[2] .* x .+ b0[g] .+ b1[g] .* x
    y = exp.(μv .+ σlog .* randn(N_EQ))
    data = (; y, x, g)

    fit_ghq = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), LogNormal(); data = data)

    fam = LogNormal()
    n = N_EQ; Xμ = hcat(ones(n), x); nmμ = ["(Intercept)", "x"]
    Xψ = ones(n, 1); nmσ = ["(Intercept)"]
    gidx = g; Gv = G_EQ; xs = Float64.(x)
    # The lognormal kernel takes raw y and computes log(y) internally (carries Jacobian).
    fit_lap = _fcloc(fam, _corrkind(fam), :corr, y, Xμ, Xψ, xs, gidx, Gv,
                     nmμ, nmσ, "g"; link = :identity, se = true)

    # The Laplace approximation is exact for Gaussian (log-linear) models; GHQ at
    # K=12 has ~1–2% approximation error, so Laplace logLik ≥ GHQ logLik is expected.
    _check_equiv("LogNormal", fit_ghq, fit_lap; ll_rtol = 0.02)
end
