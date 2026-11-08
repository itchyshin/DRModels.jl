# test_crossed_tweedie.jl — #737 twin gap: Tweedie() refuses crossed/multiple
# random effects `(1 | g) + (1 | h)` on `mu`; drmTMB's `tweedie()` fits them.
# Integrator: the shared sparse augmented-state Laplace GLMM engine
# (`src/sparse_laplace_glmm.jl`, the same crossed-mean machinery Gamma/NB2/Beta
# already reuse via `_fit_crossed_mean_laplace_nuisance`), NOT a new
# integrator. `sigma` (dispersion) and `nu` (power p) stay constant
# fixed-effect sub-models (`sigma ~ 1`, `nu ~ 1`), matching the constant-`sigma`
# restriction already imposed on Gamma's crossed route. The dispersion is
# folded into the shared engine's single scalar "nuisance" slot; the power `p`
# is profiled out by an outer 1-D Brent search (exact profile-likelihood
# optimisation over the one dimension the shared engine has no slot for — not
# a new integrator for the random effects themselves). Per-observation
# value/derivatives are obtained from `_logpdf_tweedie` via nested ForwardDiff
# rather than a hand-derived closed form (the Dunn–Smyth series has none),
# checked against central finite differences in-line in `src/tweedie.jl`.
using DRModels
using Test, Random
import Distributions

# simulate a Tweedie variate as a compound Poisson–Gamma (same helper as
# test_tweedie.jl / test_tweedie_ranef.jl)
function _rtweedie(μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
    N = rand(Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
end

@testset "#737 Tweedie crossed random effects (1|g)+(1|h)" begin

    # Pre-fix red gate (manually confirmed against origin/main before this
    # branch's implementation commit): `drm(bf(@formula(y ~ x + (1 | g) + (1 | h))),
    # Tweedie(); data = dat)` raised `"Tweedie() supports only a single `(1 | g)`
    # random intercept or `(0 + x | g)` random slope on `mu`; crossed/multiple
    # random effects are not implemented"`.

    @testset "known-DGP recovery: crossed random intercepts on the log-mean" begin
        Random.seed!(20325905)   # the ADEMP cell 65 seed (issue #737)
        G = 12; H = 10; n = 220
        g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
        β = [0.3, 0.4]; φ = 0.6; p = 1.6; σg = 0.35; σh = 0.30
        bg = σg .* randn(G); bh = σh .* randn(H)
        μ = exp.(β[1] .+ β[2] .* x .+ bg[g] .+ bh[h])
        y = [_rtweedie(μi, φ, p) for μi in μ]
        dat = (; y, x, g, h)

        t_fit = @elapsed fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h))), Tweedie(); data = dat)

        @test fit.converged
        # Speed regression guard. Before the capped first LBFGS step in
        # `_fit_tweedie_crossed_laplace`, every profile-p inner fit wasted its first
        # line-search probes at wild parameters (Dunn-Smyth window ~1e5 terms) and
        # this single fit took over an hour on a loaded machine (4428 s as one test
        # file); it now takes tens of seconds. The budget is ~30x the measured time
        # (includes first-call compilation), so only a return of the blow-up trips it.
        @test t_fit < 900
        # The optimum itself is pinned (profile loglik is flat in p; 1e-6 is far
        # tighter than the atol checks below and ~1e4x the observed run-to-run noise).
        @test loglik(fit) ≈ -304.3129220 atol = 1e-6
        @test coef(fit, :mu)[1] ≈ β[1] atol = 0.20
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.20
        @test exp(2 * coef(fit, :sigma)[1]) ≈ φ atol = 0.35
        p̂ = 1 + 1 / (1 + exp(-coef(fit, :nu)[1]))
        @test p̂ ≈ p atol = 0.25
        rs = re_sd(fit)
        @test rs[:g] ≈ σg atol = 0.20
        @test rs[:h] ≈ σh atol = 0.20
        @test isfinite(loglik(fit))
    end

    @testset "refuses a non-constant sigma/nu formula alongside crossed REs" begin
        Random.seed!(20260929)
        G = 8; H = 6; n = 100
        g = rand(1:G, n); h = rand(1:H, n); x = randn(n)
        y = abs.(randn(n)) .+ 0.1
        dat = (; y, x, g, h)
        err = nothing
        try
            drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ x)), Tweedie(); data = dat)
        catch e
            err = e
        end
        @test err !== nothing
    end
end
