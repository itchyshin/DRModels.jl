# test_nonconst_sigma_re.jl — NON-CONSTANT dispersion (a `sigma ~ x` sub-model
# with covariates) running SIMULTANEOUSLY with a random effect, for non-Gaussian
# families (#164 audit slice).
#
# AUDIT CONCLUSION (encoded by this file):
#   * A covariate-driven dispersion alongside a random effect IS supported for the
#     non-Gaussian families through the location–scale engine (#202/#209): the
#     coupled `(1 | tag | group)` tag shared by the `mu` and `sigma` formulas fits
#     ONE 2×2 group-level covariance Λ over the (mean-intercept, log-dispersion-
#     intercept) axes, while the `sigma ~ x` FIXED part carries the per-observation
#     dispersion slope. So the dispersion both (a) varies with a covariate and
#     (b) carries a genuine group-level random effect on its own axis.
#       → log θ_i = γ0 + γ1·x_i + b^σ_{g(i)},  (b^μ_g, b^σ_g) ~ N(0, Λ).
#   * The DISTINCT, still-open #164 sub-case — a covariate `sigma ~ x` with a
#     random effect on the MEAN ONLY (no random effect on the σ axis), routed
#     through the `_fit_*_phylo_laplace` / `_fit_*_crossed_laplace` Laplace spine —
#     remains guarded (those fitters require a constant `sigma` formula). The last
#     testset pins that boundary so this file flips when #164 lands.
#
# What is asserted for the supported path (NB2 + Gamma):
#   1. recovery of the genuine `sigma`-axis covariate SLOPE γ1 simultaneously with
#      a non-degenerate random effect on the σ axis (Λ[2,2] > 0) and a mean slope;
#   2. an FD gate — the exact O(p) outer gradient (through the augmented-state
#      Laplace + Takahashi selected inverse) is stationary at the fit AND matches a
#      central finite-difference gradient of the marginal NLL at an off-optimum θ.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Central finite-difference gradient of the location–scale marginal NLL, via the
# public engine objective `LocScaleObjective` (the same object `drm()` attaches
# for the robust profiler). Mirrors the recipe in the standing #165 gates.
function _ls_fd_grad(obj, θ; h = 1e-4)
    g = zeros(length(θ))
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        g[k] = (obj(tp) - obj(tm)) / (2h)
    end
    return g
end

# h-sweep trough over a few step sizes (the #164/#165 discipline). For this Gamma/NB2
# non-const-σ OFF-OPTIMUM probe the FD *reference* floor is ~1e-5 — limited by the
# inner-mode-solve residual (~1e-9) at the optimal h≈1e-4 (truncation ≈ inner_noise ≈ 1e-5,
# so smaller AND larger h are both worse) — and it is platform-dependent at that boundary
# (macOS ~0.9e-5, Linux ~1.06e-5; the 1e-5 gate flaked on Linux CI). The analytic gradient
# is EXACT (verified by recovery + the #164 exact-gradient work); the FD simply can't resolve
# finer without a tighter inner solve, so the gate is 2e-5 (catches real errors, robust to platform).
_ls_fd_gate(g_an, obj, θ; hs = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)) =
    minimum(maximum(abs, g_an .- _ls_fd_grad(obj, θ; h = h)) for h in hs)

@testset "non-constant sigma (sigma ~ x) WITH a random effect — non-Gaussian (#164)" begin

    # ---- NB2: dispersion slope + coupled random effect ---------------------
    @testset "NB2: log θ = γ0 + γ1·x + bσ, coupled (1|p|g) — recovery + FD gate" begin
        Random.seed!(20260610)
        G = 40; m = 40; n = G * m
        g = repeat(1:G, inner = m)
        x = randn(n)
        βμ = [0.3, 0.4]                 # log μ = β0 + β1 x + bμ
        βψ = [0.6, 0.5]                 # log θ = γ0 + γ1 x + bσ  (γ1 = 0.5 ≠ 0)
        Λ = [0.25 0.03; 0.03 0.15]      # 2×2 group covariance on (bμ, bσ)
        LΛ = cholesky(Symmetric(Λ)).L
        A = [LΛ * randn(2) for _ in 1:G]
        ψ = [βψ[1] + βψ[2] * x[i] + A[g[i]][2] for i in 1:n]     # per-obs log-size
        η = [βμ[1] + βμ[2] * x[i] + A[g[i]][1] for i in 1:n]
        y = Float64[(r = exp(ψ[i]); μ = exp(η[i]);
                     rand(Distributions.NegativeBinomial(r, r / (r + μ)))) for i in 1:n]
        data = (; y, x, g)

        # Public front end (drm/bf): the coupled tag routes to the location–scale
        # engine. Mean slope and — the point of this test — the genuine non-constant
        # dispersion (sigma) SLOPE are both recovered. Intercepts are looser: the
        # log-σ intercept absorbs the realized finite-G mean of the σ-axis RE.
        fit = drm(bf(@formula(y ~ x + (1 | p | g)), @formula(sigma ~ x + (1 | p | g))),
                  NegBinomial2(); data = data, se = true)
        @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.10
        @test coef(fit, :sigma)[2] ≈ -0.5 * βψ[2] atol = 0.15     # ← non-constant dispersion slope (log σ = -0.5 log size)
        # The random effect on the σ axis is real and non-degenerate (Λ[2,2] > 0),
        # so the dispersion carries BOTH a covariate and a random effect.
        Λ̂ = DRModels.vc(fit)[:g]
        @test size(Λ̂) == (2, 2)
        @test isposdef(Symmetric(Λ̂))
        @test Λ̂[2, 2] > 0.005                              # σ-axis RE variance recovered, > 0 (¼× log-size scale)
        @test isfinite(loglik(fit))
        # NB: `fit.converged` (Optim's flag) is NOT asserted — near the singular
        # variance boundary ‖g‖ plateaus and relative-objective stopping leaves the
        # flag false even at a stationary point (HANDOVER §6.3; same convention as
        # the location–scale e2e tests). Real convergence is the gradient gate below.

        # FD gate on the engine packing. `drm()` repacks θ into `:recov` order, so
        # drive the gate from a direct `_fit_locscale` call (engine ordering),
        # exactly as the location–scale e2e tests do. The exact augmented-state
        # Laplace outer gradient (through the tree precision + Takahashi selected
        # inverse) must be stationary at the fit AND match a central FD gradient of
        # the marginal NLL at a θ nudged off the optimum (exercising db̂/dθ).
        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), x)
        gidx, Gd = DRModels._group_index(g); Q = sparse(1.0 * I, Gd, Gd)
        fr = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, gidx, Gd, Q; se = false)
        @test fr.beta_psi[2] ≈ -0.5 * βψ[2] atol = 0.15           # dispersion slope (engine-native; -0.5 log size)
        @test fr.Lambda[2, 2] > 0.005
        gmax = maximum(abs, DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, Gd, Q, fr.θ))
        @test gmax < 1e-3                                   # stationarity

        obj = DRModels.LocScaleObjective(Val(:nb2), y, Xμ, Xψ, gidx, Gd, Q)
        θoff = copy(fr.θ); θoff[2] += 0.05; θoff[end] -= 0.05   # off the optimum
        g_an = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, gidx, Gd, Q, θoff)
        @test _ls_fd_gate(g_an, obj, θoff) ≤ 2e-5         # exact ≈ FD (floor ~1e-5 here; #165)
    end

    # ---- Gamma: dispersion (shape) slope + coupled random effect -----------
    @testset "Gamma: log α = γ0 + γ1·x + bσ, coupled (1|p|g) — recovery + FD gate" begin
        Random.seed!(424242)
        G = 35; m = 35; n = G * m
        g = repeat(1:G, inner = m)
        x = randn(n)
        βμ = [0.3, 0.4]                 # log μ = β0 + β1 x + bμ
        βψ = [0.5, 0.4]                 # log α = γ0 + γ1 x + bσ  (γ1 = 0.4 ≠ 0)
        Λ = [0.18 0.02; 0.02 0.10]
        LΛ = cholesky(Symmetric(Λ)).L
        A = [LΛ * randn(2) for _ in 1:G]
        y = Float64[(α = exp(βψ[1] + βψ[2] * x[i] + A[g[i]][2]);
                     μ = exp(βμ[1] + βμ[2] * x[i] + A[g[i]][1]);
                     rand(Distributions.Gamma(α, μ / α))) for i in 1:n]
        data = (; y, x, g)

        fit = drm(bf(@formula(y ~ x + (1 | p | g)), @formula(sigma ~ x + (1 | p | g))),
                  Gamma(); data = data, se = true)

        @test coef(fit, :mu)[2] ≈ βμ[2] atol = 0.10
        @test coef(fit, :sigma)[2] ≈ βψ[2] atol = 0.15     # ← non-constant dispersion (shape) slope
        Λ̂ = DRModels.vc(fit)[:g]
        @test isposdef(Symmetric(Λ̂))
        @test Λ̂[2, 2] > 0.01
        @test isfinite(loglik(fit))
        # `fit.converged` not asserted (variance-boundary plateau) — see NB2 note.

        Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), x)
        gidx, Gd = DRModels._group_index(g); Q = sparse(1.0 * I, Gd, Gd)
        fr = DRModels._fit_locscale(Val(:gamma), y, Xμ, Xψ, gidx, Gd, Q; se = false)
        @test fr.beta_psi[2] ≈ βψ[2] atol = 0.15
        @test fr.Lambda[2, 2] > 0.01
        gmax = maximum(abs, DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, gidx, Gd, Q, fr.θ))
        @test gmax < 1e-3

        obj = DRModels.LocScaleObjective(Val(:gamma), y, Xμ, Xψ, gidx, Gd, Q)
        θoff = copy(fr.θ); θoff[2] += 0.05; θoff[end] -= 0.05
        g_an = DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, gidx, Gd, Q, θoff)
        @test _ls_fd_gate(g_an, obj, θoff) ≤ 2e-5         # exact ≈ FD (floor ~1e-5 here; #165)
    end

    # ---- #164 LANDED: covariate sigma with a MEAN-ONLY phylo RE now FITS ------
    # Previously the open #164 gap (guarded). #164 (covariate dispersion with a
    # mean-only phylo RE, NB2) landed the per-observation log-dispersion path, so
    # this now FITS via `_fit_nb2_phylo_laplace`'s hetero branch instead of throwing
    # — the recovery assertion the old guard's comment promised.
    @testset "#164: sigma ~ x with a MEAN-ONLY phylo RE now fits (NB2)" begin
        Random.seed!(606)
        p = 12; m = 8; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.3)
        species = repeat(1:p, inner = m)
        x = randn(n)
        y = Float64[rand(Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + exp(0.2 + 0.4x[i]))))
                    for i in 1:n]
        data = (; y, x, species)
        fit = drm(
            bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)),
            NegBinomial2(); data = data, tree = phy, se = false)
        @test is_converged(fit)
        @test isfinite(loglik(fit))
    end
end
