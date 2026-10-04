# End-to-end fit for the non-Gaussian location–scale model (#202 groundwork).
# SMOKE level: verifies the fitter runs end to end on NB2 data with a shared
# random effect on BOTH the mean and the log-dispersion axis — for i.i.d. groups
# and for a phylogenetic tree — and returns sane output (finite marginal, a valid
# 2×2 covariance, a sensible mean slope from the well-identified mean axis).
#
# Why smoke and not recovery: the outer optimiser here uses a finite-difference
# gradient of the (inner-solved) Laplace marginal, which is too noisy/slow for
# trustworthy variance-component recovery. Tight recovery waits on the exact O(p)
# outer gradient (Takahashi) slice; marginal accuracy is already pinned by the
# marginal-vs-Gauss–Hermite gate.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_nb2_draw(η, ψ) = (r = exp(ψ); μ = exp(η);
                   Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))

@testset "location–scale fit: i.i.d. groups (NB2) end-to-end smoke" begin
    Random.seed!(20260606)
    p = 6; m = 10; n = p * m
    Λtrue = [0.25 0.05; 0.05 0.16]
    LΛ = cholesky(Symmetric(Λtrue)).L
    A = randn(p, 2) * LΛ'
    species = repeat(1:p, inner = m)
    x = randn(n)
    βμ = [0.2, 0.4]; βψ = [0.3]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nb2_draw(βμ[1] + βμ[2] * x[i] + A[species[i], 1],
                   βψ[1] + A[species[i], 2]) for i in 1:n]

    Q = sparse(1.0 * I, p, p)
    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, species, p, Q)

    @test fit.nll < 1e17                            # feasible fit (the guard worked)
    # Valid covariance: a collapsed (boundary) variance is a legitimate tiny-data
    # ML outcome, so we check finiteness + non-negative diagonal rather than strict
    # PD — strict PD is asserted where the variances are well identified (recovery,
    # inference, gamma/phylo e2e). Keeps the smoke test robust to RNG/dep drift.
    @test all(isfinite, fit.Lambda) && fit.Lambda[1, 1] ≥ 0 && fit.Lambda[2, 2] ≥ 0
    @test size(fit.Lambda) == (2, 2)
    @test all(isfinite, fit.beta_mu) && isfinite(fit.beta_psi[1])
    @test fit.converged isa Bool
end

@testset "location–scale fit: phylogenetic tree (NB2) end-to-end smoke" begin
    Random.seed!(20260607)
    p = 6; m = 8; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.25)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC = cholesky(Symmetric(C)).L
    Λtrue = [0.30 0.0; 0.0 0.20]
    LΛ = cholesky(Symmetric(Λtrue)).L
    A = LC * randn(p, 2) * LΛ'
    species = repeat(1:p, inner = m)
    x = randn(n)
    βμ = [0.15, 0.4]; βψ = [0.2]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nb2_draw(βμ[1] + βμ[2] * x[i] + A[species[i], 1],
                   βψ[1] + A[species[i], 2]) for i in 1:n]

    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, gidx, G, Q)

    @test fit.nll < 1e17                            # phylo precision path runs end-to-end (feasible)
    @test all(isfinite, fit.Lambda) && fit.Lambda[1, 1] ≥ 0 && fit.Lambda[2, 2] ≥ 0
    @test all(isfinite, fit.beta_mu) && isfinite(fit.beta_psi[1])
end

# With the exact gradient now driving an LBFGS fit, we can assert real
# convergence and parameter recovery (deferred at the smoke stage). The
# stationarity check is seed-robust: the optimiser must have driven the EXACT
# analytic gradient to ~0. Recovery tolerances are generous (single seed).
@testset "location–scale fit: gradient-based convergence + recovery (NB2)" begin
    Random.seed!(424242)
    G = 50; m = 35; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    βμ = [0.5, 0.4]; βψ = [0.3]
    Λtrue = [0.25 0.05; 0.05 0.16]                  # sd_μ = 0.5, sd_ψ = 0.4
    LΛ = cholesky(Symmetric(Λtrue)).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nb2_draw(βμ[1] + βμ[2] * x[i] + A[species[i]][1],
                   βψ[1] + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)
    fit = DRModels._fit_locscale(Val(:nb2), y, Xμ, Xψ, species, G, Q)

    gmax = maximum(abs.(DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, species, G, Q, fit.θ)))
    @test gmax < 1e-3                               # stationarity of the exact gradient (convergence evidence)
    @test fit.beta_mu[1] ≈ 0.5 atol = 0.2
    @test fit.beta_mu[2] ≈ 0.4 atol = 0.1
    @test fit.beta_psi[1] ≈ -0.15 atol = 0.25       # -0.5 * 0.3 (log σ = -0.5 log size)
    @test sqrt(fit.Lambda[1, 1]) ≈ 0.5 rtol = 0.3   # mean-axis RE SD
    @test sqrt(fit.Lambda[2, 2]) ≈ 0.2 rtol = 0.45  # scale-axis RE SD (harder); 0.5 * 0.4
end

# Certified-refinement stall stop. The refinement runs disable Optim's x/f
# tolerances, so before this callback a run that had stopped moving used its
# whole iteration budget (measured: ~1,950 BFGS iterations at a bit-identical
# objective on the Gamma phylo bootstrap fixture). Iteration counts, not wall
# clock, are asserted.
@testset "location–scale refinement stall callback" begin
    window = DRModels._LS_REFINE_STUCK_WINDOW
    st(v, g, x) = (value = v, g_norm = g, metadata = Dict{String,Any}("x" => x))
    # Bit-identical x, value and gradient norm: stop after `window` repeats.
    stop = DRModels._ls_refine_stall_callback()
    @test findfirst(_ -> stop(st(43.5, 3e-8, [1.0, 2.0])), 1:2_000) == window + 1
    # Any movement resets the count, even when value and gradient are frozen
    # (an ill-conditioned problem at a large offset), and so does a value change.
    moving = DRModels._ls_refine_stall_callback()
    @test !any(k -> moving(st(1e10, 1.0, [1.0 + k * eps(1.0), 2.0])), 1:2_000)
    descending = DRModels._ls_refine_stall_callback()
    @test !any(k -> descending(st(50.0 - 1e-15k, 1e-3, [1.0, 2.0])), 1:2_000)
    # Without `x` in the state the callback never stops a run.
    blind = DRModels._ls_refine_stall_callback()
    @test !any(_ -> blind((value = 1.0, g_norm = 1.0, metadata = Dict{String,Any}())), 1:2_000)

    # The options actually used by `_fit_locscale` carry `x` and the callback.
    Optim = DRModels.Optim
    A = Diagonal(exp.(range(log(10.0), log(1e6); length = 40)))
    fq(x) = 1e10 + 0.5 * dot(x, A * x)
    gq!(G, x) = (G .= A * x; G)
    x0 = fill(1e-3, 40)
    plain = Optim.Options(g_tol = 1e-6, iterations = 1_000, x_abstol = NaN, x_reltol = NaN,
                          f_abstol = NaN, f_reltol = NaN)
    for method in (Optim.LBFGS(), Optim.BFGS(linesearch = Optim.LineSearches.BackTracking()))
        r0 = Optim.optimize(fq, gq!, copy(x0), method, plain)
        r1 = Optim.optimize(fq, gq!, copy(x0), method, DRModels._ls_refine_options(1e-6, 1_000))
        @test Optim.iterations(r1) == Optim.iterations(r0)
        @test Optim.converged(r1) == Optim.converged(r0)
    end
    # A run whose steps are below one ULP of x never moves: stopped after the
    # window instead of spending all 2,000 iterations.
    frozen = Optim.BFGS(alphaguess = Optim.LineSearches.InitialStatic(alpha = 1e-300),
                        linesearch = Optim.LineSearches.Static())
    r = Optim.optimize(fq, gq!, copy(x0), frozen, DRModels._ls_refine_options(1e-8, 2_000))
    @test !Optim.converged(r)
    @test Optim.iterations(r) <= window + 2
end
