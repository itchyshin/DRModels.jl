# Regression tests for the location-scale profile inner solve reaching points with
# NaN / stalled gradients (issue: profile CI endpoints returned ±Inf).
#
# Two independent root causes, one testset each (both fixtures are small, ~1 s):
#  1. STIFF PRIOR (log L22 ~ -11.8, Λ⁻¹ entry ~ 1e10): one ulp of the latent mode moves
#     the inner gradient by ~2e-9 > the strict inner tolerance 1e-9 (1+‖a‖), so the inner
#     Newton cycled at the representability floor, reported failure, and the objective /
#     gradient became the 1e18 / NaN sentinels.
#  2. OBJECTIVE-RESOLUTION STALL (interior, log L22 ~ -1.3): backtracking L-BFGS stops
#     ~1e-5 from stationarity because trial objectives differ by less than the ~1e-12
#     noise floor at nll ~ 600; the exact gradient was correct (matches central
#     differences), the solve was simply not finished.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

_nbdraw(η, ψ) = (r = exp(ψ); μ = exp(η);
                 Float64(rand(Distributions.NegativeBinomial(r, r / (r + μ)))))

@testset "profile inner solve: no NaN at a stiff prior (log L22 ~ -11.8)" begin
    Random.seed!(2718)
    G = 20; m = 20; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    LΛ = cholesky(Symmetric([0.25 0.05; 0.05 0.16])).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nbdraw(0.5 + 0.4x[i] + A[species[i]][1], 0.3 + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)
    # Profile trial that (before the fix) had NO certifiable inner mode from a cold start.
    θ = [0.3108427146252975, 0.41913145730470064, -0.1811196914531035,
         -0.894833439806342, 0.00048109338762318234, -11.836382029814112]
    P = DRModels.prior_precision(Q, DRModels._ls_lc_inv2x2(θ[4:6]))
    @test maximum(abs, P) > 1e10                        # the stiff regime is really here
    η0 = Xμ * θ[1:2]; ψ0 = Xψ * θ[3:3]
    # Strict certificate: unreachable here (the ordinary fit path is deliberately unchanged).
    @test !DRModels._ls_inner_mode(Val(:nb2), y, η0, ψ0, species, G, P)[3]
    # Profile solves opt in to the representability-floor certificate.
    a, ch, ok = DRModels._ls_inner_mode(Val(:nb2), y, η0, ψ0, species, G, P; relaxed = true)
    @test ok
    g = DRModels._ls_marginal_grad(Val(:nb2), y, Xμ, Xψ, species, G, Q, θ; relaxed = true)
    @test all(isfinite, g)
    v, _, okv = DRModels._ls_marginal_nll(Val(:nb2), y, η0, ψ0, species, G, P; relaxed = true)
    @test okv && isfinite(v)
    # And through the profile nuisance solve itself.
    r = DRModels._ls_profile_nll_result(Val(:nb2), y, Xμ, Xψ, species, G, Q, θ, 2, θ[2];
                                        x0 = θ[[1, 3, 4, 5, 6]])
    @test isfinite(r.value)
end

@testset "profile nuisance solve: objective-noise stall is finished, not reported as failure" begin
    Random.seed!(11)
    G = 25; m = 12; n = G * m
    species = repeat(1:G, inner = m); x = randn(n)
    LΛ = cholesky(Symmetric([0.6 0.1; 0.1 0.5])).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_nbdraw(0.5 + 0.4x[i] + A[species[i]][1], 0.6 + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)
    θ̂ = [0.6147756391797585, 0.40077215893157714, -0.303169243648461,
          -0.24338148425132908, 0.003313144502233673, -1.272195637264719]
    # Profile trial theta[2] = 0.28298683 warm-started from the neighbouring solution:
    # main returned :not_converged here (L-BFGS+backtracking line-search failure with the
    # exact gradient still 1e-5), which the root finder turned into a ±Inf endpoint.
    x0 = [0.6511114077313405, -0.24956013696583584, -0.2708866014002984,
          -0.023514222800940354, -1.5372729723456355]      # free = [β0, ψ0, log L11, L21, log L22]
    r = DRModels._ls_profile_nll_result(Val(:nb2), y, Xμ, Xψ, species, G, Q, θ̂, 2,
                                        0.28298683132537583; x0 = x0)
    @test r.accepted
    @test r.reason === :accepted
    @test r.gradient_maxabs <= 1e-7
    # The finished solve is the true constrained minimum: it cannot be beaten by the
    # neighbouring accepted solution's value at this θ[2] (598.1508510044...).
    @test r.value ≈ 598.1508510044 atol = 1e-6
end

@testset "profile root finder: unbounded arm stops at the representable range, not in the sentinel region" begin
    # A flat profile below the threshold (gap = -1 everywhere it can be evaluated) whose
    # constrained solve overflows to the 1e18 sentinel past |t| = 700 (exp(-t) overflow).
    evalh(val) = abs(val) > 700 ? (gap = NaN, slope = NaN, ok = false, reason = :not_converged) :
                                  (gap = -1.0, slope = 0.0, ok = true)
    # Uncapped: the geometric expansion walks into the overflow region and the arm is a
    # FAILED endpoint (the reported ±Inf failure).
    bad = DRModels._ls_profile_root_result(evalh, -3.3; dir = -1.0, init = 5.0)
    @test bad.endpoint_failed && !bad.unbounded
    # Capped at |log L| <= 300 (what `_ls_profile_ci_result` does for log-Cholesky
    # diagonals): the honest outcome is "no crossing" (unbounded), not a failure.
    ok = DRModels._ls_profile_root_result(evalh, -3.3; dir = -1.0, init = 5.0,
                                          tmax = DRModels._LS_PROFILE_LOGCHOL_LIMIT + (-3.3))
    @test ok.unbounded && !ok.endpoint_failed
    @test ok.reason === :no_crossing
    @test ok.value == -Inf
end
