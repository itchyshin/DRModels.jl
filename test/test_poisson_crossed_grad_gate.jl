# test_poisson_crossed_grad_gate.jl — STANDING FD-vs-analytic gradient gate (#165)
# for the Poisson CROSSED-random-intercepts sparse-Laplace route.
#
# Companion to test_poisson_phylo_grad_gate.jl. The crossed-Poisson inner joint
# is strictly convex (convex Poisson data term + diagonal Gaussian prior), so the
# same full-Newton-in-basin fix applies to `_poisson_crossed_mode`: a backtracking
# line search STALLS on rounding-level decreases near the mode and would leave it
# only loosely converged (~1e-6), polluting the marginal FD gradient. With the
# tight inner mode the analytic outer gradient (`_poisson_crossed_laplace_fg`,
# exact implicit-function / implicit-logdet gradient that reuses
# `takahashi_selinv` / `_crossed_selected_inverse_entries`) MUST match a central
# finite-difference gradient of the TRUE marginal NLL to ≤ 1e-6.
#
# Recipe (same as the q4 / phylo gates): drive the inner Newton mode to a tight
# tolerance, warm-start every perturbed solve from the base-θ mode, and evaluate
# the gradient at a θ OFF the optimum so the implicit db̂/dθ terms are exercised.

using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Poisson crossed Laplace gradient gate (#165): FD-vs-exact ≤ 1e-6" begin
    Random.seed!(1652)
    G = 6                                   # levels of factor g
    Hh = 5                                  # levels of factor h
    reps = 4
    # A fully-crossed design: every (g, h) cell observed `reps` times.
    gidx = Int[]; hidx = Int[]
    for gi in 1:G, hj in 1:Hh, _ in 1:reps
        push!(gidx, gi); push!(hidx, hj)
    end
    n = length(gidx)
    x = randn(n)
    βt = [0.20, 0.35]
    σg = 0.40; σh = 0.30
    ug = σg .* randn(G)
    uh = σh .* randn(Hh)
    λ = exp.(βt[1] .+ βt[2] .* x .+ ug[gidx] .+ uh[hidx])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])

    Xμ = hcat(ones(n), x)
    lf = [DRModels._logfactorial(round(Int, yi)) for yi in y]

    # θ = [βμ(2); logσg; logσh], deliberately OFF the optimum.
    θ = [0.10, 0.45, log(0.55), log(0.25)]

    ntol = 1e-10                            # tight but above the warm-started roundoff floor
    nmax = 400
    val0, g_an, b_base, ok = DRModels._poisson_crossed_laplace_fg(
        y, Xμ, gidx, G, hidx, Hh, lf, θ;
        grad = true, b0 = zeros(G + Hh), newton_tol = ntol, newton_maxiter = nmax,
    )
    @test ok
    @test isfinite(val0)
    @test val0 < 1e17                       # genuine marginal, never the sentinel

    function mnll(t)
        v = DRModels._poisson_crossed_laplace_fg(
            y, Xμ, gidx, G, hidx, Hh, lf, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = ntol, newton_maxiter = nmax,
        )[1]
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at probed θ = $t"
        return v
    end

    h = 1e-4
    g_fd = similar(g_an)
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        g_fd[k] = (mnll(tp) - mnll(tm)) / (2h)
    end

    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Poisson crossed gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end
