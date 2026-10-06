# test_poisson_phylo_grad_gate.jl — STANDING FD-vs-analytic gradient gate (#165)
# for the non-Gaussian (Poisson) phylogenetic sparse-Laplace route.
#
# Mirrors test_qgate_fd_gradient.jl (the verified q=4 Q-gate) for the
# Poisson `phylo(1 | grp)` Laplace marginal: the analytic outer gradient
# (`_poisson_phylo_laplace_fg`, the exact implicit-function / implicit-logdet
# gradient that reuses `takahashi_selinv`) MUST match a central finite-difference
# gradient of the TRUE marginal NLL to ≤ 1e-6.
#
# The recipe that makes 1e-6 attainable (same as the q4 gate): drive the inner
# Newton mode to a TIGHT tolerance and warm-start every perturbed solve from the
# base-θ mode, so the finite-difference reference is not dominated by inner-mode
# stopping noise. The gradient is evaluated at a θ OFF the optimum so the
# implicit (db̂/dθ) correction terms are nonzero — a frozen-mode gradient would
# fail this gate.

using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "Poisson phylo Laplace gradient gate (#165): FD-vs-exact ≤ 1e-6" begin
    Random.seed!(165)
    p = 12
    m = 4                                  # nrep ≥ 2 (scale-RE identifiability)
    phy = random_balanced_tree(p; branch_length = 0.20)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    βt = [0.20, 0.35]
    σphy = 0.45
    C = sigma_phy_dense(phy; σ²_phy = σphy^2)
    u = cholesky(Symmetric(C)).L * randn(p)
    λ = exp.(βt[1] .+ βt[2] .* x .+ u[species])
    y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])

    Xμ = hcat(ones(n), x)
    Q, leaf_node, _ = DRModels._poisson_phylo_setup(phy, species)
    q = size(Q, 1)
    qchol = cholesky(Symmetric(Q); check = false)
    logdetQ = logdet(qchol)
    lf = [DRModels._logfactorial(round(Int, yi)) for yi in y]

    # θ = [βμ(2); logσ], deliberately OFF the optimum.
    θ = [0.10, 0.45, log(0.55)]

    # Tightly-converged base mode, reused as the warm start for every solve so
    # the FD reference sees the same inner mode the analytic gradient froze.
    #
    # Tolerance note (the recipe that makes 1e-6 attainable, mirroring the q4
    # Q-gate's estep_mode default `tol = 1e-8`): the inner Newton mode uses a
    # step-norm stop, `norm(step) <= ntol * (1 + norm(b))`, followed by a
    # backtracking line search that requires a STRICT joint decrease. When a
    # perturbed solve is warm-started *exactly at the mode*, `step ≈ 0` and the
    # line search cannot find a further decrease, so an over-tight `ntol`
    # (≈ machine-eps) skips the step-norm stop and then fails the line search,
    # returning the inner solve's `1e18` infeasibility sentinel. Differencing
    # that sentinel — not any analytic-gradient error — is what blew the central
    # difference up to ~5e22. A tight-but-sane `ntol = 1e-10` (well above the
    # ~1e-12 roundoff floor of the warm-started step) lets the step-norm stop
    # fire first, so every probed point returns the genuine marginal NLL. The
    # mode is still converged far tighter than the marginal needs: the NLL is
    # first-order flat at b̂, so a 1e-10 mode error perturbs it by O(1e-20).
    ntol = 1e-10
    nmax = 400
    val0, g_an, b_base, ok = DRModels._poisson_phylo_laplace_fg(
        y, Xμ, leaf_node, Q, logdetQ, lf, θ;
        grad = true, b0 = zeros(q), newton_tol = ntol, newton_maxiter = nmax,
    )
    @test ok
    @test isfinite(val0)
    @test val0 < 1e17                      # genuine marginal, never the sentinel

    function mnll(t)
        v = DRModels._poisson_phylo_laplace_fg(
            y, Xμ, leaf_node, Q, logdetQ, lf, Vector{Float64}(t);
            grad = false, b0 = copy(b_base), newton_tol = ntol, newton_maxiter = nmax,
        )[1]
        # Guard the FD reference: if any probed point ever returns the inner
        # solve's 1e18 sentinel, surface it as a clear failure here rather than
        # as ~1e22 finite-difference noise downstream.
        @assert isfinite(v) && v < 1e17 "marginal NLL infeasible (sentinel) at probed θ = $t"
        return v
    end

    # h = 1e-4 matches the verified q4 Q-gate (test_qgate_fd_gradient.jl): large
    # enough that central-difference truncation error O(h²) ≈ 1e-8 stays under the
    # 1e-6 gate, small enough that the marginal is locally quadratic.
    h = 1e-4
    g_fd = similar(g_an)
    for k in eachindex(θ)
        tp = copy(θ); tp[k] += h
        tm = copy(θ); tm[k] -= h
        g_fd[k] = (mnll(tp) - mnll(tm)) / (2h)
    end

    max_abs_diff = maximum(abs, g_an .- g_fd)
    @info "Poisson phylo gradient gate" max_abs_diff g_an g_fd
    @test max_abs_diff ≤ 1e-6
end
