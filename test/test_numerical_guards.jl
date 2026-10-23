# test_numerical_guards.jl — regression gates for the numerical-stability guards
# hardened in the twin code-review pass (issues #303, #308, #311, #312, #319, #321
# and low-severity #324.7). Each block reproduces the triggering config (coincident
# coords, SD-collapse boundary, collapsing VA variance, ill-scaled curvature) and
# asserts the guarded path stays finite / errors clearly instead of silently
# returning a poisoned value. (#324.6, dropping the laplace_ll ridge, is NOT applied:
# the ridge is load-bearing for optimiser stability on the q=4 path — see the PR.)

using DRModels
using Test, LinearAlgebra, Random, SparseArrays, Statistics

const _NG = DRModels   # internal kernels live under DRModels.*

@testset "numerical-stability guards" begin

    # ── #308: fz_init_from_Sigma guards SD collapse and near-singular corr ──────
    @testset "#308 fz_init_from_Sigma at SD collapse" begin
        # One among-axis variance collapsed to ~0: log(0) / 0/0 would poison the
        # spherical inversion; the SD floor + ridge loop must return finite φ_a.
        Σcollapse = Matrix(Symmetric([
            0.4 0.05 0.0 0.0
            0.05 0.5 0.0 0.0
            0.0 0.0 1e-10 0.0
            0.0 0.0 0.0 0.3]))
        φ = _NG.fz_init_from_Sigma(Σcollapse)
        @test length(φ) == 10
        @test all(isfinite, φ)

        # Exactly zero variance on an axis (the hard SD-collapse boundary).
        Σzero = Matrix(Symmetric(Diagonal([0.4, 0.5, 0.0, 0.3])))
        @test all(isfinite, _NG.fz_init_from_Sigma(Σzero))

        # A well-conditioned Σ0 must still round-trip (guard does not perturb it).
        Σok = Matrix(Symmetric([
            0.5 0.1 0.0 0.05
            0.1 0.4 0.08 0.0
            0.0 0.08 0.6 0.1
            0.05 0.0 0.1 0.45]))
        φok = _NG.fz_init_from_Sigma(Σok)
        @test all(isfinite, φok)
        @test _NG.fz_DRD(φok) ≈ Σok atol = 1e-6      # reconstructs the SPD input
    end

    # ── #312: spatial range seed guarded against degenerate coordinates ─────────
    @testset "#312 spatial range guard (G=1 / coincident coords)" begin
        # All site coordinates coincide ⇒ every pairwise distance 0 ⇒ log(0)=-Inf
        # seed. Must raise a clear error naming the coordinate problem.
        Random.seed!(312)
        G = 5; coords = zeros(G, 2); m = 4; n = G * m
        site = repeat(1:G, inner = m); x = randn(n)
        y = 0.3 .+ 0.5 .* x .+ 0.4 .* randn(n)
        data = (; y, x, site)
        err = try
            drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1)),
                Gaussian(); data = data, coords = coords)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("coincide", err.msg)

        # Single site (G=1) divides by G^2-G = 0; must error with a ≥2-sites message.
        G1 = 1; coords1 = reshape([1.0, 2.0], 1, 2); n1 = 8
        site1 = fill(1, n1); x1 = randn(n1)
        y1 = 0.3 .+ 0.5 .* x1 .+ 0.4 .* randn(n1)
        data1 = (; y = y1, x = x1, site = site1)
        err1 = try
            drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1)),
                Gaussian(); data = data1, coords = coords1)
            nothing
        catch e
            e
        end
        @test err1 isa ErrorException
        @test occursin("at least 2", err1.msg)
    end

    # ── #319: _finite_hessian is scale-aware and flags a non-usable Hessian ─────
    @testset "#319 _finite_hessian scale-aware step + non-PD flag" begin
        # Sharp (1e6) and flat (1e-4) axes together: a single absolute step cannot
        # resolve both. The per-coordinate step must recover BOTH curvatures, so
        # the SEs (sqrt of diag of the inverse) match the analytic 1e-3 and 100.
        f = x -> 0.5 * (1e6 * x[1]^2 + 1e-4 * (x[2] - 5.0)^2)
        H = _NG._finite_hessian(f, [0.0, 5.0])
        @test isposdef(Symmetric(H))
        se = sqrt.(diag(inv(Symmetric(H))))
        @test se[1] ≈ 1e-3 rtol = 1e-3
        @test se[2] ≈ 100.0 rtol = 1e-3

        # Large-magnitude coordinate: the relative step must still recover unit
        # curvature where a fixed 1e-3 step is comparatively coarse.
        g = x -> 0.5 * (x[1] - 1000.0)^2
        Hg = _NG._finite_hessian(g, [1000.0])
        @test Hg[1, 1] ≈ 1.0 rtol = 1e-4

        # A saddle (indefinite Hessian) must NOT be silently usable: the guard
        # keeps it finite (so the caller's inv does not throw into a fabricated
        # identity) but it is flagged non-PD.
        s = x -> 0.5 * (x[1]^2 - x[2]^2)
        Hs = _NG._finite_hessian(s, [0.0, 0.0])
        @test all(isfinite, Hs)
        @test !isposdef(Symmetric(Hs))
    end

    # ── #321 + #324.7: VA inner Newton stays bounded; KL log is floored ─────────
    @testset "#321 VA inner solve is bounded under strong priors" begin
        # Strong prior (large τ) with few members: the old undamped step /
        # sign-flipped det guard could produce a ~1e15 Newton step and overflow.
        # The damped step must return the correct concave limit (m→0, s→1/τ).
        for τ in (1e2, 1e4, 1e6, 1e8, 1e10, 1e12)
            m, s = _NG._poisson_va_inner(2.0, 1.5, τ)
            @test isfinite(m) && isfinite(s) && s > 0
            # s = 1/(τ + data curvature) ≤ 1/τ; approaches 1/τ as τ → ∞.
            @test s <= (1 + 1e-6) / τ
            @test abs(m) < 1e-1                        # mean shrinks to the prior mean 0
        end
        # Large-τ limit: the posterior variance is dominated by the prior.
        m∞, s∞ = _NG._poisson_va_inner(2.0, 1.5, 1e10)
        @test isapprox(s∞, 1e-10; rtol = 1e-3)
    end

    @testset "#324.7 KL log floored when inner variance collapses" begin
        # A tightly-informed Poisson VA group drives the inner s→0; with the KL
        # log floored the full VA objective stays finite (no -Inf/NaN leak).
        rng = MersenneTwister(324)
        ng, per = 20, 12
        gidx = repeat(1:ng, inner = per)
        n = ng * per
        x = randn(rng, n)
        Xμ = hcat(ones(n), x)
        y = Float64.([rand(rng, 0:6) for _ in 1:n])
        fit = _NG._fit_poisson_ranef_va(_NG.Poisson(), y, Xμ, gidx, ng,
                                        ["(Intercept)", "x"], :g, 1e-8)
        nll = fit.nll
        # Drive logσ_b very negative: the inner posterior variance collapses toward
        # 0, exactly the log(s·τ)→-Inf regime; the floored KL keeps nll finite.
        for lσb in (-6.0, -10.0, -14.0, -18.0)
            θ = vcat([log(mean(y) + eps()), 0.0], lσb)
            @test isfinite(nll(θ))
        end
    end

    # ── #303 + #324.6: PD-prior Cholesky barriers (no poisoned marginal) ────────
    @testset "#303 coevo_marginal returns a finite ℓ for a PD prior" begin
        # Regression: for a well-conditioned Λ the barrier guard leaves the exact
        # Gaussian marginal untouched (finite). The failure branch is a defensive
        # barrier (returns -Inf ⇒ Inf objective) for a non-PD prior factor, which
        # is not naturally reachable from a PD Λ at this scale.
        rng = MersenneTwister(303)
        q = 4; p = 6
        phy = random_balanced_tree(p; branch_length = 0.2)
        Λ = Matrix(Symmetric(0.4I(q) + 0.02 * (ones(q, q) - I(q))))
        σ = fill(0.4, q)
        β = zeros(2, q)
        sim = simulate_coevolution(phy, β, Λ, σ; nrep = 1, rng = rng)
        prob, Qc = make_coevo_problem(phy, sim.Y, sim.X; species = sim.species)
        ℓ, = coevo_marginal(prob, Qc, β, Λ, σ)
        @test isfinite(ℓ)
    end
end
