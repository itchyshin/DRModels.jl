# Unit tests for the private plumbing in src/locscale_simulate.jl (the coupled
# non-Gaussian location--scale parametric bootstrap): the family/response/
# design comparison helpers, the `_ls_bootstrap_prepared_state` refusals, and
# the draws of `_ls_marginal_simulator`.  The end-to-end contract (permutation,
# covariance, delegated inverse links) lives in
# test_locscale_bootstrap_simulator.jl; this file pins the pieces it skips.
# Fits are no-fit, public-shaped DrmFits built with the same constructors as
# that file, so the sampler is isolated from optimisation.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays

# Message of the ArgumentError thrown by `f`, or the other outcome if none.
function _ls_helpers_refusal(f)
    try
        f()
        return nothing
    catch e
        return e isa ArgumentError ? e.msg : e
    end
end

# i.i.d. coupled (mu, sigma) fixture, G = 4 groups, n = 24 rows.  `recov` is
# the engine tail [logL11, L21, logL22].
function _ls_helpers_fixture(kind::Symbol; recov = [log(0.55), 0.31, log(0.42)])
    G = 4
    g = [40, 10, 30, 20, 10, 40, 20, 30,
         30, 20, 40, 10, 20, 30, 10, 40,
         40, 20, 30, 10, 10, 30, 20, 40]
    n = length(g)
    x = repeat([-0.8, -0.3, 0.2, 0.7], 6)
    Xmu = hcat(ones(n), x)
    Xpsi = hcat(ones(n), x)
    if kind === :gamma
        y = [1.2 + 0.05 * i for i in 1:n]
        family = Gamma()
    else
        y = [0.15 + 0.7 * ((i - 1) % 5) / 4 for i in 1:n]
        family = Beta()
    end
    formula = bf(@formula(y ~ x + (1 | pair | g)),
                 @formula(sigma ~ x + (1 | pair | g)))
    data = (; y, x, g)
    theta = [0.25, 0.35, -0.15, 0.20, recov...]
    fitres = (; θ = theta, beta_mu = theta[1:2], beta_psi = theta[3:4],
              vcov = Matrix{Float64}(I, length(theta), length(theta)),
              nll = 0.0, converged = true)
    base = DRModels._build_locscale_drmfit(Val(kind), family, fitres, y, Xmu, Xpsi,
        ["(Intercept)", "x"], ["(Intercept)", "x"], "g";
        obs_prop = nothing, trials = nothing)
    gidx, Gfound = DRModels._group_index(g)
    @assert Gfound == G
    objective = DRModels.LocScaleObjective(Val(kind), y, Xmu, Xpsi, gidx, G,
        sparse(1.0I, G, G); whitened = true)
    fit = DRModels._withformula(DRModels._withnll(base, objective), formula)
    return (; kind, fit, data, family, Xmu, Xpsi, theta, base, objective, formula)
end

@testset "locscale_simulate helpers" begin
    @testset "_ls_bootstrap_family_matches truth table" begin
        fm = DRModels._ls_bootstrap_family_matches
        kinds = (Val(:gamma), Val(:nb2), Val(:beta), Val(:betabinomial))
        fams = (Gamma(), NegBinomial2(), Beta(), BetaBinomial())
        for (i, k) in enumerate(kinds), (j, f) in enumerate(fams)
            @test fm(k, f) == (i == j)
        end
        # Families outside the four coupled leaves never match any kind.
        for k in kinds
            @test !fm(k, Poisson())
            @test !fm(k, Gaussian())
        end
        # An unknown kind matches nothing.
        @test !fm(Val(:poisson), Gamma())
        @test !fm(:gamma, Gamma())
    end

    @testset "_ls_bootstrap_same_response / same_matrix: isequal semantics" begin
        sr = DRModels._ls_bootstrap_same_response
        sm = DRModels._ls_bootstrap_same_matrix
        a = [1.0, 2.5, NaN, 4.0]
        @test sr(a, copy(a))
        @test sr(Float64[], Float64[])
        @test !sr(a, a[1:3])                       # different length
        @test !sr([1.0, 2.0], [1.0, 2.0, 3.0])
        b = copy(a); b[2] = 2.5000001
        @test !sr(a, b)                            # one differing entry
        c = copy(a); c[3] = 0.0
        @test !sr(a, c)                            # NaN is not a number
        # isequal, not ==: NaN equals NaN, -0.0 differs from 0.0.
        @test sr([NaN], [NaN])
        @test !(NaN == NaN)
        @test !sr([-0.0], [0.0])
        @test sr([0.0], [0.0])
        @test sr([-0.0], [-0.0])

        M = [1.0 2.0; 3.0 NaN; 5.0 6.0]
        @test sm(M, copy(M))
        @test !sm(M, M')                           # size mismatch
        @test !sm(M, M[1:2, :])
        @test !sm(M, hcat(M, zeros(3)))
        N = copy(M); N[3, 1] = 5.5
        @test !sm(M, N)
        N2 = copy(M); N2[2, 2] = 0.0
        @test !sm(M, N2)
        @test sm([NaN 1.0], [NaN 1.0])
        @test !sm([-0.0 1.0], [0.0 1.0])
        @test sm([-0.0 1.0], [-0.0 1.0])
        @test sm(zeros(0, 2), zeros(0, 2))
    end

    @testset "_ls_bootstrap_same_sparse" begin
        ss = DRModels._ls_bootstrap_same_sparse
        Q = sparse([4.0 1.0 0.0; 1.0 3.0 0.0; 0.0 0.0 2.0])
        @test ss(Q, copy(Q))
        @test ss(sparse(1.0I, 3, 3), sparse(1.0I, 3, 3))
        @test !ss(Q, sparse(1.0I, 4, 4))           # size mismatch
        @test !ss(Q, Q[1:2, 1:2])
        Q2 = copy(Q); Q2[1, 2] = 1.5
        @test !ss(Q, Q2)                           # value mismatch
        Q3 = copy(Q); Q3[1, 3] = 0.25              # extra structural entry
        @test !ss(Q, Q3)
    end

    @testset "_ls_bootstrap_prepared_state refusals" begin
        fx = _ls_helpers_fixture(:gamma)
        prep = DRModels._ls_bootstrap_prepared_state
        # Sanity: the unmodified fixture is accepted.
        @test prep(fx.fit, fx.data) isa DRModels._LSMarginalBootstrapState

        # (a) fit without a LocScaleObjective: a real ordinary fit, and a
        # no-fit twin with the objective stripped.
        n = 60
        rng = MersenneTwister(11)
        x = randn(rng, n)
        yg = exp.(0.3 .+ 0.2 .* x) .* (0.5 .+ rand(rng, n))
        plain = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gamma(); data = (; y = yg, x))
        @test !(plain.nll isa DRModels.LocScaleObjective)
        @test _ls_helpers_refusal(() -> prep(plain, (; y = yg, x))) ==
              "coupled location-scale bootstrap requires a LocScaleObjective"
        yp = Float64.(rand(rng, 0:6, n))
        plain_nb = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), NegBinomial2(); data = (; y = yp, x))
        @test _ls_helpers_refusal(() -> prep(plain_nb, (; y = yp, x))) ==
              "coupled location-scale bootstrap requires a LocScaleObjective"
        yb = clamp.(0.5 .+ 0.2 .* x .+ 0.1 .* randn(rng, n), 0.05, 0.95)
        plain_beta = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Beta(); data = (; y = yb, x))
        @test _ls_helpers_refusal(() -> prep(plain_beta, (; y = yb, x))) ==
              "coupled location-scale bootstrap requires a LocScaleObjective"
        stripped = DRModels._withnll(fx.fit, nothing)
        @test _ls_helpers_refusal(() -> prep(stripped, fx.data)) ==
              "coupled location-scale bootstrap requires a LocScaleObjective"

        # (b) fit without the original DrmFormula.
        noform = DRModels._withformula(fx.fit, nothing)
        @test _ls_helpers_refusal(() -> prep(noform, fx.data)) ==
              "coupled location-scale bootstrap requires the original formula"

        # (c) family that does not match the stored objective's kind.
        for fam in (Beta(), Poisson(), NegBinomial2())
            wrongfam = DRModels._withnll(
                DRModels._build_locscale_drmfit(Val(:gamma), fam,
                    (; θ = fx.theta, beta_mu = fx.theta[1:2], beta_psi = fx.theta[3:4],
                       vcov = Matrix{Float64}(I, 7, 7), nll = 0.0, converged = true),
                    fx.data.y, fx.Xmu, fx.Xpsi, ["(Intercept)", "x"], ["(Intercept)", "x"], "g";
                    obs_prop = nothing, trials = nothing),
                fx.objective)
            wrongfam = DRModels._withformula(wrongfam, fx.formula)
            @test _ls_helpers_refusal(() -> prep(wrongfam, fx.data)) ==
                  "coupled location-scale bootstrap family does not match its stored objective"
        end

        # (d) formula lacking mu, or lacking sigma.
        f = fx.fit.formula
        for drop in (:mu, :sigma)
            kept = Pair{Symbol,Any}[p for p in f.forms if first(p) != drop]
            @test length(kept) == length(f.forms) - 1
            partial = DRModels.DrmFormula(f.response, kept, f.response2)
            fit = DRModels._withformula(fx.fit, partial)
            @test _ls_helpers_refusal(() -> prep(fit, fx.data)) ==
                  "coupled location-scale bootstrap requires mu and sigma formulas"
        end
    end

    @testset "_ls_marginal_simulator draws" begin
        N = 4000
        for kind in (:gamma, :beta)
            # (i) support, length, reproducibility at the regular fixture.
            fx = _ls_helpers_fixture(kind)
            n = length(fx.data.y)
            sim = DRModels._ls_marginal_simulator(fx.fit, fx.data)
            @test sim isa Function
            d1 = sim(MersenneTwister(2026))
            d2 = sim(MersenneTwister(2026))
            d3 = sim(MersenneTwister(2027))
            @test length(d1) == n
            @test eltype(d1) <: Real
            @test all(isfinite, d1)
            @test d1 == d2                       # fixed rng => identical draw
            @test d1 != d3                       # different rng => different draw
            for s in 1:50
                d = sim(MersenneTwister(s))
                if kind === :gamma
                    @test all(>(0), d)
                else
                    # Draws with large random-effect dispersion can round to
                    # exactly 0 or 1 in Float64, so the closed interval.
                    @test all(d -> 0 <= d <= 1, d)
                end
            end

            # (ii) at (numerically) zero random-effect variance the draws are
            # conditional-on-nothing: their mean is the fitted marginal mean.
            z = _ls_helpers_fixture(kind; recov = [log(1e-9), 0.0, log(1e-9)])
            simz = DRModels._ls_marginal_simulator(z.fit, z.data)
            rng = MersenneTwister(20261020)
            draws = reduce(hcat, [simz(rng) for _ in 1:N])
            @test size(draws) == (n, N)
            @test all(kind === :gamma ? >(0) : (d -> 0 < d < 1), draws)
            eta = z.Xmu * z.theta[1:2]
            mu = kind === :gamma ? exp.(eta) : 1 ./ (1 .+ exp.(-eta))
            @test mu ≈ DRModels._mean_response(z.family, eta)
            emp = vec(sum(draws; dims = 2)) ./ N
            sd = vec(sqrt.(sum(abs2, draws .- emp; dims = 2) ./ (N - 1)))
            se = sd ./ sqrt(N)
            # 5 Monte Carlo standard errors per row (seeded, so deterministic).
            @test all(abs.(emp .- mu) .<= 5 .* se)
            @test maximum(abs.(emp .- mu) ./ mu) < 0.15
            # The tolerance is not vacuous: a shifted mean lies far outside it.
            @test !all(abs.(emp .- 1.2 .* mu) .<= 5 .* se)
        end
    end
end
