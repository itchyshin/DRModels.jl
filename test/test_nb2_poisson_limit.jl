# NB2 at the Poisson limit (#1072, #1069, #1039).
#
# When the data carry no overdispersion the NB2 likelihood is increasing in the
# size r = 1/sigma^2 all the way to infinity, so the MLE sits at the Poisson limit
# (log sigma -> -Inf). Three things went wrong there:
#   #1072  the fit came back with no flag at all (a clean, converged-looking fit
#          whose sigma is an optimiser-stopping artefact);
#   #1069  on Poisson-generated data with a random intercept the optimiser ran onto
#          that plateau although an interior sigma had a HIGHER likelihood;
#   #1039  the LBFGS / HagerZhang line search threw an uncaught AssertionError.
# The sparse-Laplace routes (phylo / relmat / animal / spatial / crossed) clamp the
# size at exp(8), i.e. log sigma >= -4, so there the Poisson limit is that clamp: they
# refit with the clamp as the threshold and give a clamp-specific version of the warning
# (its logLik is NB2 at the capped size, not the Poisson one).
# Not covered here: location SEs with phi held at the limit (#1072 request 3) -- that
# is the SE path and is tracked separately.

using DRModels
using Test
using Random
using LinearAlgebra
using Logging
using StableRNGs
using DelimitedFiles
import Distributions
import ForwardDiff

# Underdispersed counts (Binomial variance < mean): the NB2 likelihood has no
# interior maximum in sigma, so the MLE is on the Poisson boundary by construction.
function _nb2_underdispersed(seed; n = 200)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    y = [Float64(rand(rng, Distributions.Binomial(20, 0.3))) for _ in 1:n]
    return (; y = y, x = x)
end

# The data of the #1069 / #1039 reports (50 groups x 20, group SD 0.3), read from the
# fixture: `Random.seed!` draws different counts on Julia 1.10 (fixture README).
function _nb2_poisson_glmm(seed)
    tab = readdlm(joinpath(@__DIR__, "fixtures", "nb2_poisson_limit", "data_seed$(seed).csv"), ',', Int;
                  skipstart = 1)
    return (; y = Float64.(tab[:, 1]), g = tab[:, 2])
end

# Poisson counts with a phylogenetic random intercept (48 tips x 8, phylo SD 0.3).
function _nb2pl_phylo_poisson(seed; p = 48, m = 8)
    rng = StableRNG(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    species = repeat(1:p, inner = m)
    x = randn(rng, length(species))
    C = sigma_phy_dense(phy; σ²_phy = 0.3^2)
    u = cholesky(Symmetric(C)).L * randn(rng, p)
    y = [Float64(rand(rng, Distributions.Poisson(exp(3.0 + 0.2 * x[i] + u[species[i]]))))
         for i in eachindex(species)]
    return (; y, x, species), phy, Matrix(C) ./ maximum(diag(C))
end

# As above, plus a binary dispersion covariate z for `sigma ~ 1 + z` (the covariate-sigma
# spine); `size = Inf` is Poisson, otherwise NB2 with that size.
function _nb2pl_phylo_z(seed; p = 48, m = 8, size = Inf)
    rng = StableRNG(seed)
    phy = random_balanced_tree(p; branch_length = 0.2)
    species = repeat(1:p, inner = m); n = length(species)
    x = randn(rng, n); z = Float64.(rand(rng, n) .< 0.5)
    C = sigma_phy_dense(phy; σ²_phy = 0.3^2)
    u = cholesky(Symmetric(C)).L * randn(rng, p)
    μ = exp.(3.0 .+ 0.2 .* x .+ u[species])
    y = isinf(size) ? [Float64(rand(rng, Distributions.Poisson(μi))) for μi in μ] :
        [Float64(rand(rng, Distributions.NegativeBinomial(size, size / (size + μi)))) for μi in μ]
    return (; y, x, z, species), phy
end

# Poisson counts with crossed random intercepts (G x H fully crossed, SDs 0.3).
function _nb2pl_crossed_poisson(seed; G = 30, H = 20)
    rng = StableRNG(seed)
    g = repeat(1:G, inner = H); h = repeat(1:H, outer = G)
    x = randn(rng, G * H); bg = 0.3 .* randn(rng, G); bh = 0.3 .* randn(rng, H)
    y = [Float64(rand(rng, Distributions.Poisson(exp(3.0 + 0.2 * x[i] + bg[g[i]] + bh[h[i]]))))
         for i in eachindex(g)]
    return (; y, x, g, h)
end

# Messages of every warning `f()` emits, and its value.
function _nb2pl_warnings(f)
    logger = Test.TestLogger(min_level = Logging.Warn)
    val = with_logger(f, logger)
    return val, [string(r.message) for r in logger.logs]
end

const _NB2PL_CLAMPED = r"no overdispersion that this route can resolve"
const _NB2PL_GENERIC = r"dispersion log sigma"

@testset "NB2 at the Poisson limit" begin

    @testset "#1072 the boundary is flagged, not silent" begin
        d = _nb2_underdispersed(7)
        fit = @test_logs (:warn, r"no overdispersion") match_mode = :any drm(
            bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        @test DRModels._nb2_at_poisson_limit(fit)
        # the location estimates are the Poisson ones, so the estimate itself is fine
        pfit = drm(bf(@formula(y ~ 1 + x)), DRModels.Poisson(); data = d)
        @test isapprox(coef(fit)[1:2], coef(pfit)[1:2]; atol = 1e-3)
    end

    @testset "#1072 on the (1 | g) route too" begin
        # underdispersed grouped counts: the refit cannot leave the plateau, so it warns
        rng = StableRNG(5)
        g = repeat(1:20, inner = 20)
        y = [Float64(rand(rng, Distributions.Binomial(40, 0.5))) for _ in g]
        fit, msgs = _nb2pl_warnings(() -> drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                                              data = (; y, g)))
        @test fit.theta[2] < -8
        @test DRModels._nb2_at_poisson_limit(fit)
        @test count(m -> occursin("so the fit is at the Poisson limit", m), msgs) == 1
    end

    @testset "a genuinely overdispersed fit is not flagged" begin
        rng = MersenneTwister(4)
        n = 600
        x = randn(rng, n)
        r = 1 / 0.6^2
        mu = exp.(1.4 .- 0.2 .* x)
        y = [Float64(rand(rng, Distributions.NegativeBinomial(r, r / (r + mu[i])))) for i in 1:n]
        fit = @test_logs min_level = Base.CoreLogging.Warn drm(
            bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = (; y = y, x = x))
        @test !DRModels._nb2_at_poisson_limit(fit)
    end

    @testset "#1039 an AssertionError from one line search falls through to the next" begin
        calls = Symbol[]
        out = DRModels._with_optimizer_fallback([
            () -> (push!(calls, :first); throw(AssertionError("isfinite(phi_c)"))),
            () -> (push!(calls, :second); :ok),
        ])
        @test out === :ok
        @test calls == [:first, :second]
        # anything that is not an AssertionError is not swallowed
        @test_throws ErrorException DRModels._with_optimizer_fallback([
            () -> error("a real bug"), () -> :unreached])
        # and when every attempt fails the last error surfaces
        @test_throws AssertionError DRModels._with_optimizer_fallback([
            () -> throw(AssertionError("a")), () -> throw(AssertionError("b"))])
    end

    @testset "#1039 _nb2_optimize: an assertion in both LBFGS searches ends in Nelder-Mead" begin
        # The objective asserts whenever it is differentiated (ForwardDiff duals), as the
        # Hager-Zhang search does on a runaway; both LBFGS attempts therefore fail and the
        # derivative-free attempt must still find the minimum.
        target = [1.0, -2.0]
        f(θ) = (eltype(θ) <: AbstractFloat || throw(AssertionError("isfinite(phi_c) && isfinite(dphi_c)"));
                sum(abs2, θ .- target))
        res = DRModels._nb2_optimize(f, [0.0, 0.0], 1e-8)
        @test DRModels.Optim.summary(res) == "Nelder-Mead"
        @test DRModels.Optim.minimizer(res) ≈ target atol = 1e-3
    end

    @testset "#1039 seed 20282004: the refit leaves the plateau, so no Poisson-limit warning" begin
        # On the current engine this seed no longer reaches the Hager-Zhang assertion
        # (LBFGS itself returns the Poisson plateau, log sigma -37.5); what it exercises
        # is the plateau refit, which must find drmTMB's interior optimum.
        d = _nb2_poisson_glmm(20282004)
        @test sum(d.y) == 165495                        # the #1039 data (its checksum)
        nb = @test_logs min_level = Base.CoreLogging.Warn drm(
            bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        po = drm(bf(@formula(y ~ 1 + (1 | g))), DRModels.Poisson(); data = d)
        # Replaying the optimisation from `_fit_negbin2_ranef`'s own start on the fitted
        # objective ends on the plateau ...
        y = d.y; n = length(y); m = sum(y) / n; v = sum(abs2, y .- m) / (n - 1)
        θ0 = [log(m), -0.5 * log(max(m^2 / max(v - m, 0.1 * m), 0.5)), log(0.5)]
        first_try = DRModels._nb2_optimize(nb.nll, θ0, 1e-8)
        @test DRModels._nb2_at_poisson_limit(DRModels.Optim.minimizer(first_try)[2:2])
        # ... and the kept fit is the interior refit: drmTMB's log sigma -5.049 (#1069),
        # 0.012 nats above both the plateau fit and the Poisson GLMM, at a stationary point.
        # (Not `converged`: Optim's |g| <= 1e-8 on an nll of ~4094 is met by default but not
        # under `--check-bounds=yes`, as `Pkg.test` runs: |g| = 5e-9 vs 3.4e-8.)
        @test !DRModels._nb2_at_poisson_limit(nb)
        @test nb.theta[2] ≈ -5.049 atol = 0.02
        @test loglik(nb) - (-nb.nll(DRModels.Optim.minimizer(first_try))) ≈ 0.0121 atol = 0.002
        @test loglik(nb) - loglik(po) ≈ 0.0121 atol = 0.002
        @test maximum(abs, ForwardDiff.gradient(nb.nll, nb.theta)) < 1e-4
    end

    @testset "#1069 seed 20282003: NB2 is not left on the plateau below an interior optimum" begin
        d = _nb2_poisson_glmm(20282003)
        nb = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        po = drm(bf(@formula(y ~ 1 + (1 | g))), DRModels.Poisson(); data = d)
        # NB2 nests Poisson, so its maximised logLik cannot be below the Poisson one;
        # an interior sigma (drmTMB: log sigma -3.791) gains ~1.46 nats on this seed.
        @test loglik(nb) > loglik(po) + 1.0
        @test nb.theta[2] ≈ -3.791 atol = 0.02
    end

    @testset "sparse-Laplace routes: the Poisson limit is the -4 clamp, flagged as such" begin
        d, phy, K = _nb2pl_phylo_poisson(2)
        fits = Pair{String,Any}[]
        push!(fits, "phylo" => _nb2pl_warnings(() -> drm(
            bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), NegBinomial2();
            data = d, tree = phy, se = false)))
        push!(fits, "relmat" => _nb2pl_warnings(() -> drm(
            bf(@formula(y ~ x + relmat(1 | id))), NegBinomial2();
            data = (; y = d.y, x = d.x, id = string.(d.species)), K = K, se = false)))
        push!(fits, "crossed" => _nb2pl_warnings(() -> drm(
            bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), NegBinomial2();
            data = _nb2pl_crossed_poisson(11), se = false)))
        for (route, (fit, msgs)) in fits
            @testset "$route" begin
                @test coef(fit, :sigma)[1] <= -4 + 1e-3           # at (or past) the clamp
                @test count(m -> occursin(_NB2PL_CLAMPED, m), msgs) == 1
                # it replaces the generic #959 dispersion line, whose advice does not fit
                @test !any(m -> occursin(_NB2PL_GENERIC, m), msgs)
                @test all(isfinite, coef(fit))
            end
        end
        # silent inside bootstrap / profile refits
        _, quiet = _nb2pl_warnings(() -> DRModels._without_boundary_warnings(() -> drm(
            bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), NegBinomial2();
            data = d, tree = phy, se = false)))
        @test !any(m -> occursin(_NB2PL_CLAMPED, m), quiet)
    end

    @testset "sparse-Laplace warning: every row at the clamp, not just some" begin
        # covariate-sigma layout: the per-row log sigma is Xσ β̂σ
        nb2 = DRModels.DrmFit(NegBinomial2(), [:mu => 1:1, :sigma => 2:3, :resd => 4:4],
            [:mu => ["(Intercept)"], :sigma => ["(Intercept)", "x"], :resd => ["g"]],
            [0.0, -5.0, 0.5, -1.0], Matrix(1.0I, 4, 4), -10.0, 2, true,
            Dict(:mu => ones(2)), Dict(:mu => ones(2)), Dict(:sigma => exp.([-4.5, -6.0])))
        _, msgs = _nb2pl_warnings(() -> DRModels._warn_laplace_clamp(nb2; logsigma = [-4.5, -6.0]))
        @test count(m -> occursin(_NB2PL_CLAMPED, m), msgs) == 1
        @test !any(m -> occursin(_NB2PL_GENERIC, m), msgs)
        # one row interior: not the Poisson limit, so the generic clamp line stays
        _, msgs = _nb2pl_warnings(() -> DRModels._warn_laplace_clamp(nb2; logsigma = [-4.5, -2.0]))
        @test !any(m -> occursin(_NB2PL_CLAMPED, m), msgs)
        @test any(m -> occursin(_NB2PL_GENERIC, m) && occursin("lower clamp", m), msgs)
        # the NB2 message is for NB2 only: another family at the clamp keeps the generic line
        gam = DRModels.DrmFit(DRModels.Gamma(), nb2.blocks, nb2.coefnames, nb2.theta, nb2.vcov, -10.0, 2,
            true, nb2.means, nb2.obs, nb2.scales)
        _, msgs = _nb2pl_warnings(() -> DRModels._warn_laplace_clamp(gam; logsigma = [-4.5, -6.0]))
        @test !any(m -> occursin(_NB2PL_CLAMPED, m), msgs)
        @test any(m -> occursin(_NB2PL_GENERIC, m), msgs)
    end

    @testset "covariate sigma (phylo, sigma ~ 1 + z) end to end" begin
        f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1 + z))
        rowlogσ(fit, d) = coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* d.z
        # every row at the clamp: the Poisson limit, so the NB2 message
        d, phy = _nb2pl_phylo_z(2)
        fit, msgs = _nb2pl_warnings(() -> drm(f, NegBinomial2(); data = d, tree = phy, se = false))
        @test maximum(rowlogσ(fit, d)) <= -4 + 1e-3
        @test count(m -> occursin(_NB2PL_CLAMPED, m), msgs) == 1
        @test !any(m -> occursin(_NB2PL_GENERIC, m), msgs)
        # only some rows at the clamp: not the Poisson limit, so the generic #959 line
        d, phy = _nb2pl_phylo_z(3)
        fit, msgs = _nb2pl_warnings(() -> drm(f, NegBinomial2(); data = d, tree = phy, se = false))
        @test maximum(rowlogσ(fit, d)) > -4 + 1e-3
        @test !any(m -> occursin(_NB2PL_CLAMPED, m), msgs)
        @test any(m -> occursin(_NB2PL_GENERIC, m), msgs)
        # No end-to-end pin of a KEPT refit on this spine: the one kept case found
        # (`_nb2pl_phylo_z(4; size = 1500)`, refit to a partly clamped optimum 0.25 nats
        # better) is a knife edge. On identical data the refit reached it on Julia 1.12.6
        # and missed it on 1.10.10 and 1.13.2, so pinning it tests floating-point paths.
        # The decision on this spine (rows Xσ βσ, every row at the clamp) is tested below.
    end

    @testset "#1069 on the sparse-Laplace spines: the refit decision" begin
        refit = DRModels._nb2_laplace_refit
        Optim = DRModels.Optim
        at_clamp = Optim.optimize(θ -> sum(abs2, θ .- [1.0, -6.0]), [1.0, -6.0], Optim.NelderMead())
        interior(c) = θs -> Optim.optimize(t -> sum(abs2, t .- [1.0, c]) - 1.0, θs, Optim.LBFGS())
        never = θ -> error("the refit must not run")
        θ0 = [0.0, -1.5]
        # `nll` is what the decision compares, evaluated AT each minimizer: here it puts
        # the clamp fit at 1.0, while `Optim.minimum(at_clamp)` reads 0.0.
        nll_down = θ -> θ[2] < -4 ? 1.0 : 0.5      # the interior refit is better
        nll_up = θ -> θ[2] < -4 ? 1.0 : 2.0        # the interior refit is worse
        @test refit(interior(-2.0), Val(:nb2_fixed), at_clamp, nll_down, θ0, 2:2, nothing) !== at_clamp
        @test Optim.minimizer(refit(interior(-2.0), Val(:nb2_fixed), at_clamp, nll_down, θ0, 2:2,
                                    nothing))[2] ≈ -2.0 atol = 1e-6
        @test refit(interior(-2.0), Val(:nb2_fixed), at_clamp, nll_up, θ0, 2:2, nothing) === at_clamp
        # a refit that returns to the clamp is not kept, even with a lower nll
        @test refit(interior(-7.0), Val(:nb2_fixed), at_clamp, θ -> θ[2] < -6.5 ? 0.0 : 1.0,
                    θ0, 2:2, nothing) === at_clamp
        # covariate sigma: the refit runs only when EVERY row is at the clamp
        Xσ = [1.0 0.0; 1.0 3.0]
        at_clamp2 = Optim.optimize(θ -> sum(abs2, θ .- [1.0, -6.0, 0.0]), [1.0, -6.0, 0.0], Optim.NelderMead())
        part = Optim.optimize(θ -> sum(abs2, θ .- [1.0, -6.0, 1.0]), [1.0, -6.0, 1.0], Optim.NelderMead())
        @test refit(never, Val(:nb2_hetero), part, nll_down, [0.0, -1.5, 0.0], 2:3, Xσ) === part
        @test refit(θs -> Optim.optimize(t -> sum(abs2, t .- [1.0, -2.0, 0.0]), θs, Optim.LBFGS()),
                    Val(:nb2_hetero), at_clamp2, nll_down, [0.0, -1.5, 0.0], 2:3, Xσ) !== at_clamp2
        # The spines' nll warm-starts its inner mode from shared state (`last_b`), which the
        # refit moves. A rejected refit puts it back, so the nll, gradient and Hessian that
        # follow are exactly those of a fit without the refit; a kept refit leaves it moved.
        w = [1.0, 2.0]
        moving(c) = θs -> (w .= 9.0; interior(c)(θs))
        @test refit(moving(-2.0), Val(:nb2_fixed), at_clamp, nll_up, θ0, 2:2, nothing; warm = w) === at_clamp
        @test w == [1.0, 2.0]
        @test refit(moving(-7.0), Val(:nb2_fixed), at_clamp, nll_down, θ0, 2:2, nothing; warm = w) === at_clamp
        @test w == [1.0, 2.0]
        @test refit(moving(-2.0), Val(:nb2_fixed), at_clamp, nll_down, θ0, 2:2, nothing; warm = w) !== at_clamp
        @test w == [9.0, 9.0]
        # interior fits and every other kernel are left alone
        inside = Optim.optimize(θ -> sum(abs2, θ .- [1.0, -2.0]), [1.0, -2.0], Optim.NelderMead())
        @test refit(never, Val(:nb2_fixed), inside, nll_down, θ0, 2:2, nothing) === inside
        @test refit(never, Val(:gamma_fixed), at_clamp, nll_down, θ0, 2:2, nothing) === at_clamp
    end

    @testset "#1069 on the crossed sparse-Laplace route: the refit leaves the clamp" begin
        # From its own start the optimiser stops at the clamp (log sigma -4.63) on this
        # seed; the refit finds an interior optimum about 1.9 nats better.
        d = _nb2pl_crossed_poisson(17)
        fit, msgs = _nb2pl_warnings(() -> drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                                              NegBinomial2(); data = d, se = false))
        @test coef(fit, :sigma)[1] > -4 + 0.5                # interior, well clear of the clamp
        @test coef(fit, :sigma)[1] ≈ -2.643 atol = 0.02
        @test fit.converged
        @test !any(m -> occursin(_NB2PL_CLAMPED, m) || occursin(_NB2PL_GENERIC, m), msgs)
        # The best fit AT the clamp (log sigma fixed at -4, everything else re-optimised)
        # is far worse, so a fit left at the clamp would have been wrong, not just noisy.
        Optim = DRModels.Optim
        others = [1, 2, 4, 5]
        full(v) = (θ = zeros(5); θ[others] = v; θ[3] = -4.0; θ)
        prof = Optim.optimize(v -> fit.nll(full(v)),
                              (G, v) -> (Gf = zeros(5); fit.nllgrad(Gf, full(v)); G .= Gf[others]),
                              fit.theta[others], Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()),
                              Optim.Options(iterations = 500))
        @test fit.nll(full(Optim.minimizer(prof))) - (-loglik(fit)) > 1.0
    end

    @testset "crossed AGHQ at the clamp: one warning, not one per stage" begin
        # marginal = :AGHQ needs one grouping with at most 8 levels. Its Laplace start
        # used to emit its own clamp warning as well as the AGHQ fit's.
        d = _nb2pl_crossed_poisson(3; G = 30, H = 4)
        fit, msgs = _nb2pl_warnings(() -> drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
                                              NegBinomial2(); data = d, se = false, marginal = :AGHQ))
        @test fit.marginal === :AGHQ
        @test coef(fit, :sigma)[1] <= -4 + 1e-3
        @test count(m -> occursin(_NB2PL_CLAMPED, m), msgs) == 1
        @test !any(m -> occursin(_NB2PL_GENERIC, m), msgs)
    end
end
