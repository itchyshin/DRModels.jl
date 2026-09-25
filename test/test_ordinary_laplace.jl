# Arc 2: `marginal = :Laplace` on an ordinary `(1 | g)` (ordinary_laplace.jl).
#
# In-process relationship tests (D-277), no machine-pinned constants:
#   (1) the fitted log-likelihood IS the Laplace approximation: an independent
#       per-group scalar Newton + curvature computation in this file (Distributions
#       log-densities, ForwardDiff derivatives) reproduces −nll at θ̂ and at a
#       perturbed θ, for all five families;
#   (2) θ̂ is a stationary point of that independent objective;
#   (3) the same model through an existing verified route: relmat(1 | g) with
#       K = I (Poisson/NB2/Gamma/Beta) and Poisson :AGHQ with nAGQ = 1 (one-point
#       adaptive quadrature is the Laplace approximation);
#   (4) guards (D-273): the default route is still tagged :LA, its optimum is
#       NOT the Laplace objective's (it is GHQ-32), explicit :LA equals the
#       default bit for bit, and the structured Laplace kernels are unchanged
#       when `raw_scales = false`;
#   (4b) the route uses unclamped scales (`raw_scales = true`): fits and
#       objective values OUTSIDE the legacy clamp box (dispersion shape > e^8,
#       RE log-SD < −8) still equal the independent Laplace objective. These
#       fail if the route is reverted to the clamped kernels;
#   (4c) at family sigma ≈ 0.012 the fit matches native drmTMB on the same data
#       (test/fixtures/ordinary_laplace/; the only cross-engine constants in this
#       file) and reports converged = true through the scale-free rule;
#   (5) every out-of-scope model is refused with this route's message, never
#       rerouted to :LA;
#   (6) the bridge forwards `marginal` and reports the integrator it used;
#   (7) lrtest accepts the fixed-effects model against a `:Laplace` random
#       intercept (its log-likelihood is exact), and still refuses `:LA` vs
#       `:Laplace` random-effect pairs and any VA ELBO.
# The same-target numbers against native drmTMB live in
# docs/dev-log/evidence/arc2-ordinary-laplace/ (native_fit.R, julia_fit.jl).

using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions as Dist
import ForwardDiff

const OL = DRModels

_ol_logistic(η) = 1 / (1 + exp(-η))

function _ol_sim(fam::Symbol; seed, G = 25, m = 8, sd_g = 0.6, sigma = nothing)
    rng = Random.Xoshiro(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    x = randn(rng, n)
    b = sd_g .* randn(rng, G)
    η(β0, β1) = β0 .+ β1 .* x .+ b[g]
    y = if fam === :poisson
        [rand(rng, Dist.Poisson(exp(e))) for e in η(0.4, 0.5)]
    elseif fam === :nb2
        r = 1 / 0.5^2
        [rand(rng, Dist.NegativeBinomial(r, r / (r + exp(e)))) for e in η(0.8, 0.4)]
    elseif fam === :binomial
        [rand(rng, Dist.Bernoulli(_ol_logistic(e))) ? 1.0 : 0.0 for e in η(-0.2, 0.8)]
    elseif fam === :gamma
        α = 1 / something(sigma, 0.4)^2
        [rand(rng, Dist.Gamma(α, exp(e) / α)) for e in η(0.3, 0.5)]
    else
        φ = 1 / something(sigma, 0.3)^2
        [clamp(rand(rng, Dist.Beta(_ol_logistic(e) * φ, (1 - _ol_logistic(e)) * φ)), 1e-6, 1 - 1e-6)
         for e in η(0.2, 0.6)]
    end
    return (y = Float64.(y), x = x, g = ["g$(lpad(k, 2, '0'))" for k in g])
end

_ol_family(fam) = fam === :poisson ? Poisson() : fam === :nb2 ? NegBinomial2() :
                  fam === :binomial ? OL.Binomial() : fam === :gamma ? OL.Gamma() : OL.Beta()
_ol_hassigma(fam) = fam in (:nb2, :gamma, :beta)
_ol_formula(fam) = _ol_hassigma(fam) ?
    bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)) : bf(@formula(y ~ x + (1 | g)))

# log p(y | η) in drmTMB's parameterisation (σ = the `sigma` slot on the log scale).
function _ol_logpdf(fam, y, η, logσ)
    fam === :poisson && return Dist.logpdf(Dist.Poisson(exp(η)), round(Int, y))
    fam === :binomial && return Dist.logpdf(Dist.Bernoulli(_ol_logistic(η)), y == 1)
    s = exp(-2 * logσ)                                 # size / shape / precision = 1/σ²
    fam === :nb2 && return Dist.logpdf(Dist.NegativeBinomial(s, s / (s + exp(η))), round(Int, y))
    fam === :gamma && return Dist.logpdf(Dist.Gamma(s, exp(η) / s), y)
    μ = _ol_logistic(η)
    return Dist.logpdf(Dist.Beta(μ * s, (1 - μ) * s), y)
end

# Independent Laplace negative log-likelihood: one scalar Newton solve and one
# curvature per group. θ = [β0, β1, (log σ,) log σ_b].
function _ol_reference_nll(fam, d, θ)
    hs = _ol_hassigma(fam)
    β0, β1 = θ[1], θ[2]
    logσ = hs ? θ[3] : 0.0
    σb = exp(θ[end])
    total = zero(eltype(θ))
    for lev in unique(d.g)
        idx = findall(==(lev), d.g)
        h(b) = sum(_ol_logpdf(fam, d.y[i], β0 + β1 * d.x[i] + b, logσ) for i in idx) +
               Dist.logpdf(Dist.Normal(0, σb), b)
        b = 0.0
        for _ in 1:100
            d1 = ForwardDiff.derivative(h, b)
            d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), b)
            step = d1 / d2
            b -= step
            abs(step) < 1e-13 && break
        end
        d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), b)
        total += h(b) + 0.5 * log(2π) - 0.5 * log(-d2)
    end
    return -total
end

const _OL_FAMS = (:poisson, :nb2, :binomial, :gamma, :beta)

@testset "ordinary (1 | g) marginal = :Laplace (Arc 2)" begin
    @testset "$fam: logLik is the Laplace approximation, θ̂ its optimum" for fam in _OL_FAMS
        d = _ol_sim(fam; seed = 20260924)
        fit = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
        @test fit.marginal === :Laplace
        @test fit.converged
        θ̂ = copy(fit.theta)
        @test length(θ̂) == (_ol_hassigma(fam) ? 4 : 3)
        @test dof(fit) == length(θ̂)
        @test loglik(fit) ≈ -_ol_reference_nll(fam, d, θ̂) atol = 1e-7
        θp = θ̂ .+ [0.05, -0.03, fill(0.1, length(θ̂) - 2)...]
        @test fit.nll(θp) ≈ _ol_reference_nll(fam, d, θp) atol = 1e-7
        gref = ForwardDiff.gradient(θ -> _ol_reference_nll(fam, d, θ), θ̂)
        @test maximum(abs, gref) < 1e-3
        # the route's analytic gradient agrees with the independent objective
        gfit = zeros(length(θp)); fit.nllgrad(gfit, θp)
        @test gfit ≈ ForwardDiff.gradient(θ -> _ol_reference_nll(fam, d, θ), θp) rtol = 1e-5
    end

    @testset "$fam: same model as relmat(1 | g) with K = I" for fam in (:poisson, :nb2, :gamma, :beta)
        d = _ol_sim(fam; seed = 20260925)
        fit = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
        levs = unique(d.g)                            # relmat labels follow first-seen order
        K = Matrix(1.0I, length(levs), length(levs))
        fr = _ol_hassigma(fam) ?
            bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)) :
            bf(@formula(y ~ x + relmat(1 | g)))
        fitK = drm(fr, _ol_family(fam); data = d, K = K, se = false)
        @test loglik(fit) ≈ loglik(fitK) atol = 1e-7
        @test fit.theta ≈ fitK.theta rtol = 1e-4
    end

    # The legacy structured kernels clamp −2 log σ to [−8, 8] and the RE log-SD
    # to [−8, 3] while differentiating as if unclamped. The ordinary route must
    # not: σ = 0.01 puts the shape/precision at 1e4 > e^8 ≈ 2981.
    @testset "$fam: fit outside the legacy clamp box (sigma = 0.01)" for fam in (:gamma, :beta)
        d = _ol_sim(fam; seed = 20260926, sigma = 0.01)
        fit = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
        θ̂ = copy(fit.theta)
        @test -2 * θ̂[3] > 8                          # the estimate lies beyond the old clamp
        @test loglik(fit) ≈ -_ol_reference_nll(fam, d, θ̂) atol = 1e-6
        # Optimality, scale-free: one Newton step on the independent objective
        # moves θ̂ by < 1e-5. (At precision 1e4 the curvature in β is ~1e6, so the
        # absolute-gradient `converged` flag is not a fair test here; it is not
        # asserted. A clamped route stops far from this optimum and fails both checks.)
        ref(θ) = _ol_reference_nll(fam, d, θ)
        step = ForwardDiff.hessian(ref, θ̂) \ ForwardDiff.gradient(ref, θ̂)
        @test maximum(abs, step) < 1e-5
        # The reported flag uses the same scale-free question (Newton decrement)
        # when the raw gradient is large only because the curvature is.
        @test fit.converged
    end

    # Same model, same data as native drmTMB (engine = "tmb") at family
    # sigma ≈ 0.012. The fixtures were simulated in R (seeds 31342 / 31343);
    # the reference numbers below are native drmTMB's fit of these CSV files
    # (opt$convergence == 0 for both). Before the scale-free convergence rule
    # the route reported converged = false here although it matched native.
    @testset "$fam: sigma ≈ 0.012 matches native drmTMB and reports converged" for (fam, file, ll_native, θ_native) in (
            (:gamma, "gamma_sigma0012.csv", 756.7479683642,
             [-0.273123496603, 0.350771345980, 0.398019933810, 0.099792314361,
              -0.301739125168, -4.429717996930, -0.258783385387]),
            (:beta, "beta_sigma0012.csv", 1047.6527834522,
             [-0.229630708498, 0.497315528616, -0.399362363229, 0.103774523575,
              -0.299637339714, -4.454239538407, -0.275948443061]))
        lines = readlines(joinpath(@__DIR__, "fixtures", "ordinary_laplace", file))
        rows = [replace.(split(l, ","), "\"" => "") for l in lines[2:end]]
        d = (y = [parse(Float64, r[1]) for r in rows], x = [parse(Float64, r[2]) for r in rows],
             z = [parse(Float64, r[3]) for r in rows], f = [r[4] for r in rows], g = [r[5] for r in rows])
        fit = drm(bf(@formula(y ~ x + z + f + (1 | g)), @formula(sigma ~ 1)), _ol_family(fam);
                  data = d, marginal = :Laplace, se = false)
        @test fit.converged
        @test exp(fit.theta[6]) < 0.02                  # the small-sigma regime
        @test loglik(fit) ≈ ll_native atol = 1e-6
        @test fit.theta ≈ θ_native rtol = 1e-4
        # the raw gradient is NOT small here: the flag is not the gradient rule
        g = zeros(length(fit.theta)); fit.nllgrad(g, fit.theta)
        @test maximum(abs, g) / length(d.y) > 1e-6
        # and the scale-free rule is not vacuous: 0.01 away from θ̂ it says no
        θo = fit.theta .+ 0.01
        go = zeros(length(θo)); fit.nllgrad(go, θo)
        @test !OL._ordinary_laplace_newton_converged(fit.nllgrad, θo, go)
    end

    @testset "$fam: objective at an RE log-SD below the legacy clamp (−9)" for fam in _OL_FAMS
        d = _ol_sim(fam; seed = 20260924)
        fit = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
        θo = copy(fit.theta); θo[end] = -9.0
        @test fit.nll(θo) ≈ _ol_reference_nll(fam, d, θo) atol = 1e-6
        go = zeros(length(θo)); fit.nllgrad(go, θo)
        @test go[end] ≈ ForwardDiff.gradient(θ -> _ol_reference_nll(fam, d, θ), θo)[end] rtol = 1e-4 atol = 1e-8
    end

    @testset "Binomial with trials: cbind(successes, failures)" begin
        rng = Random.Xoshiro(21)
        G, m = 20, 6; n = G * m
        g = repeat(1:G, inner = m); x = randn(rng, n); b = 0.7 .* randn(rng, G)
        ntr = rand(rng, 2:9, n)
        s = [rand(rng, Dist.Binomial(ntr[i], _ol_logistic(-0.3 + 0.6x[i] + b[g[i]]))) for i in 1:n]
        d = (s = Float64.(s), f = Float64.(ntr .- s), x = x, g = ["g$k" for k in g])
        fit = drm(bf(@formula(cbind(s, f) ~ x + (1 | g))), OL.Binomial(); data = d,
                  marginal = :Laplace, se = false)
        @test fit.marginal === :Laplace && fit.converged
        function refnll(θ)
            tot = zero(eltype(θ))
            for lev in unique(d.g)
                idx = findall(==(lev), d.g)
                h(u) = sum(Dist.logpdf(Dist.Binomial(ntr[i], _ol_logistic(θ[1] + θ[2] * x[i] + u)), s[i])
                           for i in idx) + Dist.logpdf(Dist.Normal(0, exp(θ[3])), u)
                u = 0.0
                for _ in 1:100
                    st = ForwardDiff.derivative(h, u) /
                         ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), u)
                    u -= st; abs(st) < 1e-13 && break
                end
                tot += h(u) + 0.5 * log(2π) -
                       0.5 * log(-ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), u))
            end
            return -tot
        end
        @test loglik(fit) ≈ -refnll(fit.theta) atol = 1e-7
        @test maximum(abs, ForwardDiff.gradient(refnll, fit.theta)) < 1e-3
    end

    @testset "Poisson: equals one-point AGHQ (nAGQ = 1)" begin
        d = _ol_sim(:poisson; seed = 7)
        fit = drm(_ol_formula(:poisson), Poisson(); data = d, marginal = :Laplace, se = false)
        fitA = drm(_ol_formula(:poisson), Poisson(); data = d, marginal = :AGHQ, nAGQ = 1)
        @test loglik(fit) ≈ loglik(fitA) atol = 1e-6
        @test fit.theta ≈ fitA.theta rtol = 1e-4
    end

    @testset "guard: default :LA stays GHQ-32 and differs from :Laplace" begin
        for fam in _OL_FAMS
            d = _ol_sim(fam; seed = 11)
            fitLA = drm(_ol_formula(fam), _ol_family(fam); data = d, se = false)
            fitL = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
            @test fitLA.marginal === :LA
            # the GHQ-32 objective at :LA's optimum is not the Laplace objective
            @test abs(-loglik(fitLA) - _ol_reference_nll(fam, d, fitLA.theta)) > 1e-4
            @test abs(loglik(fitLA) - loglik(fitL)) > 1e-4
        end
        # explicit :LA is the default
        d = _ol_sim(:nb2; seed = 11)
        a = drm(_ol_formula(:nb2), NegBinomial2(); data = d, se = false)
        b = drm(_ol_formula(:nb2), NegBinomial2(); data = d, se = false, marginal = :LA)
        @test loglik(a) == loglik(b) && a.theta == b.theta
    end

    @testset "guard: structured kernels unchanged when raw_scales = false" begin
        # Inside the clamp box the flag is inert; outside it only raw_scales moves.
        d = _ol_sim(:nb2; seed = 3)
        y = d.y; X = hcat(ones(length(y)), d.x)
        gidx, G = OL._group_index(d.g)
        Q = spdiagm(0 => ones(G))
        aux_c, _, _ = OL._nb2_laplace_setup(y, X)
        aux_r, _, _ = OL._nb2_laplace_setup(y, X; raw_scales = true)
        θin = [0.7, 0.4, -0.6, -0.4]
        v0, _, _ = OL._phylo_mean_laplace_nuisance_fg(Val(:nb2_fixed), aux_c, length(y), X, gidx, Q, 0.0, θin)
        v1, _, _ = OL._phylo_mean_laplace_nuisance_fg(Val(:nb2_fixed), aux_r, length(y), X, gidx, Q, 0.0, θin;
                                                     raw_scales = true)
        @test v0 == v1
        θout = [0.7, 0.4, -0.6, -9.0]                 # RE log-SD below the legacy clamp at −8
        w0, _, _ = OL._phylo_mean_laplace_nuisance_fg(Val(:nb2_fixed), aux_c, length(y), X, gidx, Q, 0.0, θout)
        w0c, _, _ = OL._phylo_mean_laplace_nuisance_fg(Val(:nb2_fixed), aux_c, length(y), X, gidx, Q, 0.0,
                                                      [0.7, 0.4, -0.6, -8.0])
        @test w0 == w0c                                # default: clamped, as before
        w1, _, _ = OL._phylo_mean_laplace_nuisance_fg(Val(:nb2_fixed), aux_r, length(y), X, gidx, Q, 0.0, θout;
                                                     raw_scales = true)
        @test w1 != w0
    end

    @testset "missing response rows are dropped, as on :LA" begin
        d = _ol_sim(:poisson; seed = 5)
        dm = (y = vcat(NaN, d.y[2:end]), x = d.x, g = d.g)
        fitm = @test_logs (:warn,) match_mode = :any drm(_ol_formula(:poisson), Poisson();
                                                        data = dm, marginal = :Laplace, se = false)
        keep = 2:length(d.y)
        fito = drm(_ol_formula(:poisson), Poisson(); data = (y = d.y[keep], x = d.x[keep], g = d.g[keep]),
                   marginal = :Laplace, se = false)
        @test loglik(fitm) ≈ loglik(fito) atol = 1e-10
        @test fitm.marginal === :Laplace
    end

    @testset "lrtest: fixed-effects model vs a :Laplace random intercept" begin
        for fam in (:poisson, :nb2)
            d = _ol_sim(fam; seed = 5)
            f0 = _ol_hassigma(fam) ? bf(@formula(y ~ x), @formula(sigma ~ 1)) : bf(@formula(y ~ x))
            fit0 = drm(f0, _ol_family(fam); data = d, se = false)       # exact loglik, tagged :LA
            fitL = drm(_ol_formula(fam), _ol_family(fam); data = d, marginal = :Laplace, se = false)
            fitLA = drm(_ol_formula(fam), _ol_family(fam); data = d, se = false)
            @test OL._fit_is_re_free(fit0)
            @test !OL._fit_is_re_free(fitL) && !OL._fit_is_re_free(fitLA)
            t = @test_logs (:warn, r"BOUNDARY") match_mode = :any lrtest(fit0, fitL)
            @test t.statistic ≈ 2 * (loglik(fitL) - loglik(fit0))
            @test t.dof == 1
            @test (@test_logs (:warn,) match_mode = :any anova(fit0, fitL)) == t
            # the default route's own test is unchanged (same tag on both sides)
            tLA = @test_logs (:warn,) match_mode = :any lrtest(fit0, fitLA)
            @test tLA.statistic ≈ 2 * (loglik(fitLA) - loglik(fit0))
            # two random-effect fits with different integrators stay refused, and the
            # message names the integrators rather than blaming an ELBO
            err = @test_throws ArgumentError lrtest(fitLA, fitL)
            msg = sprint(showerror, err.value)
            @test occursin("GHQ-32", msg) && occursin("`:Laplace`", msg)
            @test !occursin("ELBO", msg)
        end
        # a VA ELBO is still never compared with a log-likelihood, even an exact one
        d = _ol_sim(:poisson; seed = 5)
        fit0 = drm(bf(@formula(y ~ x)), Poisson(); data = d, se = false)
        fitVA = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = d, marginal = :VA, se = false)
        err = @test_throws ArgumentError lrtest(fit0, fitVA)
        @test occursin("ELBO", sprint(showerror, err.value))
    end

    @testset "out-of-scope models are refused, never rerouted" begin
        dp = _ol_sim(:poisson; seed = 9)
        dp = merge(dp, (h = repeat(["a", "b", "c", "d"], length(dp.y) ÷ 4), z = randn(Random.Xoshiro(1), length(dp.y))))
        dn = merge(_ol_sim(:nb2; seed = 9), (z = randn(Random.Xoshiro(2), length(dp.y)),))
        # Each refusal is an ArgumentError FROM THIS ROUTE (its message names the
        # route and the offending piece), not some unrelated downstream error.
        function refuse(f, fam, d, what::Regex; kw...)
            err = @test_throws ArgumentError drm(f, fam; data = d, marginal = :Laplace, kw...)
            msg = sprint(showerror, err.value)
            @test occursin(r"marginal = :Laplace is not available for \w+\(\) with", msg)
            @test occursin(what, msg)
        end
        refuse(bf(@formula(y ~ x + (1 + x | g))), Poisson(), dp, r"only `\(1 \| g\)` is covered")
        refuse(bf(@formula(y ~ x + (0 + x | g))), Poisson(), dp, r"only `\(1 \| g\)` is covered")
        refuse(bf(@formula(y ~ x + (1 | g) + (1 | h))), Poisson(), dp, r"crossed/multiple random effects")
        refuse(bf(@formula(y ~ x)), Poisson(), dp, r"no random effect")
        refuse(bf(@formula(y ~ x + (1 | g)), @formula(zi ~ 1)), Poisson(), dp, r"a `zi` formula")
        refuse(bf(@formula(y ~ x + (1 | g))), Poisson(), dp, r"`method = :REML`"; method = :REML)
        refuse(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ z)), NegBinomial2(), dn,
               r"non-constant `sigma` formula")
        refuse(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1 + (1 | g))), NegBinomial2(), dn,
               r"random effect on `sigma`")
        refuse(bf(@formula(y ~ x + (1 | p | g)), @formula(sigma ~ 1 + (1 | p | g))), NegBinomial2(), dn,
               r"random effect on `sigma`")
        levs = unique(dp.g)
        refuse(bf(@formula(y ~ x + relmat(1 | g))), Poisson(), dp, r"phylogenetic/structured random effect";
               K = Matrix(1.0I, length(levs), length(levs)))
        # `method = :Laplace` points the caller at `marginal`
        errm = @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dp,
                                              method = :Laplace)
        @test occursin(r"is not the Laplace/VA/AGHQ selector.*Use `marginal = :Laplace`", sprint(showerror, errm.value))
        @test !occursin(":LAPLACE", sprint(showerror, errm.value))   # canonical spelling, not the uppercased key
        # an ML-only family's REML refusal names :Laplace among the marginal options
        for m in (:REML, :bogus)
            errr = @test_throws ArgumentError OL._reject_method_as_marginal(OL.Beta(), m)
            @test occursin(r"ML-only.*`:Laplace`", sprint(showerror, errr.value))
        end
    end

    @testset "bridge forwards `marginal` and reports the integrator used" begin
        d = _ol_sim(:binomial; seed = 13)
        fit = drm(_ol_formula(:binomial), OL.Binomial(); data = d, marginal = :Laplace)
        outL = drm_bridge(; formula = "y ~ x + (1 | g)", family = "binomial", data = d,
                          options = Dict(:marginal => "Laplace"))
        outD = drm_bridge(; formula = "y ~ x + (1 | g)", family = "binomial", data = d)
        @test outL["marginal"] == "Laplace"
        @test outD["marginal"] == "LA"
        @test outL["loglik"] ≈ loglik(fit) atol = 1e-8
        @test abs(outL["loglik"] - outD["loglik"]) > 1e-4
    end
end
