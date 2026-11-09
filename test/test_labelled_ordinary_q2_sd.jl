# Ordinary (non-structured) correlated intercept-plus-slope with one SD formula per
# coefficient on Poisson and NB2: labelled `(1 + x | p | g)` (both) and unlabelled
# `(1 + x | g)` (Poisson; NB2 unlabelled is the two-field route). drmTMB contract:
# unit-scale fields (u0, u1) iid over groups ~ N(0, R), R = [1 rho; rho 1],
# rho = 0.999999 tanh(eta_cor), b_k = s_k[g] u_k, s_k = exp(Z_k alpha_k).
# The oracle shares no fitter code: a dense ForwardDiff Laplace with the full
# kron(R, I) prior at the fitted theta. Tolerances fixed before the first run:
# -loglik vs oracle atol 1e-4; rho recovery |rho_hat - 0.5| < 0.45 (one dataset, no
# calibration claim); labelled loglik >= independent - 1e-6; eta_rho = 0 reproduces
# the independent identity-kernel objective to 1e-6.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

function _lord_fixture(; G = 48, m = 20, seed = 20261108, rho = 0.5, sigma_nb = 0.45)
    rng = MersenneTwister(seed)
    ids = ["site_$(i)" for i in randperm(rng, G)]
    gi = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.2, 1.2; length = m)), G)
    z0g = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z1g = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    n0, n1 = randn(rng, G), randn(rng, G)
    u0, u1 = n0, rho .* n0 .+ sqrt(1 - rho^2) .* n1
    a0, a1 = [-0.7, 0.35], [-0.9, 0.45]
    eta = 0.2 .+ 0.3 .* x .+ (exp.(a0[1] .+ a0[2] .* z0g) .* u0)[gi] .+
          x .* (exp.(a1[1] .+ a1[2] .* z1g) .* u1)[gi]
    yp = Float64.([rand(rng, Distributions.Poisson(exp(v))) for v in eta])
    r = 1 / sigma_nb^2
    yn = Float64.([rand(rng, Distributions.NegativeBinomial(r, r / (r + exp(v)))) for v in eta])
    return (; data = (; yp, yn, x, z0 = z0g[gi], z1 = z1g[gi], g = ids[gi]), gi, G)
end

_lord_rho(eta) = 0.999999 * tanh(eta)

function _lord_formula(resp::Symbol, family; lhs = "1 + x | p | g")
    mu = eval(Meta.parse("@formula($resp ~ x + ($lhs))"))
    sd0 = @formula(sd(g, coef(intercept)) ~ z0)
    sd1 = @formula(sd(g, coef(x)) ~ z1)
    family isa Poisson && return bf(mu, sd0, sd1)
    return bf(mu, @formula(sigma ~ 1), sd0, sd1)
end

# Independent (rho = 0) identity-kernel comparator: the animal q2 route with A = I.
function _lord_indep_formula(resp::Symbol, family)
    mu = eval(Meta.parse("@formula($resp ~ x + animal(1 + x | g))"))
    sd0 = @formula(sd(g, animal, coef(intercept)) ~ z0)
    sd1 = @formula(sd(g, animal, coef(x)) ~ z1)
    family isa Poisson && return bf(mu, sd0, sd1)
    return bf(mu, @formula(sigma ~ 1), sd0, sd1)
end

# Dense Laplace oracle. theta = (beta, [log sigma], a0, a1, eta_rho).
function _lord_oracle(theta, family, y, x, z0, z1, gidx, G)
    n = length(y)
    off = family isa Poisson ? 0 : 1
    beta = theta[1:2]
    a0, a1 = theta[(3 + off):(4 + off)], theta[(5 + off):(6 + off)]
    rho = _lord_rho(theta[7 + off])
    P = inv(Symmetric(kron([1.0 rho; rho 1.0], Matrix{Float64}(I, G, G))))
    eta0 = hcat(ones(n), x) * beta
    l0 = exp.(hcat(ones(n), z0) * a0)
    l1 = x .* exp.(hcat(ones(n), z1) * a1)
    function joint(u)
        value = 0.5 * dot(u, P * u)
        for i in 1:n
            eta = eta0[i] + l0[i] * u[gidx[i]] + l1[i] * u[G + gidx[i]]
            if family isa Poisson
                value -= Distributions.logpdf(Distributions.Poisson(exp(eta)), Int(y[i]))
            else
                r = exp(-2 * theta[3])
                value -= Distributions.logpdf(
                    Distributions.NegativeBinomial(r, r / (r + exp(eta))), Int(y[i]))
            end
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(2G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 1000); autodiff = :forward)
    u = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, u)
    return joint(u) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Matrix(P)))
end

@testset "labelled ordinary q2" begin
    fx = _lord_fixture()
    data = fx.data
    Aid = Matrix{Float64}(I, fx.G, fx.G)
    for (family, resp) in ((Poisson(), :yp), (NegBinomial2(), :yn))
        fname = nameof(typeof(family))
        @testset "$fname" begin
            fit = drm(_lord_formula(resp, family), family; data)
            @test fit.converged
            @test first(fit.blocks[end]) === :recov
            @test fit.coefnames[end] == (:recov => ["g:atanh_rho"])
            theta = copy(fit.theta)
            y = getproperty(data, resp)
            oracle = _lord_oracle(theta, family, y, data.x, data.z0, data.z1, fx.gi, fx.G)
            @test -loglikelihood(fit) ≈ oracle atol = 1e-4
            @test abs(_lord_rho(theta[end]) - 0.5) < 0.45
            @test all(isfinite, stderror(fit))

            # rho = 0 restriction: the independent identity-kernel fit.
            fit0 = drm(_lord_indep_formula(resp, family), family; data, A = Aid, se = false)
            @test length(fit0.theta) == length(theta) - 1
            @test loglikelihood(fit) >= loglikelihood(fit0) - 1e-6
            @test fit.nll(vcat(fit0.theta, 0.0)) ≈ -loglikelihood(fit0) atol = 1e-6

            # Prediction parses the labelled formula.
            beta = coef(fit, :mu)
            nd = (; x = [-0.5, 0.0, 0.8], g = fill(first(data.g), 3), z0 = zeros(3), z1 = zeros(3))
            eta_nd = hcat(ones(3), nd.x) * beta
            @test predict(fit, nd; type = :link) ≈ eta_nd atol = 1e-10
            @test predict_parameters(fit, nd)[:mu] ≈ exp.(eta_nd) atol = 1e-10

            # Marginal simulator draws (u0, u1) with the fitted rho: E[y] = exp(eta_f + var(b)/2).
            rho = _lord_rho(theta[end])
            off = family isa Poisson ? 0 : 1
            s0 = exp.(hcat(ones(length(y)), data.z0) * theta[(3 + off):(4 + off)])
            s1 = exp.(hcat(ones(length(y)), data.z1) * theta[(5 + off):(6 + off)])
            eta_f = hcat(ones(length(y)), data.x) * theta[1:2]
            Ey = exp.(eta_f .+ 0.5 .* (s0 .^ 2 .+ data.x .^ 2 .* s1 .^ 2 .+
                                       2 * rho .* data.x .* s0 .* s1))
            Y = simulate(fit; nsim = 4000, rng = MersenneTwister(11), data)
            ym = vec(mean(Y; dims = 2))
            for rows in (findall(>(0.9), data.x), findall(<(-0.9), data.x))
                @test abs(mean(ym[rows]) / mean(Ey[rows]) - 1) < 0.07
            end

            # Bridge: the R-shaped formula string reaches the same fit and labels it.
            fam_s = family isa Poisson ? "poisson" : "nbinom2"
            sig_s = family isa Poisson ? "" : "sigma ~ 1; "
            br = drm_bridge(; formula = "$resp ~ x + (1 + x | p | g); $sig_s" *
                            "sd(g, coef(intercept)) ~ z0; sd(g, coef(x)) ~ z1",
                            family = fam_s, data, newdata = nd)
            @test br["converged"]
            @test br["coefficients"] ≈ coef(fit) atol = 1e-6
            @test length(unique(br["coef_names"])) == length(br["coef_names"])
        end
    end

    @testset "Poisson unlabelled (1 + x | g) is the same correlated model" begin
        fl = drm(_lord_formula(:yp, Poisson()), Poisson(); data, se = false)
        fu = drm(_lord_formula(:yp, Poisson(); lhs = "1 + x | g"), Poisson(); data, se = false)
        @test fu.coefnames == fl.coefnames
        @test loglikelihood(fu) ≈ loglikelihood(fl) atol = 1e-8
        @test fu.theta ≈ fl.theta atol = 1e-6
    end

    @testset "NB2 labelled agrees with the unlabelled two-field route" begin
        # The two routes differ only in the rho bound (tanh vs 0.999999 tanh).
        fl = drm(_lord_formula(:yn, NegBinomial2()), NegBinomial2(); data, se = false)
        fu = drm(_lord_formula(:yn, NegBinomial2(); lhs = "1 + x | g"), NegBinomial2();
                 data, se = false)
        @test loglikelihood(fl) ≈ loglikelihood(fu) atol = 1e-6
        @test fl.theta ≈ fu.theta atol = 2e-3
    end

    @testset "refusals stay explicit" begin
        one_sd = bf(@formula(yp ~ x + (1 + x | p | g)), @formula(sd(g, coef(intercept)) ~ z0))
        # One SD formula is accepted since the non-Gaussian correlated-block change: the other
        # coefficient keeps a constant SD (the explicit `~ 1` form is the same model).
        both = bf(@formula(yp ~ x + (1 + x | p | g)), @formula(sd(g, coef(intercept)) ~ z0),
                  @formula(sd(g, coef(x)) ~ 1))
        @test loglikelihood(drm(one_sd, Poisson(); data, se = false)) ≈
              loglikelihood(drm(both, Poisson(); data, se = false)) atol = 1e-6
        @test_throws ArgumentError drm(_lord_formula(:yp, Poisson()), Poisson();
                                       data, A = Aid, se = false)
        # Gamma ordinary correlated slopes stay refused (native drmTMB refuses them too).
        gdata = merge(data, (; yg = data.yp .+ 1.0))
        @test_throws Exception drm(_lord_formula(:yg, Gamma()), Gamma(); data = gdata, se = false)
        # NB2 needs a fixed `sigma ~ 1`.
        nb_sig = bf(@formula(yn ~ x + (1 + x | p | g)), @formula(sigma ~ z0),
                    @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1))
        @test_throws ArgumentError drm(nb_sig, NegBinomial2(); data, se = false)
        # SD targets must select the intercept and the slope once each.
        wrong = bf(@formula(yp ~ x + (1 + x | p | g)), @formula(sd(g, coef(intercept)) ~ z0),
                   @formula(sd(g, coef(z0)) ~ z1))
        @test_throws ArgumentError drm(wrong, Poisson(); data, se = false)
        # An extra random effect is refused, not silently dropped.
        extra = bf(@formula(yp ~ x + (1 + x | p | g) + (1 | z0)),
                   @formula(sd(g, coef(intercept)) ~ z0), @formula(sd(g, coef(x)) ~ z1))
        @test_throws ArgumentError drm(extra, Poisson(); data, se = false)
    end
end


@testset "labelled ordinary q2: simulate(newdata) is population level ($resp)" for
        (resp, fam) in ((:yp, Poisson()), (:yn, NegBinomial2()))
    fx = _lord_fixture()
    fit = drm(_lord_formula(resp, fam), fam; data = fx.data)
    nd = (; x = [-0.5, 0.0], z0 = [1.0, -1.0], z1 = [0.3, 0.2], g = ["site_1", "site_2"])
    S = simulate(fit; newdata = nd, nsim = 4000, rng = MersenneTwister(1))
    @test size(S) == (2, 4000)
    b = coef(fit, :mu)
    @test vec(mean(S; dims = 2)) ≈ exp.(b[1] .+ b[2] .* nd.x) rtol = 0.1
end
