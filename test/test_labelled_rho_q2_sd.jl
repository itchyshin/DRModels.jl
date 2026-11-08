# Labelled (correlated) structured intercept-plus-slope q2 with one SD formula
# per coefficient, `animal|relmat(1 + x | p | g)`, on the Poisson, NB2 and
# Gaussian routes. drmTMB contract: unit-scale fields (u0, u1) ~ N(0, R (x) C),
# R = [1 rho; rho 1], rho = 0.999999 tanh(eta_cor_phylo), b_k = s_k[g] u_k,
# C RAW. Oracles share no fitter code: a dense ForwardDiff Laplace with the full
# kron(R, C) prior (counts) and a dense MvNormal log density (Gaussian), both at
# the fitted theta. Tolerances fixed before the first run: nll vs oracle
# atol 1e-4 (Laplace) / 1e-6 (Gaussian); rho recovery |rho_hat - 0.5| < 0.45
# (one dataset, no calibration claim); labelled loglik >= unlabelled - 1e-6.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

# Shared latent fields with a free correlation: unit fields (u0, u1) ~ N(0, R (x) A),
# R = [1 rho; rho 1]; b0 = s0[g] u0, b1 = s1[g] u1 (drmTMB's q2 covariance contract).
function _lrho_fixture(; G = 48, m = 24, seed = 202611073, rho = 0.5, sigma_nb = 0.45,
                       sigma_g = 0.5)
    rng = MersenneTwister(seed)
    ids = ["animal_$(i)" for i in randperm(rng, G)]
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.4; length = G))
    A = Matrix(Diagonal(scale) * R * Diagonal(scale))
    group = repeat(ids, inner = m)
    gidx = repeat(1:G, inner = m)
    z0group = ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0)
    z1group = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    alpha0 = [-0.7, 0.35]
    alpha1 = [-0.9, 0.45]
    L = cholesky(Symmetric(A)).L
    n0, n1 = randn(rng, G), randn(rng, G)
    u0 = L * n0
    u1 = L * (rho .* n0 .+ sqrt(1 - rho^2) .* n1)
    eta = 0.2 .+ 0.3 .* x .+
        (exp.(alpha0[1] .+ alpha0[2] .* z0group) .* u0)[gidx] .+
        x .* (exp.(alpha1[1] .+ alpha1[2] .* z1group) .* u1)[gidx]
    yp = Float64.([rand(rng, Distributions.Poisson(exp(v))) for v in eta])
    r = 1 / sigma_nb^2
    yn = Float64.([rand(rng, Distributions.NegativeBinomial(r, r / (r + exp(v)))) for v in eta])
    yg = eta .+ sigma_g .* randn(rng, length(eta))
    return (; data = (; yp, yn, yg, x, z0 = z0group[gidx], z1 = z1group[gidx], g = group),
            A, gidx, z0group, z1group, alpha0, alpha1, rho)
end

_lrho_rho(eta) = 0.999999 * tanh(eta)

function _lrho_formula(resp::Symbol, kind::Symbol, family; labelled = true)
    lhs = labelled ? "1 + x | p | g" : "1 + x | g"
    mu = eval(Meta.parse("@formula($resp ~ x + $kind($lhs))"))
    sd0 = eval(Meta.parse("@formula(sd(g, $kind, coef(intercept)) ~ z0)"))
    sd1 = eval(Meta.parse("@formula(sd(g, $kind, coef(x)) ~ z1)"))
    family isa Poisson && return bf(mu, sd0, sd1)
    return bf(mu, @formula(sigma ~ 1), sd0, sd1)
end

# Dense Laplace oracle for counts. theta = (beta, [log sigma], a0, a1, eta_rho).
function _lrho_count_oracle(theta, family, y, x, z0, z1, gidx, C)
    G, n = size(C, 1), length(y)
    off = family isa Poisson ? 0 : 1
    beta = theta[1:2]
    a0, a1 = theta[(3 + off):(4 + off)], theta[(5 + off):(6 + off)]
    rho = _lrho_rho(theta[7 + off])
    R = [1.0 rho; rho 1.0]
    P = inv(Symmetric(kron(R, C)))          # prior precision of (u0; u1)
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

# Dense MvNormal oracle for Gaussian. theta = (beta, log sigma, a0, a1, eta_rho).
function _lrho_gaussian_oracle(theta, y, x, z0, z1, gidx, C)
    G, n = size(C, 1), length(y)
    rho = _lrho_rho(theta[8])
    L = zeros(n, 2G)
    d0 = exp.(hcat(ones(n), z0) * theta[4:5]); d1 = exp.(hcat(ones(n), z1) * theta[6:7])
    for i in 1:n
        L[i, gidx[i]] = d0[i]; L[i, G + gidx[i]] = x[i] * d1[i]
    end
    V = exp(2 * theta[3]) * Matrix(I, n, n) + L * kron([1.0 rho; rho 1.0], C) * L'
    return -Distributions.logpdf(Distributions.MvNormal(hcat(ones(n), x) * theta[1:2],
                                                        Symmetric(V)), y)
end

@testset "labelled q2 ($(nameof(typeof(family))), $kind)" for
        (family, resp) in ((Poisson(), :yp), (NegBinomial2(), :yn), (Gaussian(), :yg)),
        kind in (:animal, :relmat)
    small = family isa Gaussian
    fx = small ? _lrho_fixture(; G = 30, m = 12) : _lrho_fixture()
    data = fx.data
    kw = kind === :animal ? (; A = fx.A) : (; K = fx.A)
    f = _lrho_formula(resp, kind, family)
    fit = drm(f, family; data, kw...)
    @test fit.converged
    @test first(fit.blocks[end]) === :recov
    @test fit.coefnames[end] == (:recov => ["g:atanh_rho"])
    theta = copy(fit.theta)
    y = getproperty(data, resp)
    oracle = small ?
        _lrho_gaussian_oracle(theta, y, data.x, data.z0, data.z1, fx.gidx, fx.A) :
        _lrho_count_oracle(theta, family, y, data.x, data.z0, data.z1, fx.gidx, fx.A)
    @test -loglikelihood(fit) ≈ oracle atol = (small ? 1e-6 : 1e-4)
    @test abs(_lrho_rho(theta[end]) - fx.rho) < 0.45
    # The independent fit is the rho = 0 restriction: labelled loglik is no lower.
    sekw = small ? (;) : (; se = false)
    fit0 = drm(_lrho_formula(resp, kind, family; labelled = false), family;
               data, kw..., sekw...)
    @test loglikelihood(fit) >= loglikelihood(fit0) - 1e-6
    @test length(fit0.theta) == length(theta) - 1
    # The objective with eta_rho = 0 reproduces the independent fit's objective.
    @test fit.nll(vcat(fit0.theta, 0.0)) ≈ -loglikelihood(fit0) atol = 1e-6
    @test all(isfinite, stderror(fit))

    # Population-level prediction parses the labelled formula.
    beta = coef(fit, :mu)
    nd = (; x = [-0.5, 0.0, 0.8], g = fill(first(data.g), 3), z0 = zeros(3), z1 = zeros(3))
    eta_nd = hcat(ones(3), nd.x) * beta
    @test predict(fit, nd; type = :link) ≈ eta_nd atol = 1e-10
    @test predict_parameters(fit, nd)[:mu] ≈ (small ? eta_nd : exp.(eta_nd)) atol = 1e-10

    # Bridge: the R-shaped formula string reaches the same fit and labels it.
    fam_s = family isa Poisson ? "poisson" : family isa NegBinomial2 ? "nbinom2" : "gaussian"
    sig_s = family isa Poisson ? "" : "sigma ~ 1; "
    br = drm_bridge(; formula = "$resp ~ x + $kind(1 + x | p | g); $sig_s" *
                    "sd(g, $kind, coef(intercept)) ~ z0; sd(g, $kind, coef(x)) ~ z1",
                    family = fam_s, data, kw..., newdata = nd)
    @test br["converged"]
    @test br["coefficients"] ≈ coef(fit) atol = 1e-6
    @test length(br["coef_names"]) == length(theta)
    @test length(unique(br["coef_names"])) == length(br["coef_names"])
end

@testset "labelled q2: marginal simulator uses the fitted rho (Gaussian)" begin
    fx = _lrho_fixture(; G = 30, m = 12)
    data = fx.data
    fit = drm(_lrho_formula(:yg, :animal, Gaussian()), Gaussian(); data, A = fx.A)
    rho = _lrho_rho(fit.theta[end])
    s0 = exp.(hcat(ones(length(data.x)), data.z0) * fit.theta[4:5])
    s1 = exp.(hcat(ones(length(data.x)), data.z1) * fit.theta[6:7])
    cgg = diag(fx.A)[fx.gidx]
    v = exp(2 * fit.theta[3]) .+ cgg .* (s0 .^ 2 .+ data.x .^ 2 .* s1 .^ 2 .+
                                         2 * rho .* data.x .* s0 .* s1)
    Y = simulate(fit; nsim = 3000, rng = MersenneTwister(11), data, A = fx.A)
    vs = vec(var(Y; dims = 2))
    for rows in (findall(>(0.9), data.x), findall(<(-0.9), data.x))
        @test abs(mean(vs[rows]) / mean(v[rows]) - 1) < 0.05
    end
end

@testset "labelled q2: refusals stay explicit" begin
    fx = _lrho_fixture(; G = 30, m = 12)
    data = fx.data
    one_sd = bf(@formula(yp ~ x + animal(1 + x | p | g)),
                @formula(sd(g, animal, coef(intercept)) ~ z0))
    @test_throws ArgumentError drm(one_sd, Poisson(); data, A = fx.A, se = false)
    # Gamma now has a labelled q2 route (test_gamma_nb2_twofield_sd.jl); Binomial still refuses.
    # A labelled intercept-only structured term is not a q2 block.
    lab_int = bf(@formula(yp ~ x + animal(1 | p | g)),
                 @formula(sd(g, animal) ~ z0))
    @test_throws Exception drm(lab_int, Poisson(); data, A = fx.A, se = false)
end
