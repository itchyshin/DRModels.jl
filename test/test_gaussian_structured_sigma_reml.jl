# REML for the Gaussian structured sigma intercept with an SD formula:
#   y ~ x,  sigma ~ [1 | x] + animal|relmat|phylo|spatial(1 | id),  sd(id, level, sigma) ~ z,  method = :REML
# Native drmTMB (REML = TRUE) fits these by moving beta_mu AND beta_sigma into TMB's random block (one
# Laplace approximation over (v, beta_mu, beta_sigma), flat priors on the betas), reporting
# `estimator_exact = FALSE`. See docs/design/gaussian-structured-sigma-reml.md.
#
# Two checks, thresholds fixed BEFORE the five fixtures were fitted:
#   (1) agreement with native drmTMB 0.8.1 REML on five committed-seed datasets (the numbers are
#       `_GSR_R`, produced by test/fixtures/gaussian_structured_sigma_reml_r_reference.R from the
#       data of `gsr_fixture`): |logLik_J - logLik_R| < 1e-6; every coefficient within 1e-4 R-SEs;
#       every Julia SE within 1e-3 (relative) of the R SE (R-side SEs are the sdreport covariance of
#       the fixed + integrated coefficients, H^-1 + J V J').
#   (2) an independent dense oracle (own Newton, ForwardDiff derivatives, no DRModels internals):
#       reml_loglik and ml_loglik equal the oracle's restricted / ML Laplace values to 1e-8, and
#       the restricted value at the fitted SD coefficients is not beaten by perturbing them.
using DRModels, Test, LinearAlgebra, ForwardDiff

include(joinpath(@__DIR__, "fixtures", "gaussian_structured_sigma_reml_fixture.jl"))

const _GSR_R = Dict(
    :animal => (ll = -246.8411694911,
        est = [0.2074227661, 0.5003678728, -0.7490456702, -0.9678793593, 1.2099227301],
        se = [0.0187639846, 0.0177021075, 0.0822602262, 0.2386550936, 0.3749914786]),
    :relmat => (ll = -216.1615389419,
        est = [0.2309577916, 0.4775972424, -0.7911921754, -0.8306184746, 0.4977161558],
        se = [0.0193007213, 0.0193023009, 0.1216019400, 0.1934439317, 0.2791439455]),
    :phylo => (ll = -277.9428048145,
        est = [0.1895180569, 0.4961212678, -0.6676480280, -0.8127208818, 0.1771748306],
        se = [0.0269867468, 0.0228527608, 0.1298213204, 0.1657303486, 0.2689521694]),
    :spatial => (ll = -327.8667075420,
        est = [0.1741483673, 0.5219578251, -0.4894701843, -0.8140094900, 0.3441391138],
        se = [0.0322962683, 0.0341702005, 0.1971952947, 0.2161965315, 0.2803360257]),
    :relmat_sx => (ll = -265.7748552297,
        est = [0.2126615732, 0.5149575765, -0.5452798541, 0.3300045906, -0.7174874573, 0.6544851179],
        se = [0.0244327121, 0.0181860009, 0.1319312228, 0.0441416653, 0.1870446368, 0.2958646902]),
)

function _gsr_fit(fx; method = :REML)
    d = fx.data
    c = fx.cell
    fm = c === :animal ? bf(@formula(y ~ x), @formula(sigma ~ 1 + animal(1 | id)), @formula(sd(id, animal, sigma) ~ z)) :
         c === :relmat ? bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)), @formula(sd(id, relmat, sigma) ~ z)) :
         c === :relmat_sx ? bf(@formula(y ~ x), @formula(sigma ~ x + relmat(1 | id)), @formula(sd(id, relmat, sigma) ~ z)) :
         c === :phylo ? bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 | id)), @formula(sd(id, phylogenetic, sigma) ~ z)) :
                        bf(@formula(y ~ x), @formula(sigma ~ 1 + spatial(1 | id)), @formula(sd(id, spatial, sigma) ~ z))
    kw = c === :animal ? (; A = fx.K, ids = fx.ids) : (c === :relmat || c === :relmat_sx) ? (; K = fx.K) :
         c === :phylo ? (; tree = fx.tree) : (; coords = fx.coords)
    return drm(fm, Gaussian(); data = d, method, kw...)
end

# Dense oracle for the restricted (reml = true) or ML Laplace value at the given parameters.
function _gsr_oracle(fx, beta, gamma, alpha; reml::Bool)
    d = fx.data; n = length(d.y); G = length(fx.ids)
    gi = [parse(Int, id[2:end]) for id in d.id]
    zg = zeros(G); zg[gi] .= d.z
    s = exp.(alpha[1] .+ alpha[2] .* zg)
    Xs = size(gamma, 1) == 2 ? hcat(ones(n), d.x) : ones(n, 1)
    Kinv = inv(Symmetric(fx.K)); ldK = logdet(cholesky(Symmetric(fx.K)))
    p = length(beta) + length(gamma)
    function negjoint(w)
        v = w[1:G]
        b = reml ? w[G+1:G+2] : beta
        g = reml ? w[G+3:end] : gamma
        eta = Xs * g .+ s[gi] .* v[gi]
        r = d.y .- (b[1] .+ b[2] .* d.x)
        return sum(0.5 * log(2π) .+ eta .+ 0.5 .* r .^ 2 .* exp.(-2 .* eta)) + 0.5 * dot(v, Kinv * v)
    end
    w = reml ? vcat(zeros(G), beta, gamma) : zeros(G)
    for _ in 1:200
        g = ForwardDiff.gradient(negjoint, w)
        maximum(abs, g) < 1e-9 && break
        H = ForwardDiff.hessian(negjoint, w)
        step = H \ g
        # Inside the quadratic basin take the full Newton step. The Armijo test below compares f values near
        # 270 whose true decrease is under roundoff, so on some CPU/SIMD paths it rejects every step and the
        # loop stalls at |g| ~ 1e-7; the Laplace value moves at first order in that gap (through log det H).
        if maximum(abs, g) < 1e-4
            w .-= step
            continue
        end
        t = 1.0; f0 = negjoint(w)
        while t > 1e-12 && negjoint(w .- t .* step) > f0; t /= 2; end
        w .-= t .* step
    end
    H = ForwardDiff.hessian(negjoint, w)
    @assert maximum(abs, ForwardDiff.gradient(negjoint, w)) < 1e-8   # oracle's own mode, not a model tolerance
    nll = negjoint(w) + 0.5 * logdet(cholesky(Symmetric(H))) + 0.5 * ldK - (reml ? 0.5 * p * log(2π) : 0.0)
    return -nll
end

@testset "Gaussian structured sigma intercept: REML" begin
    for cell in (:animal, :relmat, :phylo, :spatial, :relmat_sx)
        @testset "$cell" begin
            fx = gsr_fixture(cell)
            fit = _gsr_fit(fx)
            R = _GSR_R[cell]
            @test fit.converged
            @test estimation_method(fit) === :REML
            se = sqrt.(diag(fit.vcov))
            # (1) native drmTMB REML
            @test abs(reml_loglik(fit) - R.ll) < 1e-6
            @test abs(loglik(fit) - R.ll) < 1e-6
            @test all(abs.(fit.theta .- R.est) ./ R.se .< 1e-4)
            @test all(abs.(se ./ R.se .- 1) .< 1e-3)
            # (2) dense oracle
            nb = length(coef(fit, :mu)); ns = length(coef(fit, :sigma))
            beta = fit.theta[1:nb]; gamma = fit.theta[nb+1:nb+ns]; alpha = fit.theta[nb+ns+1:end]
            lr = _gsr_oracle(fx, beta, gamma, alpha; reml = true)
            @test abs(reml_loglik(fit) - lr) < 1e-8
            @test abs(ml_loglik(fit) - _gsr_oracle(fx, beta, gamma, alpha; reml = false)) < 1e-8
            for k in 1:2, δ in (-0.05, 0.05)
                a = copy(alpha); a[k] += δ
                @test _gsr_oracle(fx, beta, gamma, a; reml = true) <= lr + 1e-8
            end
        end
    end

    @testset "ordinary sigma random effects keep refusing REML" begin
        fx = gsr_fixture(:relmat)
        d = fx.data
        @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | id)),
            @formula(sd(id, sigma) ~ z)), Gaussian(); data = d, method = :REML)
    end
end
