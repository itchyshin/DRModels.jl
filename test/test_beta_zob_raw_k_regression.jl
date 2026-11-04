# Beta and ZeroOneBeta relmat/animal intercept fits, and the Beta animal
# sd(group, animal) SD regression, use the supplied K / A AS SUPPLIED (raw), as native
# drmTMB does. K has a NON-UNIT diagonal (0.5 to 2.5) and correlated off-diagonals;
# the rows are shuffled so the first-seen group order differs from the label order.
# Each fit's objective at its own optimum is compared with an independent dense Laplace
# oracle coded here from the RAW K. Emulated bugs (K rescaled to a correlation matrix,
# K left in label order, reversed group SD vector, bootstrap draws from the correlation
# factor) must move the oracle or the draw far beyond the tolerance.
using DRModels, Test, Random, LinearAlgebra
import Distributions
using SpecialFunctions: loggamma

# -log marginal (Laplace) for b ~ N(0, D K D), D = diag(sd), eta = Xmu*beta + b[g].
# `family` is :beta or :zob (interior Beta row minus log(1 - zoi); atoms constant).
function _b29_oracle(family::Symbol, y, Xmu, beta, logsigma, sdvec, gidx, K;
                     zoi = 0.0, coi = 0.5)
    G = size(K, 1)
    Dinv = Diagonal(1 ./ sdvec)
    Qtot = Matrix(Dinv * inv(Symmetric(K)) * Dinv)
    eta0 = Xmu * beta
    phi = exp(-2 * logsigma)
    function nll_i(yi, eta)
        if family === :zob && (yi == 0 || yi == 1)
            return yi == 1 ? -(log(zoi) + log(coi)) : -(log(zoi) + log1p(-coi))
        end
        mu = 1 / (1 + exp(-eta))
        a = mu * phi
        b = (1 - mu) * phi
        v = -(loggamma(phi) - loggamma(a) - loggamma(b) +
              (a - 1) * log(yi) + (b - 1) * log1p(-yi))
        return family === :zob ? v - log1p(-zoi) : v
    end
    function joint(b)
        value = 0.5 * dot(b, Qtot * b)
        for i in eachindex(y)
            value += nll_i(y[i], eta0[i] + b[gidx[i]])
        end
        value
    end
    res = DRModels.Optim.optimize(joint, zeros(G), DRModels.Optim.LBFGS(),
        DRModels.Optim.Options(g_tol = 1e-10, iterations = 1000); autodiff = :forward)
    mode = DRModels.Optim.minimizer(res)
    H = DRModels.ForwardDiff.hessian(joint, mode)
    return joint(mode) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Qtot)
end

function _b29_fixture(seed; sdtrue = 0.8, G = 24, m = 8, groupsd = nothing)
    rng = MersenneTwister(seed)
    R = 0.5 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = sqrt.(range(0.5, 2.5; length = G))          # K diagonal 0.5 ... 2.5
    Kbase = Diagonal(scale) * R * Diagonal(scale)        # in label order 1..G
    names = ["g" * lpad(string(i), 2, '0') for i in 1:G]
    rows = shuffle(rng, repeat(1:G, inner = m))          # shuffled group order
    labels = names[rows]
    order = unique(rows)                                 # first-seen order of labels
    K = Matrix(Kbase[order, order])                      # K in first-seen order
    gidx = [findfirst(==(r), order) for r in rows]
    sdg = groupsd === nothing ? fill(sdtrue, G) : groupsd[1:G]   # label order
    b = sdg .* (cholesky(Symmetric(Kbase)).L * randn(rng, G))
    x = randn(rng, length(rows))
    eta = 0.3 .+ 0.25 .* x .+ b[rows]
    phi = 1 / 0.35^2
    mu = 1  ./ (1 .+ exp.(-eta))
    y = Float64[rand(rng, Distributions.Beta(v * phi, (1 - v) * phi)) for v in mu]
    y = clamp.(y, nextfloat(0.0), prevfloat(1.0))
    return (; rng, labels, K, Kbase, order, gidx, rows, x, y, names, G)
end

@testset "Beta relmat/animal(1 | g) intercept uses raw K (dense Laplace oracle)" begin
    fx = _b29_fixture(20261103)
    data = (; y = fx.y, x = fx.x, g = fx.labels)
    Xmu = hcat(ones(length(fx.y)), fx.x)
    for (kw, marker) in ((:K, :relmat), (:A, :animal))
        formula = marker === :relmat ? bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1)) :
                                       bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1))
        fit = drm(formula, DRModels.Beta(); data, (kw => fx.K,)..., se = false)
        @test fit.converged
        theta = coef(fit)
        sd = exp(theta[end])
        oracle = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], fill(sd, fx.G), fx.gidx, fx.K)
        @test fit.nll(theta) ≈ oracle atol = 2e-3
        # emulated bugs: correlation-rescaled K, and K left in label order
        d = sqrt.(diag(fx.K))
        wrong_scale = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], fill(sd, fx.G),
                                  fx.gidx, fx.K ./ (d * d'))
        @test abs(wrong_scale - oracle) > 0.05
        wrong_order = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], fill(sd, fx.G),
                                  fx.gidx, fx.Kbase)
        @test abs(wrong_order - oracle) > 0.05
    end
end

@testset "ZeroOneBeta relmat/animal(1 | g) intercept uses raw K (dense Laplace oracle)" begin
    fx = _b29_fixture(20261104; sdtrue = 0.8, m = 10)
    rng = MersenneTwister(20261105)
    y = copy(fx.y)
    atom = rand(rng, length(y)) .< 0.15
    y[atom] .= ifelse.(rand(rng, count(atom)) .< 0.5, 0.0, 1.0)
    nb = count(atom)
    zoi = nb / length(y)
    coi = count(==(1.0), y) / nb
    data = (; y, x = fx.x, g = fx.labels)
    Xmu = hcat(ones(length(y)), fx.x)
    for (kw, marker) in ((:K, :relmat), (:A, :animal))
        formula = marker === :relmat ?
            bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)) :
            bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1))
        fit = drm(formula, DRModels.ZeroOneBeta(); data, (kw => fx.K,)..., se = false)
        @test fit.converged
        theta = coef(fit)[1:4]                      # mu(2), sigma, resd; zoi/coi are closed form
        sd = exp(theta[4])
        oracle = _b29_oracle(:zob, y, Xmu, theta[1:2], theta[3], fill(sd, fx.G), fx.gidx, fx.K;
                             zoi, coi)
        @test fit.nll(theta) ≈ oracle atol = 2e-3
        d = sqrt.(diag(fx.K))
        wrong_scale = _b29_oracle(:zob, y, Xmu, theta[1:2], theta[3], fill(sd, fx.G),
                                  fx.gidx, fx.K ./ (d * d'); zoi, coi)
        @test abs(wrong_scale - oracle) > 0.05
        wrong_order = _b29_oracle(:zob, y, Xmu, theta[1:2], theta[3], fill(sd, fx.G),
                                  fx.gidx, fx.Kbase; zoi, coi)
        @test abs(wrong_order - oracle) > 0.05
    end
end

@testset "Beta animal sd(g, animal) ~ z uses raw A; simulator draws from raw A" begin
    G = 24
    zlab = repeat([-1.0, 1.0], G ÷ 2)                    # label order
    fx = _b29_fixture(20261106; G, m = 8, groupsd = exp.(log(0.5) .+ 0.5 .* zlab))
    zrow = zlab[fx.rows]
    data = (; y = fx.y, x = fx.x, z = zrow, g = fx.labels)
    formula = bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z))
    fit = drm(formula, DRModels.Beta(); data, A = fx.K, se = false)
    @test fit.converged
    Xmu = hcat(ones(length(fx.y)), fx.x)
    zfirst = zlab[fx.order]                              # z per group, first-seen order
    theta = copy(fit.theta)
    alpha = theta[4:5]
    sdvec = exp.(alpha[1] .+ alpha[2] .* zfirst)
    oracle = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], sdvec, fx.gidx, fx.K)
    @test fit.nll(theta) ≈ oracle atol = 2e-3
    d = sqrt.(diag(fx.K))
    wrong_scale = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], sdvec, fx.gidx,
                              fx.K ./ (d * d'))
    @test abs(wrong_scale - oracle) > 0.05
    wrong_sd = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], reverse(sdvec), fx.gidx, fx.K)
    @test abs(wrong_sd - oracle) > 0.05
    wrong_order = _b29_oracle(:beta, fx.y, Xmu, theta[1:2], theta[3], sdvec, fx.gidx, fx.Kbase)
    @test abs(wrong_order - oracle) > 0.05

    # fit, predict-side group SD, and bootstrap simulator agree on the raw scale
    simulator = DRModels._marginal_simulator_build(fit, data; A = fx.K)
    @test simulator !== nothing
    draw = simulator(MersenneTwister(77))
    rng = MersenneTwister(77)
    eta = predict(fit, data; type = :link)
    gsd = sdvec
    make(factor) = begin
        r = MersenneTwister(77)
        u = gsd .* (factor * randn(r, fx.G))
        mu = 1 ./ (1 .+ exp.(-clamp.(eta .+ u[fx.gidx], -30.0, 30.0)))
        DRModels._simulate_once(fit, r; mu)
    end
    @test draw == make(cholesky(Symmetric(fx.K)).L)
    @test norm(draw .- make(cholesky(Symmetric(fx.K ./ (d * d'))).L)) > 0.1
end
