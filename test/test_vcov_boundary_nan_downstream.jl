# Accessors on fits with a flagged boundary coordinate (DRModels.jl#1383).
#
# `vcov` is NaN at a flagged coordinate and `stderror` is NaN there (not computed). Every
# quantity that does not use the flagged coordinate must keep its own standard error:
# `0 * NaN` is NaN, so a product with the full matrix would lose it. Six fits, one per
# route: a tiny covariate (an unidentified slope) in a Gaussian and a Gamma fit, a random
# intercept variance at zero in a Binomial and a Poisson Laplace fit, a Tweedie power at
# its clamp, and a bivariate structured fit with a rank-one among-group covariance.

using DRModels
using Test
using Random
using LinearAlgebra
using Logging
using StableRNGs
import Distributions

_quiet(f) = Logging.with_logger(f, Logging.NullLogger())

# Shared checks: the coordinate is flagged (public accessor `bridge_diagnostics(fit).boundary`),
# `vcov` is NaN exactly there, `stderror` is NaN there and finite elsewhere, the Wald interval
# is unbounded there, and the unflagged block is finite and symmetric.
function _check_flagged(fit, k)
    p = length(fit.theta)
    keep = setdiff(1:p, k)
    @test bridge_diagnostics(fit).boundary == [k]
    V = vcov(fit)
    @test all(isnan, V[k, :]) && all(isnan, V[:, k])
    @test all(isfinite, V[keep, keep]) && issymmetric(V[keep, keep])
    se = stderror(fit)
    @test isnan(se[k])
    @test all(isfinite, se[keep])
    ci = confint(fit; method = :wald)
    @test ci[k].lower == -Inf && ci[k].upper == Inf
    @test all(r -> isfinite(r.lower) && isfinite(r.upper), ci[keep])
    out = sprint(show, MIME"text/plain"(), fit)
    @test occursin("Wald SEs withheld", out)
    ct = coeftable(fit)
    @test length(ct.rownms) == p
end

@testset "boundary coordinate: all accessors, six fits" begin
    @testset "Gaussian, tiny covariate in the mean" begin
        rng = Random.MersenneTwister(20260715)
        n_id, n_each = 80, 6
        sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
        b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
        id = repeat(1:n_id, inner = n_each); sexl = sex[id]
        y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+ randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
        z = 1e-5 .* randn(rng, n_id * n_each)
        dat = (; y, sex = sexl, id, z)
        fit = _quiet() do
            drm(bf(@formula(y ~ sex + z + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex)),
                Gaussian(); data = dat)
        end
        _check_flagged(fit, 3)
        # predict with se: rows with z = 0 do not use the flagged slope
        nd = (; sex = [0.0, 1.0, 0.0], z = [0.0, 0.0, 1.0], id = [1, 2, 3])
        se = predict(fit, nd; se = true).se
        @test isfinite(se[1]) && isfinite(se[2]) && isnan(se[3])
        @test se[1] ≈ sqrt(vcov(fit)[1, 1]) rtol = 1e-12
        sp = predict_parameters(fit, nd; se = true)
        @test isfinite(sp[:mu].se[1]) && isnan(sp[:mu].se[3])
        # the delta-method ratio keeps its standard error for rows free of the coordinate
        r = repeatability(fit, (; sex = [0.0, 1.0, 0.5]))
        @test all(isfinite, r.se_logit) && r.status === :ok
        # a quantity that does not involve it keeps its standard error
        @test isfinite(bias_correct(fit, th -> th[1] + th[2]).se)
    end

    @testset "Gamma, tiny covariate in the mean" begin
        rng = StableRNG(13)
        n = 150
        x = randn(rng, n); z = 1e-5 .* randn(rng, n)
        y = [rand(rng, Distributions.Gamma(4.0, exp(0.8 + 0.2 * x[i]) / 4.0)) for i in 1:n]
        dat = (; y, x, z)
        fit = _quiet() do
            drm(bf(@formula(y ~ x + z), @formula(sigma ~ 1)), Gamma(); data = dat)
        end
        _check_flagged(fit, 3)
        se = predict(fit, (; x = [0.0, 1.0, 0.0], z = [0.0, 0.0, 1.0]); se = true).se
        @test isfinite(se[1]) && isfinite(se[2]) && isnan(se[3])
    end

    xi = [1.43941117899116, 0.778826426368403, 0.192743160380563, -0.3244252191031, 2.09766874641634, 0.369686510094048, 1.79478191004505, 0.606502031264434, 0.0312916522780309, -0.910572582677395, 0.806725615836395, 0.612734177592478, -0.611163641309853, -0.159343666027289, 1.05500813010401, -1.06458870705857, 1.12065628331957, 0.387953130608195, -0.224004141372002, -0.109023652368118]
    g = repeat(["g01", "g02", "g03", "g04"], 5)
    s = [4, 6, 4, 7, 6, 3, 4, 6, 3, 5, 5, 6, 5, 7, 7, 5, 6, 5, 2, 5]
    fl = [4, 2, 4, 1, 2, 5, 4, 2, 5, 3, 3, 2, 3, 1, 1, 3, 2, 3, 6, 3]

    @testset "Binomial, Laplace, group SD at zero" begin
        d = (; x = xi, g, s, fl)
        fit = _quiet() do
            drm(bf(@formula(cbind(s, fl) ~ x + (1 | g))), Binomial(); data = d, marginal = :Laplace)
        end
        _check_flagged(fit, 3)
        pe = predict(fit, (; x = [0.0, 1.0], g = ["g01", "g02"], n = [8, 8]); se = true)
        @test all(isfinite, pe.se)
    end

    @testset "Poisson, Laplace, group SD at zero" begin
        d = (; x = xi, g, y = s)
        fit = _quiet() do
            drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = d, marginal = :Laplace)
        end
        _check_flagged(fit, 3)
        pe = predict(fit, (; x = [0.0, 1.0], g = ["g01", "g02"]); se = true)
        @test all(isfinite, pe.se)
    end

    @testset "Tweedie, power at its clamp" begin
        n = 200
        x = randn(MersenneTwister(3), n)
        d = (; y = vcat(1.3, zeros(n - 1)), x)
        fit = _quiet() do
            drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = d)
        end
        _check_flagged(fit, 4)
        pe = predict(fit, (; x = [0.0, 1.0]); se = true)
        @test all(isfinite, pe.se)
    end

    @testset "bivariate structured, rank-one among-group covariance" begin
        rng = MersenneTwister(20260905)
        G = 16; nrep = 4
        K = Matrix{Float64}(I, G, G)
        group = repeat(1:G, inner = nrep)
        n = length(group)
        x = randn(rng, n)
        a = 0.6 .* randn(rng, G)
        dat = (; y1 = 0.3 .+ 0.2 .* x .+ a[group] .+ 0.35 .* randn(rng, n),
                 y2 = -0.1 .+ 0.4 .* x .+ a[group] .+ 0.35 .* randn(rng, n),
                 x, grp = ["g$(i)" for i in group])
        form = bf(mu1 = @formula(y1 ~ x + relmat(1 | grp)),
                  mu2 = @formula(y2 ~ x + relmat(1 | grp)),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        fit = _quiet() do
            drm(form, Gaussian(); data = dat, K = K, g_tol = 1e-6, method = :ML)
        end
        _check_flagged(fit, 10)
        # the location block is unaffected by the flagged coordinate
        @test all(isfinite, stderror(fit)[1:9])
        @test isfinite(bias_correct(fit, th -> th[1] - th[2]).se)
    end
end
