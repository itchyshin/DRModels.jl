# Binomial() contract suite: every response-domain refusal in `drm(..., Binomial())`
# with its message, and the accessor contract on a plain fixed-effect fit in both
# response forms (`cbind(successes, failures)` and a 0/1 Bernoulli vector). Every
# expected value below comes from arithmetic done in this file (an IRLS logistic
# fit, the closed-form Fisher information X'WX, hand-computed logits, Normal
# quantiles), never from the fit under test.
# Complements test_binomial.jl (recovery only) and the ranef / aghq / separation /
# clamp files. Test-only: no src/ change.
using DRModels
using Test, Random, Statistics, LinearAlgebra
import Distributions

_blogistic(η) = 1 / (1 + exp(-η))

# Logistic MLE by iteratively reweighted least squares, independent of the package
# optimiser. Row i has `ntr[i]` trials and `s[i]` successes.
function _irls_binomial(X, s, ntr)
    β = zeros(size(X, 2))
    for _ in 1:100
        μ = _blogistic.(X * β)
        W = ntr .* μ .* (1 .- μ)
        z = X * β .+ (s .- ntr .* μ) ./ W
        βn = (X' * (W .* X)) \ (X' * (W .* z))
        δ = maximum(abs.(βn .- β)); β = βn
        δ < 1e-13 && break
    end
    return β
end

@testset "Binomial contract" begin

    @testset "refusals (response domain)" begin
        Random.seed!(7)
        n = 40
        x = randn(n)
        s = Float64.(rand(0:6, n)); f = Float64.(rand(1:6, n))
        y = Float64.(rand(Bool, n))
        data = (; y, s, f, x)
        bern = @formula(y ~ x)
        cb = @formula(cbind(s, f) ~ x)
        msg1 = "Binomial() with a single-column response requires a 0/1 (Bernoulli) vector; " *
               "use cbind(successes, failures) for trial counts"
        msgc = "Binomial() requires non-negative integer successes and failures"

        # a proportion is not a valid single-column response (use cbind for counts)
        @test_throws ErrorException(msg1) drm(bf(bern), Binomial(); data = merge(data, (; y = fill(0.5, n))))
        @test_throws ErrorException(msg1) drm(bf(bern), Binomial(); data = merge(data, (; y = vcat(2.0, y[2:end]))))
        @test_throws ErrorException(msg1) drm(bf(bern), Binomial(); data = merge(data, (; y = vcat(-1.0, y[2:end]))))

        # cbind: negative or non-integer successes / failures
        @test_throws ErrorException(msgc) drm(bf(cb), Binomial(); data = merge(data, (; s = vcat(-1.0, s[2:end]))))
        @test_throws ErrorException(msgc) drm(bf(cb), Binomial(); data = merge(data, (; s = vcat(1.5, s[2:end]))))
        @test_throws ErrorException(msgc) drm(bf(cb), Binomial(); data = merge(data, (; f = vcat(-2.0, f[2:end]))))
        @test_throws ErrorException(msgc) drm(bf(cb), Binomial(); data = merge(data, (; f = vcat(0.25, f[2:end]))))

        # mismatched cbind: a failures column of the wrong length is refused (not
        # silently recycled), as is an absent column
        @test_throws Exception drm(bf(cb), Binomial(); data = merge(data, (; f = f[1:end-1])))
        @test_throws Exception drm(bf(@formula(cbind(s, nope) ~ x)), Binomial(); data = data)
    end

    # Zero-trial rows (s = f = 0) are accepted today and contribute 0 to the
    # log-likelihood (C(0,0) = 1), so the fit must equal the fit without them.
    @testset "zero-trial rows are inert (current behaviour)" begin
        Random.seed!(8)
        n = 60
        x = randn(n)
        s = Float64.(rand(1:6, n)); f = Float64.(rand(1:6, n))
        zero_rows = [3, 17, 40]
        s0 = copy(s); f0 = copy(f); s0[zero_rows] .= 0; f0[zero_rows] .= 0
        keep = setdiff(1:n, zero_rows)
        fit0 = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = (; s = s0, f = f0, x))
        fitk = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = (; s = s0[keep], f = f0[keep], x = x[keep]))
        @test coef(fit0) ≈ coef(fitk) atol = 1e-6
        @test loglik(fit0) ≈ loglik(fitk) atol = 1e-8
        @test vcov(fit0) ≈ vcov(fitk) rtol = 1e-5
    end

    @testset "accessor contract: cbind(successes, failures)" begin
        Random.seed!(20261020)
        n = 300
        x = randn(n)
        ntr = rand(5:25, n)                       # varying trials
        β = [0.3, 0.8]
        p = _blogistic.(β[1] .+ β[2] .* x)
        s = [rand(Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
        data = (; s = Float64.(s), f = Float64.(ntr .- s), x)
        X = hcat(ones(n), x)

        fit = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = data)
        k = 2
        @test is_converged(fit)

        # coefficient names and order
        @test coeftable(fit).rownms == ["mu: (Intercept)", "mu: x"]
        @test [c.param for c in confint(fit)] == [:mu, :mu]
        @test [c.coef for c in confint(fit)] == ["(Intercept)", "x"]

        # coefficients and Fisher information from an independent IRLS
        bh = _irls_binomial(X, Float64.(s), Float64.(ntr))
        @test coef(fit) ≈ bh atol = 1e-5
        @test coef(fit, :mu) ≈ bh atol = 1e-5
        μh = _blogistic.(X * bh)
        Vh = inv(Symmetric(X' * ((ntr .* μh .* (1 .- μh)) .* X)))   # canonical link: observed = expected

        V = vcov(fit)
        @test size(V) == (k, k)
        @test V ≈ V' rtol = 1e-10
        @test isposdef(Symmetric(V))
        @test V ≈ Vh rtol = 1e-4
        @test stderror(fit) ≈ sqrt.(diag(V))
        @test stderror(fit) ≈ sqrt.(diag(Vh)) rtol = 1e-4

        # Wald intervals: estimate ± z * se, z from the Normal quantile
        for lev in (0.95, 0.9, 0.99)
            z = Distributions.quantile(Distributions.Normal(), (1 + lev) / 2)
            ci = confint(fit; level = lev)
            @test [c.estimate for c in ci] ≈ coef(fit)
            @test [c.lower for c in ci] ≈ coef(fit) .- z .* stderror(fit) rtol = 1e-9
            @test [c.upper for c in ci] ≈ coef(fit) .+ z .* stderror(fit) rtol = 1e-9
        end
        w95 = [c.upper - c.lower for c in confint(fit; level = 0.95)]
        w99 = [c.upper - c.lower for c in confint(fit; level = 0.99)]
        @test all(w99 .> w95)
        @test_throws ArgumentError confint(fit; level = 1.2)

        # fitted is invlink(Xb) by hand, and response residuals are obs - fitted
        @test fitted(fit) ≈ μh atol = 1e-6
        @test fitted(fit) ≈ _blogistic.(coef(fit)[1] .+ coef(fit)[2] .* x) rtol = 1e-12
        @test residuals(fit) ≈ Float64.(s) ./ ntr .- fitted(fit) rtol = 1e-10
        r = residuals(fit; type = :quantile)
        @test length(r) == n && all(isfinite, r)
        @test_throws ArgumentError residuals(fit; type = :pearson)

        # response-scale predict_parameters / predict on new data, hand logits
        nd = (; x = [-1.0, 0.0, 2.0])
        pp = predict_parameters(fit, nd)
        @test collect(keys(pp)) == [:mu]
        @test pp[:mu] ≈ _blogistic.(coef(fit)[1] .+ coef(fit)[2] .* nd.x) rtol = 1e-12
        @test all(0 .< pp[:mu] .< 1)
        @test predict_parameters(fit, nd; type = :link)[:mu] ≈ coef(fit)[1] .+ coef(fit)[2] .* nd.x rtol = 1e-12
        @test predict(fit, nd) ≈ pp[:mu]
        @test predict_parameters(fit, data)[:mu] ≈ fitted(fit)
        # delta-method SE on the response scale: sqrt(x'Vx) * μ(1 - μ)
        ps = predict_parameters(fit, nd; se = true)[:mu]
        xr = hcat(ones(3), nd.x)
        ηr = xr * coef(fit)
        sel = sqrt.([xr[i, :]' * V * xr[i, :] for i in 1:3])
        @test ps.value ≈ _blogistic.(ηr)
        @test ps.se ≈ sel .* _blogistic.(ηr) .* (1 .- _blogistic.(ηr)) rtol = 1e-8

        # information criteria and counts
        ll = sum(Distributions.logpdf(Distributions.Binomial(ntr[i], μh[i]), s[i]) for i in 1:n)
        @test loglik(fit) ≈ ll atol = 1e-6
        @test dof(fit) == k
        @test nobs(fit) == n
        @test dof_residual(fit) == n - k
        @test aic(fit) ≈ -2 * loglik(fit) + 2k
        @test aic(fit) ≈ -2 * ll + 2k atol = 1e-5
        @test bic(fit) ≈ -2 * loglik(fit) + k * log(n)
        @test bic(fit) ≈ -2 * ll + k * log(n) atol = 1e-5
        @test bic(fit) - aic(fit) ≈ k * (log(n) - 2)
        @test isfinite(deviance(fit))

        # simulate: success counts in 0..ntr, reproducible under a seeded rng,
        # replicate mean matches ntr * mu
        sim = simulate(fit; rng = MersenneTwister(5))
        @test length(sim) == n
        @test all(0 .<= sim .<= ntr) && all(isinteger, sim)
        @test sim == simulate(fit; rng = MersenneTwister(5))
        @test sim != simulate(fit; rng = MersenneTwister(6))
        nsim = 4000
        M = simulate(fit; nsim = nsim, rng = MersenneTwister(11))
        @test size(M) == (n, nsim)
        @test all(0 .<= M .<= ntr)
        sd_mean = sqrt.(ntr .* μh .* (1 .- μh) ./ nsim)
        @test all(abs.(vec(mean(M; dims = 2)) .- ntr .* μh) .< 5 .* sd_mean)
        @test_throws ArgumentError simulate(fit; nsim = 0)
    end

    @testset "accessor contract: 0/1 Bernoulli response" begin
        Random.seed!(20261021)
        n = 500
        x = randn(n)
        p = _blogistic.(-0.2 .+ 0.6 .* x)
        y = Float64.(rand(n) .< p)
        data = (; y, x)
        X = hcat(ones(n), x)
        fit = drm(bf(@formula(y ~ x)), Binomial(); data = data)
        @test is_converged(fit)
        @test coeftable(fit).rownms == ["mu: (Intercept)", "mu: x"]

        bh = _irls_binomial(X, y, ones(n))
        @test coef(fit) ≈ bh atol = 1e-5
        μh = _blogistic.(X * bh)
        Vh = inv(Symmetric(X' * ((μh .* (1 .- μh)) .* X)))
        @test vcov(fit) ≈ Vh rtol = 1e-4
        @test stderror(fit) ≈ sqrt.(diag(vcov(fit)))
        @test fitted(fit) ≈ μh atol = 1e-6
        @test residuals(fit) ≈ y .- fitted(fit) rtol = 1e-10

        # Bernoulli log-likelihood: sum of y log μ + (1 - y) log(1 - μ)
        ll = sum(y .* log.(μh) .+ (1 .- y) .* log.(1 .- μh))
        @test loglik(fit) ≈ ll atol = 1e-6
        @test aic(fit) ≈ -2 * ll + 4 atol = 1e-5
        @test bic(fit) ≈ -2 * ll + 2 * log(n) atol = 1e-5
        @test nobs(fit) == n

        pp = predict_parameters(fit, (; x = [0.0, 1.0]))
        @test pp[:mu] ≈ _blogistic.(coef(fit)[1] .+ coef(fit)[2] .* [0.0, 1.0]) rtol = 1e-12

        sim = simulate(fit; rng = MersenneTwister(3))
        @test length(sim) == n
        @test all(v -> v == 0 || v == 1, sim)
        @test sim == simulate(fit; rng = MersenneTwister(3))
        @test abs(mean(sim) - mean(μh)) < 5 * sqrt(mean(μh .* (1 .- μh)) / n)

        # prior weights multiply each row's log-likelihood: integer weights equal
        # duplicated rows
        w = Float64.(rand(MersenneTwister(2), 1:3, n))
        rep = vcat((fill(i, Int(w[i])) for i in 1:n)...)
        fw = drm(bf(@formula(y ~ x)), Binomial(); data = data, weights = w)
        fd = drm(bf(@formula(y ~ x)), Binomial(); data = (; y = y[rep], x = x[rep]))
        @test coef(fw) ≈ coef(fd) atol = 1e-6
        @test loglik(fw) ≈ loglik(fd) atol = 1e-6
    end
end
