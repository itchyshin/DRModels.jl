# Parametric bootstrap on non-Gaussian families. bootstrap_ci was Gaussian-only
# because simulate() only knew how to draw Gaussian responses; this exercises the
# generalised simulate + the family-agnostic bootstrap path.
using DRModels
using Test, Random
import Distributions

@testset "Parametric bootstrap — non-Gaussian families" begin
    @testset "Poisson bootstrap brackets β and covers truth" begin
        Random.seed!(20260801)
        n = 400; x = randn(n)
        β = [0.4, 0.6]
        y = Float64[rand(Distributions.Poisson(exp(β[1] + β[2] * x[i]))) for i in 1:n]
        dat = (; y, x)
        rows = bootstrap_ci(bf(@formula(y ~ x)), Poisson(); data = dat, B = 200,
                            rng = MersenneTwister(1))
        xr = first(r for r in rows if r.coef == "x")
        @test xr.lower < xr.estimate < xr.upper
        @test xr.lower < β[2] < xr.upper
        @test isfinite(xr.lower) && isfinite(xr.upper)
    end

    @testset "Gamma bootstrap brackets β" begin
        Random.seed!(20260802)
        n = 400; x = randn(n)
        β = [0.5, 0.4]; α = 8.0
        μ = exp.(β[1] .+ β[2] .* x)
        y = Float64[rand(Distributions.Gamma(α, μ[i] / α)) for i in 1:n]
        dat = (; y, x)
        rows = bootstrap_ci(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gamma();
                            data = dat, B = 200, rng = MersenneTwister(2))
        xr = first(r for r in rows if r.coef == "x")
        @test xr.lower < xr.estimate < xr.upper
        @test xr.lower < β[2] < xr.upper
    end

    @testset "simulate draws valid non-Gaussian responses" begin
        Random.seed!(20260803)
        n = 300; x = randn(n)
        yp = Float64[rand(Distributions.Poisson(exp(0.3 + 0.5x[i]))) for i in 1:n]
        fitp = drm(bf(@formula(y ~ x)), Poisson(); data = (; y = yp, x))
        sp = simulate(fitp; rng = MersenneTwister(3))
        @test length(sp) == n
        @test all(sp .>= 0) && all(isinteger, sp)      # counts
    end

    @testset "simulate covers non-Gaussian family surface" begin
        rng = MersenneTwister(20260804)

        n = 180
        x = randn(rng, n)

        # Binomial two-column response: simulate successes and keep trials valid.
        ntr = fill(8, n)
        p = @. 1 / (1 + exp(-(0.2 + 0.7x)))
        s = Float64[rand(rng, Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
        fail = Float64.(ntr) .- s
        fit_bin = drm(bf(@formula(cbind(s, fail) ~ x)), Binomial(); data = (; s, fail, x))
        ybin = simulate(fit_bin; rng = MersenneTwister(4))
        @test all(0 .<= ybin .<= ntr)
        @test all(isinteger, ybin)

        # Beta-binomial: same two-column response reconstruction plus dispersion.
        φbb = 12.0
        sbb = Float64[rand(rng, Distributions.BetaBinomial(ntr[i], p[i] * φbb, (1 - p[i]) * φbb)) for i in 1:n]
        fbb = Float64.(ntr) .- sbb
        fit_bb = drm(bf(@formula(cbind(sbb, fbb) ~ x), @formula(sigma ~ 1)), BetaBinomial();
            data = (; sbb, fbb, x))
        ybb = simulate(fit_bb; rng = MersenneTwister(5))
        @test all(0 .<= ybb .<= ntr)
        @test all(isinteger, ybb)

        # NB2, Beta, Gamma, LogNormal, Student all need their per-row scale vectors.
        μ = exp.(0.3 .+ 0.4 .* x)
        ynb = Float64[rand(rng, Distributions.NegativeBinomial(4.0, 4.0 / (4.0 + μ[i]))) for i in 1:n]
        fit_nb = drm(bf(@formula(ynb ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = (; ynb, x))
        @test all(simulate(fit_nb; rng = MersenneTwister(6)) .>= 0)

        ybeta = Float64[rand(rng, Distributions.Beta(p[i] * 20.0, (1 - p[i]) * 20.0)) for i in 1:n]
        fit_beta = drm(bf(@formula(ybeta ~ x), @formula(sigma ~ 1)), Beta(); data = (; ybeta, x))
        sbeta = simulate(fit_beta; rng = MersenneTwister(7))
        @test all(0 .< sbeta .< 1)

        yg = Float64[rand(rng, Distributions.Gamma(8.0, μ[i] / 8.0)) for i in 1:n]
        fit_gamma = drm(bf(@formula(yg ~ x), @formula(sigma ~ 1)), Gamma(); data = (; yg, x))
        @test all(simulate(fit_gamma; rng = MersenneTwister(8)) .> 0)

        yln = Float64[exp(0.1 + 0.3 * x[i] + 0.25 * randn(rng)) for i in 1:n]
        fit_ln = drm(bf(@formula(yln ~ x), @formula(sigma ~ 1)), LogNormal(); data = (; yln, x))
        @test all(simulate(fit_ln; rng = MersenneTwister(9)) .> 0)

        yt = Float64[0.2 + 0.4 * x[i] + 0.6 * rand(rng, Distributions.TDist(8.0)) for i in 1:n]
        fit_t = drm(bf(@formula(yt ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student();
            data = (; yt, x))
        @test all(isfinite, simulate(fit_t; rng = MersenneTwister(10)))

        # Boundary/mixture and ordinal variants.
        yzob = Vector{Float64}(undef, n)
        for i in 1:n
            yzob[i] = rand(rng) < 0.2 ? (rand(rng) < 0.35 ? 1.0 : 0.0) :
                rand(rng, Distributions.Beta(p[i] * 18.0, (1 - p[i]) * 18.0))
        end
        fit_zob = drm(bf(@formula(yzob ~ x), @formula(sigma ~ 1), @formula(zoi ~ 1), @formula(coi ~ 1)),
            ZeroOneBeta(); data = (; yzob, x))
        szob = simulate(fit_zob; rng = MersenneTwister(11))
        @test all(0 .<= szob .<= 1)

        ytwn = Vector{Float64}(undef, n)
        for i in 1:n
            λ = μ[i]^(0.5) / (1.4 * 0.5)
            γ = 1.4 * 0.5 * μ[i]^0.5
            sh = 1.0
            N = rand(rng, Distributions.Poisson(λ))
            ytwn[i] = N == 0 ? 0.0 : rand(rng, Distributions.Gamma(N * sh, γ))
        end
        fit_tw = drm(bf(@formula(ytwn ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie();
            data = (; ytwn, x))
        @test all(simulate(fit_tw; rng = MersenneTwister(12)) .>= 0)

        K = 4
        yord = Vector{Float64}(undef, n)
        cuts = [-0.8, 0.2, 1.1]
        for i in 1:n
            u = rand(rng)
            yord[i] = K
            for k in 1:(K - 1)
                if u < 1 / (1 + exp(-(cuts[k] - 0.5 * x[i])))
                    yord[i] = k
                    break
                end
            end
        end
        fit_ord = drm(bf(@formula(yord ~ x)), CumulativeLogit(); data = (; yord, x))
        sord = simulate(fit_ord; rng = MersenneTwister(13))
        @test all(1 .<= sord .<= K)
        @test all(isinteger, sord)
    end

    @testset "bootstrap and profile smoke checks beyond Gaussian" begin
        rng = MersenneTwister(20260805)
        n = 160
        x = randn(rng, n)
        y = Float64[rand(rng, Distributions.Poisson(exp(0.3 + 0.5 * x[i]))) for i in 1:n]
        form = bf(@formula(y ~ x))
        dat = (; y, x)
        fit = drm(form, Poisson(); data = dat)

        prof = confint(fit; method = :profile)
        @test length(prof) == length(coef(fit))
        @test all(r.lower < r.estimate < r.upper for r in prof)

        boot = bootstrap_ci(form, Poisson(); data = dat, B = 20,
            rng = MersenneTwister(14), threads = true)
        boot_fit = bootstrap_ci(fit; data = dat, B = 20,
            rng = MersenneTwister(14), threads = true)
        @test length(boot) == length(coef(fit))
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in boot)
        @test boot_fit == boot

        bsum = bootstrap_summary(form, Poisson(); data = dat, B = 20,
            rng = MersenneTwister(14), threads = true)
        bsum_fit = bootstrap_summary(fit; data = dat, B = 20,
            rng = MersenneTwister(14), threads = true)
        @test length(bsum) == length(boot)
        @test all(r.std_error > 0 for r in bsum)
        @test [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in bsum] ==
              [(r.param, r.coef, r.estimate, r.lower, r.upper) for r in boot]
        @test bsum_fit == bsum

        bres = bootstrap_result(form, Poisson(); data = dat, B = 12,
            rng = MersenneTwister(16), threads = true)
        bres_fit = bootstrap_result(fit; data = dat, B = 12,
            rng = MersenneTwister(16), threads = true)
        @test bres.attempted == bres.used == 12
        @test bres.failed == 0
        @test isempty(bres.failures)
        @test length(bres.summary) == length(coef(fit))
        @test bres_fit.summary == bres.summary
        @test bres_fit.seeds == bres.seeds
        @test bres_fit.attempted == bres.attempted
        @test bres_fit.used == bres.used
        @test bres_fit.failed == bres.failed
        @test bres.threaded == (Threads.nthreads() > 1)
        @test bres.worker_threads == (Threads.nthreads() > 1 ? min(12, Threads.nthreads()) : 1)
        @test bres.blas_oversubscribed == (bres.threaded && bres.blas_threads > 1)

        bres_serial = bootstrap_result(fit; data = dat, B = 12,
            rng = MersenneTwister(17))
        bres_threaded = bootstrap_result(fit; data = dat, B = 12,
            rng = MersenneTwister(17), threads = true)
        @test bres_threaded.summary == bres_serial.summary
        @test bres_threaded.seeds == bres_serial.seeds

        ntr = fill(6, n)
        p = @. 1 / (1 + exp(-(0.1 + 0.4 * x)))
        s = Float64[rand(rng, Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
        fail = Float64.(ntr) .- s
        binform = bf(@formula(cbind(s, fail) ~ x))
        binboot = bootstrap_ci(binform, Binomial(); data = (; s, fail, x), B = 12,
            rng = MersenneTwister(15), threads = true)
        @test length(binboot) == 2
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in binboot)
    end
end
