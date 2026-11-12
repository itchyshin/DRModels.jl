using DRModels, Test, Random, Logging
import DelimitedFiles

# SkewNormal fixed-effect fits restart on the slant (see `_SKEWNORMAL_RESTARTS` in
# src/skewnormal.jl). The likelihood in the slant has two or three local optima on some
# data sets and the single default start can stop at the worse one. The fit also runs from
# four fixed starts and keeps one only on a clear gain (the shared `_q2_boundary_restarts`
# rule), so a fit whose default start is already best is returned unchanged.
#
# Reference values (generators in test/fixtures/skewnormal_restarts/):
#  * sim159: a simulated set (n = 200). drmTMB 0.8.1 reaches -316.0686132784; the default
#    start stops 1.28 nats lower. Reference = a real drmTMB run (gen_skewnormal_restarts.R).
#  * cell_y_sk (mu ~ x1, sigma ~ x2, nu ~ x1): drmTMB 0.8.1 reaches -188.4150738871; the
#    default start stops at -188.4180374904. Reference = a real drmTMB run.
#  * cell_y_g (same formula, symmetric data): drmTMB 0.8.1 stops at -243.9303305893, which
#    is NOT the best optimum. The best known value -243.8767839692 comes from a 40-start
#    search (gen_skewnormal_restarts_widesearch.jl), not from drmTMB.
# The "no restart" fit is the same fit with `DRMControl(maxit = 1000)`: a user-set
# optimiser control skips the restarts and 1000 is Optim's own default iteration limit, so
# that fit is the default start alone. These tests therefore fail if the restarts are removed.

const _SNR_DIR = joinpath(@__DIR__, "fixtures", "skewnormal_restarts")
function _snr_data(name)
    m, h = DelimitedFiles.readdlm(joinpath(_SNR_DIR, name), ',', Any; header = true)
    h = Symbol.(vec(h))
    return NamedTuple{Tuple(h)}(Tuple(Float64.(m[:, j]) for j in eachindex(h)))
end
_snr_norestart() = DRMControl(maxit = 1000)
_snr_f(mu, sig, nu) = bf(mu, sig, nu)

@testset "SkewNormal slant restarts" begin
    @testset "sim159: reaches drmTMB's optimum, default start stops 1.28 nats lower" begin
        d = _snr_data("sim159.csv")
        f = _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))
        thetaR = [1.146720007, 0.6041880792, 0.1800795534, 0.2696337805, -0.3225204401, 1.373632544]
        seR = [0.0906847945, 0.07507046635, 0.05199084912, 0.04785143591, 0.578565815, 0.4931256729]
        llR = -316.0686132784
        base = drm(f, SkewNormal(); data = d, control = _snr_norestart())
        @test loglikelihood(base) ≈ -317.34700932 atol = 1e-6      # the default start alone
        @test !haskey(base.scales, :q2_restarts)
        fit = drm(f, SkewNormal(); data = d)
        @test fit.converged
        @test loglikelihood(fit) ≈ llR atol = 1e-4                 # agreement rule: logLik 1e-4
        @test all(abs.(fit.theta .- thetaR) .<= 1e-3 .* seR)       # estimates within 1e-3 of R's SE
        @test loglikelihood(fit) > loglikelihood(base) + 1.0
        rs = fit.scales[:q2_restarts]
        @test rs[1] == 1.0 && rs[2] >= 1.0 && rs[4] == 0.0
        @test rs[3] ≈ loglikelihood(fit) - loglikelihood(base) atol = 1e-6
        # the objective is the one R evaluates: R's estimate gives R's log-likelihood
        @test -fit.nll(thetaR) ≈ llR atol = 1e-4
    end

    @testset "cell_y_sk: the 0.003 gap to drmTMB closes" begin
        d = _snr_data("cell_y_sk.csv")
        f = _snr_f(@formula(y_sk ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1))
        thetaR = [1.298753216, 0.3239575557, -0.2378711945, 0.06611080913, 0.6613016179, 1.23569556]
        seR = [0.06235827146, 0.0628022488, 0.05740424586, 0.04989686766, 1.383802734, 0.9304989189]
        llR = -188.4150738871
        base = drm(f, SkewNormal(); data = d, control = _snr_norestart())
        @test loglikelihood(base) ≈ -188.4180374904 atol = 1e-6
        fit = drm(f, SkewNormal(); data = d)
        @test loglikelihood(fit) ≈ llR atol = 1e-4
        @test all(abs.(fit.theta .- thetaR) .<= 1e-3 .* seR)
        @test fit.scales[:q2_restarts][2] >= 1.0
    end

    @testset "cell_y_g: beats drmTMB, reaches the wide-search optimum" begin
        d = _snr_data("cell_y_g.csv")
        f = _snr_f(@formula(y_g ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1))
        llR = -243.9303305893            # drmTMB 0.8.1, a worse local optimum
        llbest = -243.8767839692         # best of 40 starts (not from drmTMB)
        base = drm(f, SkewNormal(); data = d, control = _snr_norestart())
        @test loglikelihood(base) ≈ -244.0174467279 atol = 1e-6
        fit = drm(f, SkewNormal(); data = d)
        @test loglikelihood(fit) ≈ llbest atol = 1e-6
        @test loglikelihood(fit) > llR + 0.05
        @test fit.scales[:q2_restarts][2] >= 1.0
    end

    @testset "a fit whose default start is already best is unchanged (bit-identical)" begin
        d = _snr_data("cell_y_sk.csv")
        for f in (_snr_f(@formula(y_sk ~ x1), @formula(sigma ~ 1), @formula(nu ~ 1)),
                  _snr_f(@formula(y_sk ~ x1), @formula(sigma ~ x1), @formula(nu ~ 1)))
            fit = drm(f, SkewNormal(); data = d)
            base = drm(f, SkewNormal(); data = d, control = _snr_norestart())
            @test fit.scales[:q2_restarts][1:3] == [1.0, 0.0, 0.0]
            @test fit.theta == base.theta
            @test fit.loglik == base.loglik
            @test fit.vcov == base.vcov
            @test fit.converged == base.converged
        end
        # pinned value of the pre-restart release for the simplest cell
        f0 = _snr_f(@formula(y_sk ~ x1), @formula(sigma ~ 1), @formula(nu ~ 1))
        @test loglikelihood(drm(f0, SkewNormal(); data = d)) ≈ -189.6242535996 atol = 1e-8
    end

    @testset "a user control on the optimiser skips the restarts" begin
        d = _snr_data("sim159.csv")
        f = _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))
        for ctl in (DRMControl(maxit = 1000), DRMControl(multi_start = 2), DRMControl(optimizer = :bfgs))
            fit = with_logger(NullLogger()) do   # multi_start warns when its starts disagree
                drm(f, SkewNormal(); data = d, control = ctl)
            end
            @test !haskey(fit.scales, :q2_restarts)
        end
        # weights and se alone do not take over the optimiser
        fit = drm(f, SkewNormal(); data = d, control = DRMControl(se = false))
        @test haskey(fit.scales, :q2_restarts)
    end

    @testset "post-fit functions work on a restarted fit" begin
        d = _snr_data("sim159.csv")
        f = _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))
        fit = drm(f, SkewNormal(); data = d)
        @test fit.scales[:q2_restarts][2] >= 1.0
        @test all(isfinite, vcov(fit))
        @test all(isfinite, stderror(fit))
        @test isfinite(aic(fit)) && isfinite(bic(fit))
        @test length(fitted(fit)) == length(d.y)
        pp = predict_parameters(fit, d)
        @test Set(keys(pp)) == Set([:mu, :sigma, :nu])
        sim = simulate(fit; nsim = 2, rng = Random.MersenneTwister(1))
        @test size(sim, 1) == length(d.y)
        # newdata simulation skips the bookkeeping key
        sim2 = simulate(fit; nsim = 1, newdata = (x = [-1.0, 0.0, 1.0],), rng = Random.MersenneTwister(1))
        @test size(sim2, 1) == 3
        null = drm(_snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ 1)), SkewNormal(); data = d)
        @test loglikelihood(fit) >= loglikelihood(null) - 1e-8
        lr = lrtest(null, fit)
        @test 0 <= lr.pvalue <= 1 && lr.dof == 1
    end
end
