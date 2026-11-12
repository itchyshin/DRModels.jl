using DRModels, Test, Random, Logging
import DelimitedFiles

# SkewNormal fixed-effect fits restart on the slant (see `_skewnormal_slant_restarts` in
# src/skewnormal.jl). The likelihood in the slant has two or three local optima on some
# data sets and the single default start can stop at the worse one. The fit also runs from
# four fixed starts. A restart replaces the incumbent only if Optim reports convergence, its
# plain objective is lower by more than 1e-5, and its estimate is an interior optimum
# (positive definite Hessian, finite and not inflated standard errors). So a fit whose
# default start is already best is returned unchanged, and a restart never gives a lower
# logLik, never an unconverged fit and never a degenerate one.
#
# Reference values (generators in test/fixtures/skewnormal_restarts/):
#  * sim159: a simulated set (n = 200). drmTMB 0.8.1 reaches -316.0686132784; the default
#    start stops 1.28 nats lower. Reference = a real drmTMB run (gen_skewnormal_restarts.R).
#  * cell_y_sk (mu ~ x1, sigma ~ x2, nu ~ x1): drmTMB 0.8.1 reaches -188.4150738871; the
#    default start stops at -188.4180374904. Reference = a real drmTMB run.
#  * cell_y_g (same formula, symmetric data): drmTMB 0.8.1 stops at -243.9303305893, which
#    is NOT the best optimum. The best known value -243.8767839692 comes from a 40-start
#    search (gen_skewnormal_restarts_widesearch.jl), not from drmTMB.
# Refusal fixtures (gen_skewnormal_restarts_ridge.jl): a restart that has a higher objective
# but must not be taken.
#  * ridge_sim235: the mirror start runs to a slant of about 8e2 (logLik 5.8 higher, Optim
#    says converged, Hessian not usable). Without the interior check it replaces a good fit.
#  * ridge_sim25: the default fit is itself unconverged on a ridge (slant 1e15); the restart
#    that ends on another ridge has a higher logLik by 0.13 and must not replace it.
#  * ridge_rev23: unconverged default on a ridge. Optim stores an objective from dual numbers
#    that is wrong by 15035 nats at slant 1.7e15, so a restart looks like a huge gain, but
#    the plain objective is equal to rounding. The fit must stay the default one.
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
# a fit on a ridge warns about a singular Hessian; the warning is not under test
_snr_fit(f, d; kw...) = with_logger(NullLogger()) do
    drm(f, SkewNormal(); data = d, kw...)
end

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
        for ctl in (DRMControl(maxit = 1000), DRMControl(multi_start = 2), DRMControl(optimizer = :bfgs),
                    DRMControl(start = zeros(6)))   # one control field each: all four skip
            fit = with_logger(NullLogger()) do   # multi_start warns when its starts disagree
                drm(f, SkewNormal(); data = d, control = ctl)
            end
            @test !haskey(fit.scales, :q2_restarts)
        end
        # se alone does not take over the optimiser
        fit = drm(f, SkewNormal(); data = d, control = DRMControl(se = false))
        @test haskey(fit.scales, :q2_restarts)
    end

    @testset "each control field skips the restarts, one at a time" begin
        d = _snr_data("sim159.csv")
        f = _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))
        ref = drm(f, SkewNormal(); data = d)
        @test haskey(ref.scales, :q2_restarts) && ref.scales[:q2_restarts][2] >= 1.0
        for (name, ctl) in (("maxit", DRMControl(maxit = 1000)), ("multi_start", DRMControl(multi_start = 2)),
                            ("optimizer", DRMControl(optimizer = :bfgs)), ("start", DRMControl(start = zeros(6))))
            fit = with_logger(NullLogger()) do
                drm(f, SkewNormal(); data = d, control = ctl)
            end
            @test !haskey(fit.scales, :q2_restarts)
            # maxit and start leave the default start alone, so the 1.28 nats are not found;
            # multi_start and a different optimiser may find them by their own means
            name in ("maxit", "start") && @test loglikelihood(fit) < loglikelihood(ref) - 0.5
        end
    end

    @testset "degenerate restarts are refused (interior check)" begin
        # sim235: the mirror start gets to a slant of about 8e2 with a logLik 5.8 higher.
        d = _snr_data("ridge_sim235.csv")
        f = _snr_f(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ x))
        base = _snr_fit(f, d; control = _snr_norestart())
        iα = 4                                  # first slant coefficient: mu 2, sigma 1
        start = copy(base.theta); start[iα:end] .*= -1
        r = DRModels.Optim.optimize(base.nll, start, DRModels.Optim.LBFGS(),
                                    DRModels.Optim.Options(g_tol = 1e-8); autodiff = :forward)
        @test DRModels.drm_optim_converged(r)                         # Optim says converged ...
        @test -DRModels.Optim.minimum(r) > loglikelihood(base) + 5     # ... with a much higher objective ...
        @test !DRModels._skewnormal_interior_ok(base.nll, DRModels.Optim.minimizer(r))   # ... that is not an optimum
        fit = _snr_fit(f, d)
        @test is_converged(base)
        @test fit.scales[:q2_restarts][1:3] == [1.0, 0.0, 0.0]
        @test fit.theta == base.theta && fit.loglik == base.loglik && fit.vcov == base.vcov
        @test is_converged(fit)
        @test maximum(abs, fit.theta) < 10
    end

    @testset "an unconverged default fit on a ridge is not replaced by another ridge fit" begin
        for (file, f) in (("ridge_sim25.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))),
                          ("ridge_rev23.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ 1))))
            d = _snr_data(file)
            base = _snr_fit(f, d; control = _snr_norestart())
            fit = _snr_fit(f, d)
            @test !is_converged(base)
            @test fit.scales[:q2_restarts][1:3] == [1.0, 0.0, 0.0]
            @test fit.theta == base.theta && fit.loglik == base.loglik && isequal(fit.vcov, base.vcov)
            @test !is_converged(fit)                       # the user is still told
            @test maximum(abs, fit.theta) > 1e10           # the ridge, as on main
        end
        # the stored objective is the trap on ridge_rev23 (dual-number value far from the plain one)
        d = _snr_data("ridge_rev23.csv")
        base = _snr_fit(_snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ 1)), d; control = _snr_norestart())
        @test base.nll(base.theta) ≈ 77.9635963430 atol = 1e-6
    end

    @testset "restart rule on a synthetic double well (convergence flag)" begin
        # theta = [a, s]; the slant is s (index 2). The default start sits in the shallow well
        # near s = +2; the deeper well is at s = -3. In the rough variant the deeper well has a
        # kink at its bottom (|s + 3|), so L-BFGS reaches it with a lower objective and a positive
        # definite Hessian but never gets the gradient under 1e-8: Optim does NOT report
        # convergence (1000 iterations). Such a restart must not replace the converged default.
        well(s) = (s - 2)^2 * (s + 3)^2 / 10 + 0.3s
        smooth(th) = th[1]^2 + well(th[2])
        rough(th) = th[1]^2 + well(th[2]) + (th[2] < 0 ? abs(th[2] + 3.0) : zero(th[2]))
        θ0 = [0.0, 2.0]
        opt(nll, th) = DRModels.Optim.optimize(nll, th, DRModels.Optim.LBFGS(),
                                               DRModels.Optim.Options(g_tol = 1e-8); autodiff = :forward)
        res_s = opt(smooth, θ0)
        @test DRModels.drm_optim_converged(res_s)
        _, winner, gain, flipped = DRModels._skewnormal_slant_restarts(smooth, res_s, θ0, 2, 1e-8)
        @test winner >= 1 && gain > 0.5 && !flipped    # a deeper converged well is taken
        res_r = opt(rough, θ0)
        @test DRModels.drm_optim_converged(res_r)
        rr = opt(rough, [0.0, -2.0])                    # what a restart sees in the rough well
        @test !DRModels.drm_optim_converged(rr)
        @test rough(DRModels.Optim.minimizer(rr)) < rough(DRModels.Optim.minimizer(res_r)) - 0.5
        @test DRModels._skewnormal_interior_ok(rough, DRModels.Optim.minimizer(rr))
        res, winner, gain, flipped = DRModels._skewnormal_slant_restarts(rough, res_r, θ0, 2, 1e-8)
        @test winner == 0 && gain == 0.0 && !flipped
        @test res === res_r
    end

    @testset "no restart ever lowers the logLik or the converged status" begin
        for (file, f) in (("sim159.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))),
                          ("cell_y_sk.csv", _snr_f(@formula(y_sk ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1))),
                          ("cell_y_g.csv", _snr_f(@formula(y_g ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1))),
                          ("ridge_sim235.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ x))),
                          ("ridge_sim25.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))),
                          ("ridge_rev23.csv", _snr_f(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ 1))))
            d = _snr_data(file)
            fit = _snr_fit(f, d)
            base = _snr_fit(f, d; control = _snr_norestart())
            @test loglikelihood(fit) >= loglikelihood(base)
            @test is_converged(fit) >= is_converged(base)
        end
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
