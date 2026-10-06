# Staged (frozen-margin) latent-normal association — drmTMB's
# `associate_pairs()`. First slice: the `gaussian_bernoulli` pair class.
#
# This is a TWO-STAGE estimator: margins are fitted independently, frozen, and
# only the association is estimated. The tests therefore check the estimator AND
# the refusals — an over-broad staged route that silently accepts an unreviewed
# pair class would be worse than one that errors.

using DRModels
using Test
using Random
import Distributions

@testset "staged association (drmTMB associate_pairs)" begin

    N = Distributions.Normal()
    rng = MersenneTwister(5)
    n = 3000
    x = randn(rng, n)
    eta_true = 0.55
    z1 = randn(rng, n)
    z2 = eta_true .* z1 .+ sqrt(1 - eta_true^2) .* randn(rng, n)
    y = 0.3 .+ 0.7 .* x .+ z1
    pb = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x)))
    zb = Float64.(z2 .> [Distributions.quantile(N, 1 - p) for p in pb])
    data = (; y = y, z = zb, x = x)

    fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data)
    fb = drm(bf(@formula(z ~ x)), Binomial(); data = data)

    a = associate_pairs(fg, fb; kernel = latent_normal())
    s = association(a)

    @testset "recovers the latent association" begin
        @test s.pair_class === :gaussian_bernoulli
        # Measured across 8 seeds at this n: mean 0.5433, sd 0.0108, bias -0.007.
        # The tolerance is ~4 sd, not a guess.
        @test isapprox(s.eta, eta_true; atol = 0.05)
        @test -1 < s.eta < 1
        @test s.nobs == n
    end

    @testset "optimum diagnostics" begin
        @test abs(s.score) < 1e-3                 # score ≈ 0 at the optimum
        # drmTMB's `curvature` is the LOGLIK curvature (it negates the objective's
        # second difference), so it is NEGATIVE at a maximum. Sign matters.
        @test s.curvature < 0
        @test s.near_boundary == false
        @test s.multistart_disagreement == false
    end

    @testset "argument order does not matter" begin
        flipped = association(associate_pairs(fb, fg; kernel = latent_normal()))
        @test isapprox(flipped.eta, s.eta; atol = 1e-8)
    end

    @testset "the kernel must be explicit" begin
        # drmTMB: "Supply an explicit kernel = latent_normal() declaration."
        @test_throws ArgumentError associate_pairs(fg, fb)
        @test_throws ArgumentError associate_pairs(fg, fb; kernel = :latent_normal)
    end

    @testset "all five reviewed pair classes recover the latent association" begin
        # Shared latents: margin-1 quantities from z1, margin-2 from z2, coupled at
        # eta_true. Discrete margins are built by inverting their own CDF at the PIT
        # u = Phi(z), which is exactly the latent-normal copula the estimator assumes.
        rng2 = MersenneTwister(4)
        m = 2000
        xx = randn(rng2, m)
        w1 = randn(rng2, m)
        w2 = eta_true .* w1 .+ sqrt(1 - eta_true^2) .* randn(rng2, m)
        uu1 = Distributions.cdf.(N, w1)
        uu2 = Distributions.cdf.(N, w2)
        yg = 0.3 .+ 0.7 .* xx .+ w1
        p1 = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* xx)))
        p2 = 1 ./ (1 .+ exp.(-(0.1 .- 0.4 .* xx)))
        bb1 = Float64.(uu1 .> 1 .- p1)
        bb2 = Float64.(uu2 .> 1 .- p2)
        sg = 0.6
        rr = 1 / sg^2
        mm1 = exp.(1.5 .+ 0.3 .* xx)
        mm2 = exp.(1.4 .- 0.2 .* xx)
        nb(mu, u) = Float64(Distributions.quantile(
            Distributions.NegativeBinomial(rr, rr / (rr + mu)), u))
        cc1 = [nb(mm1[i], uu1[i]) for i in 1:m]
        cc2 = [nb(mm2[i], uu2[i]) for i in 1:m]
        dd = (; y = yg, b1 = bb1, b2 = bb2, c1 = cc1, c2 = cc2, x = xx)

        gf  = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dd)
        bf1 = drm(bf(@formula(b1 ~ x)), Binomial(); data = dd)
        bf2 = drm(bf(@formula(b2 ~ x)), Binomial(); data = dd)
        cf1 = drm(bf(@formula(c1 ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = dd)
        cf2 = drm(bf(@formula(c2 ~ x), @formula(sigma ~ 1)), NegBinomial2(); data = dd)

        # Tolerance is MEASURED, not guessed: a 5-seed study at this n gives
        #   gaussian_nbinom2    mean 0.5392  sd 0.0146  max|dev| 0.0256
        #   bernoulli_nbinom2   mean 0.5389  sd 0.0219  max|dev| 0.0403
        #   bernoulli_bernoulli mean 0.5467  sd 0.0268  max|dev| 0.0325
        #   nbinom2_nbinom2     mean 0.5408  sd 0.0141  max|dev| 0.0248
        # 0.10 is ~2.5x the worst observed deviation.
        for (label, f1, f2, cls) in (
                ("gaussian_bernoulli",  gf,  bf2, :gaussian_bernoulli),
                ("gaussian_nbinom2",    gf,  cf2, :gaussian_nbinom2),
                ("bernoulli_nbinom2",   bf1, cf2, :bernoulli_nbinom2),
                ("bernoulli_bernoulli", bf1, bf2, :bernoulli_bernoulli),
                ("nbinom2_nbinom2",     cf1, cf2, :nbinom2_nbinom2))
            si = association(associate_pairs(f1, f2; kernel = latent_normal()))
            @test si.pair_class === cls
            @test isapprox(si.eta, eta_true; atol = 0.10)
            @test si.curvature < 0          # loglik curvature at a maximum
            @test si.near_boundary == false
        end
    end

    @testset "quadrature diagnostics are retained for both-censored classes" begin
        # drmTMB keeps `abs.error` from its adaptive integration; a rectangle
        # probability that silently lost precision would corrupt the association
        # with no visible failure. The closed-form classes have no integral at all.
        rng3 = MersenneTwister(8)
        m = 400
        xx = randn(rng3, m)
        w1 = randn(rng3, m)
        w2 = 0.5 .* w1 .+ sqrt(0.75) .* randn(rng3, m)
        uu1 = Distributions.cdf.(N, w1); uu2 = Distributions.cdf.(N, w2)
        pp = 1 ./ (1 .+ exp.(-(0.2 .+ 0.4 .* xx)))
        dd = (; b1 = Float64.(uu1 .> 1 .- pp), b2 = Float64.(uu2 .> 1 .- pp),
                y = 0.2 .+ 0.5 .* xx .+ w1, x = xx)
        f1 = drm(bf(@formula(b1 ~ x)), Binomial(); data = dd)
        f2 = drm(bf(@formula(b2 ~ x)), Binomial(); data = dd)
        arect = associate_pairs(f1, f2; kernel = latent_normal())
        diag = integration_diagnostics(arect)
        @test diag !== nothing
        @test length(diag.probability) == m
        @test all(diag.probability .> 0)
        @test all(isfinite, diag.abs_error)
        @test diag.worst_relative_error < 1e-6      # adaptive rtol is 1e-10
        # closed-form class => no integral => nothing to report
        fgz = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dd)
        @test integration_diagnostics(associate_pairs(fgz, f2; kernel = latent_normal())) === nothing
    end

    @testset "unreviewed pair classes are refused, not approximated" begin
        fp = drm(bf(@formula(c ~ x)), Poisson();
                 data = (; c = Float64.(rand(rng, 0:4, n)), x = x))
        # gaussian × poisson is not one of drmTMB's five reviewed classes.
        @test_throws ArgumentError associate_pairs(fg, fp; kernel = latent_normal())
        # gaussian × gaussian is not a staged class either (that is biv_gaussian).
        fg2 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = data)
        @test_throws ArgumentError associate_pairs(fg, fg2; kernel = latent_normal())
    end

    @testset "the binomial margin must be literal Bernoulli" begin
        ntr = 8
        s_cnt = Float64.([count(_ -> rand(rng) < pb[i], 1:ntr) for i in 1:n])
        f_cnt = Float64.(ntr .- s_cnt)
        dcnt = (; s = s_cnt, f = f_cnt, x = x, y = y)
        fbin = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = dcnt)
        fgc = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dcnt)
        @test_throws ArgumentError associate_pairs(fgc, fbin; kernel = latent_normal())
    end

    @testset "only an intercept-only association is implemented" begin
        @test_throws ArgumentError associate_pairs(fg, fb; kernel = latent_normal(),
                                                   association = @formula(association ~ x))
    end

    @testset "the uncertainty caveat is carried in the result" begin
        # Frozen margins ⇒ the association SE is conditional. That must be stated
        # wherever it is reported, not left for the reader to infer.
        @test occursin("conditional", s.uncertainty)
        @test occursin("frozen", s.uncertainty)
    end
end
