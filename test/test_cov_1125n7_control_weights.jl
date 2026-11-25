# cov lane 1125n7 — DRMControl prior weights and scope helpers (src/control.jl).
using DRModels
using Test, Random, StatsModels

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

@testset "cov 1125n7: _check_prior_weights" begin
    w = DRModels._check_prior_weights([1.0, 2.0, 0.5])
    @test w == [1.0, 2.0, 0.5]
    @test DRModels._check_prior_weights(nothing) === nothing
    @test DRModels._check_prior_weights(ones(3)) == ones(3)
    for bad in ([], [1.0, -0.1], [NaN], [Inf], zeros(4))
        @test_throws ArgumentError DRModels._check_prior_weights(bad)
    end
    @test_throws ArgumentError DRModels._check_prior_weights("not-a-vector")
    @test_throws ArgumentError DRModels._check_prior_weights(Dict(:a => 1))
end

@testset "cov 1125n7: _ctl_has_latent_terms and simple Gaussian RE" begin
    d = (; y = randn(20), x = randn(20), g = repeat(1:4, 5))
    f_fe = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    f_re = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
    f_sd = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1))
    @test !DRModels._ctl_has_latent_terms(f_fe)
    @test DRModels._ctl_has_latent_terms(f_re)
    @test DRModels._ctl_has_latent_terms(f_sd)
    @test DRModels._ctl_simple_gaussian_re(f_re, Gaussian())
    @test !DRModels._ctl_simple_gaussian_re(f_fe, Gaussian())
    @test !DRModels._ctl_simple_gaussian_re(f_re, Poisson())
    f_slope = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1))
    @test !DRModels._ctl_simple_gaussian_re(f_slope, Gaussian())
end

@testset "cov 1125n7: _ctl_weights_latent_ok shapes" begin
    d = (; y = randn(12), g = repeat(1:3, 4))
    f_meta = bf(@formula(y ~ meta_V(v)), @formula(sigma ~ 1))
    @test DRModels._ctl_weights_latent_ok(f_meta, Gaussian())
    f_pois = bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))
    @test DRModels._ctl_weights_latent_ok(f_pois, Poisson())
    f_gaus_re = bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))
    @test DRModels._ctl_weights_latent_ok(f_gaus_re, Gaussian())
    f_bin = bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))
    @test !DRModels._ctl_weights_latent_ok(f_bin, Binomial())
    fb = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x), sigma1 = @formula(sigma1 ~ 1),
           sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    @test DRModels._ctl_weights_latent_ok(fb, Gaussian())
end

@testset "cov 1125n7: _kw_advice and _kw_r_note fragments" begin
    for (key, frag) in (
        (:method, "ML-only"),
        (:tree, "phylo"),
        (:K, "relmat"),
        (:coords, "spatial"),
        (:contrasts, "treatment"),
        (:se, "DRMControl"),
    )
        adv = DRModels._kw_advice(key, Student())
        @test occursin(frag, adv)
    end
    note = DRModels._kw_r_note(:tree, Student())
    @test occursin("drmTMB", note) || occursin("verified", note)
    @test occursin("refused", DRModels._kw_advice(:A, LogNormal())) ||
        occursin("animal", DRModels._kw_advice(:A, LogNormal()))
end

@testset "cov 1125n7: _refuse_unsupported_kwargs message" begin
    f = bf(@formula(y ~ x), @formula(sigma ~ 1))
    msg = _msg(() -> DRModels._refuse_unsupported_kwargs([:tree], f, Poisson(), [:data, :g_tol]))
    @test occursin("Poisson", msg)
    @test occursin("`tree`", msg)
    @test occursin("accepts", msg)
end

@testset "cov 1125n7: DRMControl constructor" begin
    c = DRMControl(maxit = 200, multi_start = 2, optimizer = :lbfgs)
    @test c.maxit == 200
    @test c.multi_start == 2
    for ctor in (
        () -> DRMControl(multi_start = 0),
        () -> DRMControl(jitter = -1.0),
        () -> DRMControl(optimizer = :foo),
    )
        @test_throws ArgumentError ctor()
    end
end