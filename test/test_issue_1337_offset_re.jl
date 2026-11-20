# Issue #1337: Gaussian() and Binomial() accept offset(...) + one (1 | g), drmTMB parity.
using DRModels, Random, Test

@testset "issue #1337: offset with one ordinary (1 | g) on Gaussian and Binomial" begin
    Random.seed!(1)
    n = 120
    g = repeat(string.("g", 1:12), inner = 10)
    x1 = randn(n)
    off = randn(n) ./ 2
    d = (;
        yg = x1 .+ off .+ randn(12)[repeat(1:12, inner = 10)] .+ randn(n),
        yb = Float64.(rand(n) .< 0.5),
        yp = Float64.(rand(0:6, n)),
        x1, g, off,
    )
    # Issue reproducer (Julia): Poisson and NegBinomial2 still fit; pins from the issue body.
    fp = drm(bf(@formula(yp ~ x1 + offset(off) + (1 | g))), Poisson(); data = d, se = false)
    @test fp.converged
    @test loglik(fp) ≈ -302.8879001869965 atol = 1e-6
    fn = drm(bf(@formula(yp ~ x1 + offset(off) + (1 | g)), @formula(sigma ~ 1)),
              NegBinomial2(); data = d, se = false)
    @test fn.converged
    @test loglik(fn) ≈ -280.4357885417233 atol = 1e-6

    # Gaussian and Binomial: no longer refused; offset is y - off with (1 | g) unchanged.
    fg = drm(bf(@formula(yg ~ x1 + offset(off) + (1 | g)), @formula(sigma ~ 1)),
             Gaussian(); data = d, se = false)
    @test fg.converged
    fb = drm(bf(@formula(yb ~ x1 + offset(off) + (1 | g))), Binomial(); data = d, se = false)
    @test fb.converged
    @test length(coef(fb, :mu)) == 2
    @test isfinite(loglik(fg)) && isfinite(loglik(fb))

    # Gaussian: same likelihood as fitting the shifted response without the offset term.
    f0 = drm(bf(@formula(y0 ~ x1 + (1 | g)), @formula(sigma ~ 1)), Gaussian();
             data = merge(d, (; y0 = d.yg .- d.off)), control = DRMControl(se = false))
    @test loglik(fg) ≈ loglik(f0) rtol = 1e-8
    @test coef(fg, :mu) ≈ coef(f0, :mu) rtol = 1e-8
    @test exp(coef(fg, :resd)[1]) ≈ exp(coef(f0, :resd)[1]) rtol = 1e-6

    # Binomial: offset in η at coefficient 1; Laplace and default LA agree closely.
    fbl = drm(bf(@formula(yb ~ x1 + offset(off) + (1 | g))), Binomial();
              data = d, marginal = :Laplace, se = false)
    @test fbl.converged
    @test coef(fbl, :mu) ≈ coef(fb, :mu) atol = 5e-2
    @test loglik(fbl) ≈ loglik(fb) atol = 5e-2

    # refusals that stay
    @test_throws ArgumentError drm(bf(@formula(yg ~ x1 + offset(off) + (1 + x1 | g))), Gaussian(); data = d)
    @test_throws ArgumentError drm(bf(@formula(yb ~ x1 + offset(off) + (1 + x1 | g))), Binomial(); data = d)
end