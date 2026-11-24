# test_cov_1124d_associate_margin.jl — lane 1124d: staged-association margin guards
# (converged margin, row count) without predict-missing or parity harness paths.
using DRModels
using Test, Random

function _unconverged_margin(f::DrmFit)
    vals = Any[getfield(f, k) for k in fieldnames(typeof(f))]
    fn = fieldnames(typeof(f))
    vals[findfirst(==(:converged), fn)] = false
    return typeof(f).name.wrapper(vals...)
end

_msg(f) = try
    f()
    ""
catch e
    sprint(showerror, e)
end

@testset "cov 1124d: associate_pairs margin validation" begin
    rng = MersenneTwister(1124)
    n = 100
    x = randn(rng, n)
    y = 0.3 .+ 0.5 .* x .+ randn(rng, n)
    z = Float64.(rand(rng, n) .> 0.4)
    d = (; y, z, x)
    fg = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = d)
    fb = drm(bf(@formula(z ~ x)), Binomial(); data = d)

    @testset "non-converged margin is refused with an actionable message" begin
        msg = _msg(() -> associate_pairs(_unconverged_margin(fg), fb; kernel = latent_normal()))
        @test occursin("fit_1", msg)
        @test occursin("did not converge", msg)
        @test occursin("frozen", msg)
    end

    @testset "different row counts are refused" begin
        dshort = (; y = y[1:60], x = x[1:60])
        fshort = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = dshort)
        msg = _msg(() -> associate_pairs(fshort, fb; kernel = latent_normal()))
        @test occursin("same rows", msg)
        @test occursin("60", msg)
        @test occursin("100", msg)
    end

    @testset "association() summary fields on a healthy pair" begin
        a = associate_pairs(fg, fb; kernel = latent_normal())
        s = association(a)
        @test s.pair_class === :gaussian_bernoulli
        @test isfinite(s.eta) && abs(s.eta) < 1
        @test s.nobs == n
        @test occursin("conditional", s.uncertainty)
    end
end