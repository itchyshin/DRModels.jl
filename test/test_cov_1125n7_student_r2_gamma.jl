# cov lane 1125n7 — student logpdf, R² markers, gamma SD targets (src/student.jl, r2.jl, gamma.jl).
using DRModels
using Test, StatsModels, Distributions, Random

@testset "cov 1125n7: _student_logpdf_std limits" begin
    z = 0.5
    for ην in (0.0, 3.0, 8.0, 15.0)
        lp = DRModels._student_logpdf_std(z, ην)
        @test isfinite(lp)
        ν = 2 + exp(ην)
        ref = logpdf(TDist(ν), z)
        @test lp ≈ ref rtol = ην > 7 ? 5e-9 : 1e-6
    end
    # Large ν → Normal
    lp_hi = DRModels._student_logpdf_std(0.0, 20.0)
    @test lp_hi ≈ logpdf(Normal(0, 1), 0.0) rtol = 1e-6
end

@testset "cov 1125n7: _student_nu_boundary on synthetic scales" begin
    fit_lo = DrmFit(Student(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [1.0;;],
                    -1.0, 5, true, Dict(:mu => ones(5)), Dict(:mu => ones(5)),
                    Dict(:sigma => ones(5), :nu => fill(1e4, 5)))
    @test DRModels._student_nu_boundary(fit_lo) ≈ 1e4
    fit_hi = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [1.0;;],
                    -1.0, 5, true, Dict(:mu => ones(5)), Dict(:mu => ones(5)), Dict(:sigma => ones(5)))
    @test DRModels._student_nu_boundary(fit_hi) === nothing
end

@testset "cov 1125n7: _r2_marker on formula terms" begin
    bar = (@formula(y ~ (1 | g))).rhs
    @test DRModels._r2_marker(bar) == "a random effect"
    mv = (@formula(y ~ meta_V(v))).rhs
    @test occursin("meta_V", DRModels._r2_marker(mv))
    phy = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ phylo(1 | sp))")).rhs
    @test occursin("phylo", DRModels._r2_marker(phy))
    ttmp = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ temporal(1 | id, t, ar1))")).rhs
    @test occursin("temporal", DRModels._r2_marker(ttmp))
    @test DRModels._r2_marker((@formula(y ~ x)).rhs) === nothing
end

@testset "cov 1125n7: _r2_random_part on formula-backed fit" begin
    f_re = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    fit_re = DrmFit(Gaussian(), [:mu => 1:2], [:mu => ["(Intercept)", "x"]], zeros(2),
                    [0.1 0.0; 0.0 0.1], -1.0, 10, true,
                    Dict(:mu => ones(10)), Dict(:mu => ones(10)), Dict(:sigma => ones(10)),
                    f_re, nothing, nothing, nothing, :ML, NaN, -1.0, :LA)
    @test DRModels._r2_random_part(fit_re) == "a random effect"
    f_fe = bf(@formula(y ~ x), @formula(sigma ~ 1))
    fit_fe = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [0.1;;],
                    -1.0, 5, true, Dict(:mu => ones(5)), Dict(:mu => ones(5)),
                    Dict(:sigma => ones(5)),
                    f_fe, nothing, nothing, nothing, :ML, NaN, -1.0, :LA)
    @test DRModels._r2_random_part(fit_fe) === nothing
end

@testset "cov 1125n7: r2_constant_sigma on synthetic fit" begin
    y = [1.0, 2.5, 3.0, 4.5]
    ŷ = [1.1, 2.4, 3.1, 4.3]
    fit = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [0.1;;],
                 -2.0, length(y), true, Dict(:mu => y), Dict(:mu => ŷ), Dict(:sigma => ones(4)))
    r2 = r2_constant_sigma(fit)
    @test 0 < r2 < 1
    fit_no_int = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["x"]], [0.5], [0.1;;],
                        -2.0, 4, true, Dict(:mu => y), Dict(:mu => ŷ), Dict(:sigma => ones(4)))
    @test_throws ArgumentError r2_constant_sigma(fit_no_int)
end

@testset "cov 1125n7: _gamma_sd_targets ordinary vs sd()" begin
    f_plain = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    t2 = DRModels._gamma_sd_targets(f_plain)
    @test t2 == DRModels._sd_target_parts(f_plain)
    f_sd = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ 1))
    t3 = DRModels._gamma_sd_targets(f_sd)
    @test length(t3) >= 1
    @test all(t -> t.target.group == :g, t3)
end

@testset "cov 1125n7: _warn_student_nu_boundary quiet guard" begin
    fit = DrmFit(Student(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], [1.0;;],
                 -1.0, 5, true, Dict(:mu => ones(3)), Dict(:mu => ones(3)),
                 Dict(:sigma => ones(3), :nu => fill(1e4, 3)))
    out = DRModels._without_boundary_warnings() do
        DRModels._warn_student_nu_boundary(fit)
    end
    @test out === fit
end