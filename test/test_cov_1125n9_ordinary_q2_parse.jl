# cov lane 1125n9 — ordinary correlated SD block detection (src/ordinary_q2_sd.jl).
using DRModels
using Test, Random, StatsModels, Distributions

function _mu_rhs(sym::AbstractString)
    Core.eval(@__MODULE__, Meta.parse("@formula(y ~ $sym)")).rhs
end

@testset "cov 1125n9: _ordinary_q2_block" begin
    u = _mu_rhs("x + (1 + x1 | g)")
    b = DRModels._ordinary_q2_block(u)
    @test b !== nothing
    @test b.group == :g && b.slope == :x1 && b.label === nothing
    l = _mu_rhs("x + (1 + x1 | p | g)")
    bl = DRModels._ordinary_q2_block(l)
    @test bl.label == :p && bl.group == :g
    @test DRModels._ordinary_q2_block(_mu_rhs("x + (1 | g)")) === nothing
    @test DRModels._ordinary_q2_block(_mu_rhs("x + (0 + x1 | g)")) === nothing
    @test DRModels._ordinary_q2_block(nothing) === nothing
    t = _mu_rhs("1")
    @test DRModels._ordinary_q2_block(t) === nothing
end

@testset "cov 1125n9: _ordinary_q2_sd_block on formulas" begin
    fp = bf(@formula(y ~ x + (1 + x1 | g)),
            @formula(sd(g, coef(intercept)) ~ z0),
            @formula(sd(g, coef(x1)) ~ z1))
    @test DRModels._ordinary_q2_sd_block(fp, DRModels.Poisson()) !== nothing
    fn = bf(@formula(y ~ x + (1 + x1 | p | g)),
            @formula(sd(g, coef(intercept)) ~ z0))
    @test DRModels._ordinary_q2_sd_block(fn, DRModels.NegBinomial2()) !== nothing
    f0 = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))
    @test DRModels._ordinary_q2_sd_block(f0, DRModels.Poisson()) === nothing
    f_phy = bf(@formula(y ~ x + phylo(1 | g)), @formula(sigma ~ 1))
    @test DRModels._ordinary_q2_sd_block(f_phy, DRModels.Poisson()) === nothing
end

@testset "cov 1125n9: ordinary q2 route refusals" begin
    Random.seed!(1125)
    n = 60
    g = repeat(1:4, outer = 15)
    x1 = randn(n)
    z0 = randn(n)
    y = rand.(Distributions.Poisson.(exp.(0.2 .+ 0.3 .* x1)))
    d = (; y = Float64.(y), x1, z0, g)
    f = bf(@formula(y ~ x1 + (1 + x1 | g)),
           @formula(sd(g, coef(intercept)) ~ z0),
           @formula(sd(g, coef(x1)) ~ z0))
    msg = try
        drm(f, DRModels.Poisson(); data = d, tree = ones(4, 4))
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("ordinary", msg) || occursin("tree", msg)
    f2 = bf(@formula(y ~ x1 + (1 + x1 | g) + (1 | h)),
            @formula(sd(g, coef(intercept)) ~ z0))
    msg2 = try
        drm(f2, DRModels.Poisson(); data = merge(d, (; h = repeat(1:3, outer = 20))), tree = nothing)
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test !isempty(msg2)
    f3 = bf(@formula(y ~ x1 + (1 + x1 | g)),
            @formula(sd(g, coef(intercept)) ~ z0),
            @formula(zi ~ 1))
    msg3 = try
        drm(f3, DRModels.Poisson(); data = d)
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("zi", msg3) || occursin("hu", msg3)
end