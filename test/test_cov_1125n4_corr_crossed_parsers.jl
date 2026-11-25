# test_cov_1125n4_corr_crossed_parsers.jl — crossed Gaussian RE parsing (src/gaussian_corr_crossed_sd.jl).
using DRModels
using Test, StatsModels, Random

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

function _re_lhs(sym::AbstractString)
    t = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ $sym)"))
    rhs = t.rhs
    return rhs isa FunctionTerm && rhs.f === (|) ? rhs.args[1] : rhs
end

@testset "cov 1125n4: _cc_term_kind" begin
    i1 = _re_lhs("1")
    @test DRModels._cc_term_kind(i1) == (:intercept, nothing)
    s = _re_lhs("(0 + x | g)")
    @test DRModels._cc_term_kind(s) == (:slope, :x)
    blk = _re_lhs("(1 + x | g)")
    @test DRModels._cc_term_kind(blk) === nothing
    lab = _re_lhs("(1 + x | tag | g)")
    @test DRModels._cc_term_kind(lab) === nothing
end

@testset "cov 1125n4: _corr_crossed_split" begin
    blk = _re_lhs("(1 + x1 | g)")
    i1 = _re_lhs("1")
    re = [(blk, :g), (i1, :h)]
    sp = DRModels._corr_crossed_split(re)
    @test sp !== nothing
    @test sp[1] == [1]
    @test sp[2] == [2]
    re2 = [(blk, :g)]
    @test DRModels._corr_crossed_split(re2) === nothing
    s2 = _re_lhs("(0 + z | h)")
    re3 = [(blk, :g), (s2, :h), (i1, :k)]
    sp3 = DRModels._corr_crossed_split(re3)
    @test sp3 !== nothing
    @test length(sp3[1]) == 1 && length(sp3[2]) == 2
    two_blk = [(_re_lhs("(1 + x1 | g)"), :g), (_re_lhs("(1 + x1 | h)"), :h)]
    @test DRModels._corr_crossed_split(two_blk) == ([1, 2], Int[])
end

@testset "cov 1125n4: _re_multi_corr_vars (q34 block)" begin
    b2 = _re_lhs("(1 + x1 | g)")
    @test DRModels._re_multi_corr_vars(b2) == [:x1]
    b3 = _re_lhs("(1 + x1 + x2 | g)")
    @test DRModels._re_multi_corr_vars(b3) == [:x1, :x2]
    b4 = _re_lhs("(1 + x1 + x2 + x3 | g)")
    @test length(DRModels._re_multi_corr_vars(b4)) == 3
    @test DRModels._re_multi_corr_vars(_re_lhs("1")) === nothing
    @test DRModels._re_multi_corr_vars(_re_lhs("(1 | g)")) === nothing
end

@testset "cov 1125n4: Gaussian crossed block fit (tiny)" begin
    Random.seed!(1125)
    n = 120
    g = repeat(1:6, outer = 20)
    g2 = repeat(1:4, outer = 30)
    x1 = randn(n)
    y = 0.2 .+ 0.4 .* x1 .+ randn(n) .* 0.5
    d = (; y, x1, g, g2)
    fit = drm(bf(@formula(y ~ 1 + x1 + (1 + x1 | g) + (1 | g2)), @formula(sigma ~ 1)),
              Gaussian(); data = d)
    @test isfinite(loglik(fit))
    @test loglik(fit) > -1e5
    @test nobs(fit) == n
    @test length(coef(fit)) > 4
end

@testset "cov 1125n4: crossed-route refusals" begin
    d = (; y = randn(40), x1 = randn(40), g = repeat(1:4, 10), g2 = repeat(1:5, 8))
    msg = _msg(() -> drm(bf(@formula(y ~ 1 + x1 + (1 + x1 | g) + (1 | g2) + phylo(1 | g)),
                          @formula(sigma ~ 1)), Gaussian(); data = d, tree = ones(4, 4)))
    @test occursin("correlated", msg) || occursin("phylo", msg)
    msg2 = _msg(() -> drm(bf(@formula(y ~ 1 + (1 + x1 | g) + (1 | g)),
                           @formula(sigma ~ 1 + x1)), Gaussian(); data = d))
    @test !isempty(msg2)
    msg3 = _msg(() -> drm(bf(@formula(y ~ 1 + (1 + x1 | g) + (1 | g)),
                            @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML))
    @test !isempty(msg3)  # REML allowed for intercept-only extras; refusal must be explicit when thrown
end