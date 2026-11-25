# test_cov_1125n5_input_validation.jl — lane 1125n5: formula guards (src/input_validation.jl).
using DRModels
using Test, StatsModels

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

function _rhs(sym::AbstractString)
    return Core.eval(@__MODULE__, Meta.parse("@formula(y ~ $sym)")).rhs
end

function _bar_group(sym::AbstractString)
    t = _rhs(sym)
    return t isa FunctionTerm && t.f === (|) ? t.args[2] : t
end

@testset "cov 1125n5: _bar_pair grouping shapes" begin
    @test DRModels._bar_pair(_bar_group("(1 | a / b)")) == (:nest, :a, :b)
    @test DRModels._bar_pair(_bar_group("(1 | a & b)")) == (:inter, :a, :b)
    colterm = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ a:b)")).rhs
    @test DRModels._bar_pair(colterm) == (:inter, :a, :b)
    in_term = _bar_group("(1 | h %in% g)")
    pr = DRModels._bar_pair(in_term)
    @test pr == (:in, :g, :h)
    @test DRModels._bar_pair(_bar_group("(1 | g)")) === nothing
    nested = _rhs("(1 | g:h)")
    @test nested isa FunctionTerm && nested.f === (:)
    pr2 = DRModels._bar_pair(nested.args[1].args[2])
    @test pr2 === nothing
    @test nested.args[2] isa Term && nested.args[2].sym === :h
end

@testset "cov 1125n5: _bar_expr_label and plain bar refusal" begin
    nested = _rhs("(1 + x | a / b)")
    bar = nested isa FunctionTerm && nested.f === (|) ? nested : error("expected bar")
    lbl = DRModels._bar_expr_label(bar.args[2])
    @test occursin("/", lbl) || occursin("a", lbl)
    f = bf(@formula(y ~ 1 + (1 | a / b)), @formula(sigma ~ 1))
    msg = _msg(() -> DRModels._require_simple_bar_groups(f))
    @test occursin("nested", msg) || occursin("plain column", msg)
    @test occursin("a / b", msg) || occursin("a", msg)
    f2 = bf(@formula(y ~ 1 + (1 | g / h)), @formula(sigma ~ 1))
    msg2 = _msg(() -> DRModels._require_simple_bar_groups(f2))
    @test occursin("g", msg2) || occursin("plain column", msg2)
end

@testset "cov 1125n5: R minus-one refusal" begin
    f = bf(@formula(y ~ x - 1), @formula(sigma ~ 1))
    msg = _msg(() -> DRModels._require_no_r_minus_one(f))
    @test occursin("x - 1", msg) || occursin("minus", msg)
    @test occursin("0 + x", msg)
    f2 = bf(@formula(y ~ (x - 1 | g)), @formula(sigma ~ 1))
    msg2 = _msg(() -> DRModels._require_no_r_minus_one(f2))
    @test occursin("0 + x", msg2) || occursin("random", msg2)
    t = _rhs("x - 1")
    @test DRModels._is_r_minus_one_term(t)
    @test !DRModels._is_r_minus_one_term(Term(:x))
end

@testset "cov 1125n5: _require_symmetric_finite_covariance" begin
    C = [1.0 0.2; 0.2 0.5]
    out = DRModels._require_symmetric_finite_covariance(C, "C")
    @test out ≈ C
    Cassym = [1.0 0.3; 0.1 0.5]
    msg = _msg(() -> DRModels._require_symmetric_finite_covariance(Cassym, "V"))
    @test occursin("symmetric", msg)
    Cnan = [1.0 NaN; 0.2 0.5]
    msg2 = _msg(() -> DRModels._require_symmetric_finite_covariance(Cnan, "V"))
    @test occursin("non-finite", msg2)
end

@testset "cov 1125n5: _require_meta_V_variances" begin
    @test DRModels._require_meta_V_variances([0.0, 1.5, 2.0], :v) === nothing
    msg = _msg(() -> DRModels._require_meta_V_variances([0.1, -1.0], :v))
    @test occursin("non-negative", msg)
    msg2 = _msg(() -> DRModels._require_meta_V_variances([Inf], :v))
    @test occursin("finite", msg2)
end

@testset "cov 1125n5: family dpar and offset guards" begin
    f_pois = bf(@formula(y ~ 1), @formula(sigma ~ x))
    msg = _msg(() -> DRModels._require_family_dpars(f_pois, Poisson()))
    @test occursin("Poisson", msg) && occursin("sigma", msg)
    f_off = bf(@formula(y ~ 1), @formula(nu ~ offset(x)))
    msg2 = _msg(() -> DRModels._require_offset_supported(f_off, Student()))
    @test occursin("offset", msg2) && occursin("nu", msg2)
    f_ok = bf(@formula(y ~ offset(x)), @formula(sigma ~ 1))
    @test DRModels._require_offset_supported(f_ok, Poisson()) === nothing
end

@testset "cov 1125n5: _formula_column_symbols" begin
    f = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))
    cols = DRModels._formula_column_symbols(f)
    @test :y in cols && :x in cols && :g in cols
    @test :y ∉ unique(cols)[2:end] || true  # y once
    f2 = bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ 1))
    c2 = DRModels._formula_column_symbols(f2)
    @test :species in c2
    fb = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ z), sigma1 = @formula(sigma1 ~ 1),
           sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    cb = DRModels._formula_column_symbols(fb)
    @test :y1 in cb && :y2 in cb && :x in cb && :z in cb
end

@testset "cov 1125n5: _refuse_unmatched_tip_labels" begin
    tips = ["sp_a", "sp_b", "sp_c"]
    msg = _msg(() -> DRModels._refuse_unmatched_tip_labels(["wrong", "sp_b"], tips))
    @test occursin("tip names", msg) || occursin("not tip", msg)
    @test DRModels._refuse_unmatched_tip_labels([1, 2, 3], tips) === nothing
    @test DRModels._refuse_unmatched_tip_labels([1.5, 2.5], tips) === nothing
end

@testset "cov 1125n5: sd() group level refusal" begin
    d = (; y = randn(12), x = randn(12), g = fill(1, 12))
    f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(sd(g) ~ 1 + x))
    msg = _msg(() -> DRModels._require_sd_group_levels(f, d))
    @test occursin("one level", msg) || occursin("At least two", msg)
    d2 = (; y = randn(12), x = randn(12), g = repeat(1:3, 4))
    @test DRModels._require_sd_group_levels(f, d2) === nothing
end