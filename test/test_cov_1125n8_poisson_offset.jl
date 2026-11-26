# cov lane 1125n8 — Poisson offset helpers (src/poisson.jl #938).
using DRModels
using Test
using StatsModels
using Statistics: mean

function _argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

@testset "cov 1125n8: _extract_offset" begin
    rhs = (@formula(y ~ 1 + x + offset(z))).rhs
    fixed, off = DRModels._extract_offset(rhs)
    @test off !== nothing
    @test fixed isa StatsModels.Term || fixed isa Tuple
    rhs2 = (@formula(y ~ offset(a) + offset(b) + x)).rhs
    fixed2, off2 = DRModels._extract_offset(rhs2)
    @test off2 isa Tuple
    @test length(off2) == 2
    rhs3 = (@formula(y ~ offset(z))).rhs
    fixed3, off3 = DRModels._extract_offset(rhs3)
    @test off3 !== nothing
    @test fixed3 isa StatsModels.ConstantTerm
    rhs4 = (@formula(y ~ x)).rhs
    fixed4, off4 = DRModels._extract_offset(rhs4)
    @test off4 === nothing
    err = try
        DRModels._extract_offset((@formula(y ~ offset(a, b))).rhs)
        ""
    catch e
        e isa ErrorException ? e.msg : rethrow()
    end
    @test occursin("exactly one term", err)
end

@testset "cov 1125n8: _offset_vector and _offset_for_fit" begin
    n = 5
    data = (; y = randn(n), z = collect(1.0:5.0), log_e = log.(collect(1.0:5.0)))
    zterm = (@formula(y ~ z)).rhs
    v1 = DRModels._offset_vector(zterm, data)
    @test v1 == data.z
    rhs2, off2 = DRModels._extract_offset((@formula(y ~ offset(z) + offset(log_e))).rhs)
    v2 = DRModels._offset_vector(off2, data)
    @test v2 ≈ data.z .+ data.log_e
    off = DRModels._offset_for_fit(DRModels.Poisson(), zterm, data, n)
    @test off == data.z
    msg = _argmsg(() -> DRModels._offset_for_fit(DRModels.Poisson(), zterm, data, n - 1))
    @test occursin("one value per observation", msg)
end

@testset "cov 1125n8: _eta_with_offset and _offset_start" begin
    X = [ones(4) randn(4)]
    β = [1.0, 0.5]
    off = [0.1, -0.2, 0.3, 0.0]
    η0 = X * β
    @test DRModels._eta_with_offset(X, β, nothing) ≈ η0
    @test DRModels._eta_with_offset(X, β, off) ≈ η0 .+ off
    @test DRModels._add_offset(η0, nothing) ≈ η0
    @test DRModels._add_offset(η0, off) ≈ η0 .+ off
    θ = [2.0, 0.1]
    θs = DRModels._offset_start(θ, X, off)
    @test θs[1] ≈ 2.0 - mean(off)
    @test θs[2] == θ[2]
    θ2 = DRModels._offset_start(θ, X, nothing)
    @test θ2 == θ
end

@testset "cov 1125n8: _refuse_offset_unless_ordinary_intercept!" begin
    re = Tuple{Any,Any}[]
    msg = _argmsg(() -> DRModels._refuse_offset_unless_ordinary_intercept!(
        DRModels.Poisson(), re, :phylo, nothing))
    @test occursin("phylogenetic", msg) || occursin("structured", msg)
    @test DRModels._refuse_offset_unless_ordinary_intercept!(
        DRModels.Poisson(), re, nothing, nothing) === nothing
end