# cov lane 1125n7 — control scope weights + marginal method rejects (control + variational stubs).
using DRModels
using Test, Random

@testset "cov 1125n7: _ctl_weights under scope" begin
    delete!(task_local_storage(), DRModels._CTL_KEY)
    @test DRModels._ctl_weights(5) == ones(5)
    @test DRModels._ctl_weights_or_nothing(3) === nothing
    w = [2.0, 1.0, 3.0, 0.5, 4.0]
    scope = DRModels._ControlScope(DRMControl(), w, bf(@formula(y ~ 1), @formula(sigma ~ 1)),
                                   false, false, false, Float64[], 0, NamedTuple[], :none)
    @test task_local_storage(DRModels._CTL_KEY, scope) do
        ww = DRModels._ctl_weights(5)
        ww == w && DRModels._ctl_weights_or_nothing(5) == w
    end
    @test_throws ArgumentError task_local_storage(DRModels._CTL_KEY, scope) do
        DRModels._ctl_weights(5; supported = false, route = "cov lane probe")
    end
    @test_throws DimensionMismatch task_local_storage(DRModels._CTL_KEY, scope) do
        DRModels._ctl_weights(4)
    end
    delete!(task_local_storage(), DRModels._CTL_KEY)
end

@testset "cov 1125n7: _marginal_method symbols" begin
    @test DRModels._marginal_method(:LA) isa DRModels.Laplace
    @test DRModels._marginal_method(:la) isa DRModels.Laplace
    @test DRModels._marginal_method(:VA) isa DRModels.Variational
    @test DRModels._marginal_method(:AGHQ) isa DRModels.AGHQ
    @test_throws ArgumentError DRModels._marginal_method(:Laplace)
    @test_throws ArgumentError DRModels._marginal_method(:nope)
end

@testset "cov 1125n7: _reject_method_as_marginal" begin
    @test DRModels._reject_method_as_marginal(Poisson(), nothing) === nothing
    @test DRModels._reject_method_as_marginal(Poisson(), :ML) === :ML
    for bad in (:LA, :VA, :AGHQ, :Laplace)
        @test_throws ArgumentError DRModels._reject_method_as_marginal(Poisson(), bad)
    end
    @test_throws ArgumentError DRModels._reject_method_as_marginal(Poisson(), :REML)
    @test DRModels._reject_method_as_marginal(Gaussian(), :REML; allow_reml = true) === :REML
end

@testset "cov 1125n7: _aghq_reject and _va_reject copy" begin
    msg = try
        DRModels._aghq_reject(Gamma(), "phylo(1 | g)")
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("AGHQ", msg)
    msg2 = try
        DRModels._va_reject(Beta(), "crossed random intercepts")
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("VA", msg2)
    msg3 = try
        DRModels._reject_reml_route(Student(), "temporal()")
        ""
    catch e
        e isa ArgumentError ? e.msg : rethrow()
    end
    @test occursin("REML", msg3)
end

@testset "cov 1125n7: _fit_va stub" begin
    @test_throws ErrorException DRModels._fit_va()
end

@testset "cov 1125n7: _kwname on Expr" begin
    @test DRModels._kwname(Expr(:kw, :data, :x)) == :data
    @test DRModels._kwname(:(x::Vector)) == :x
    @test DRModels._kwname(:(kwargs...)) == :...
end