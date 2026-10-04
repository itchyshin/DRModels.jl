# DRModels.jl#1038 and #1025: bootstrap_result used to refit a :VA or :AGHQ
# seed as the default :LA integrator, and a REML seed as ML, except for the
# single case marginal = :Laplace. The interval would then describe a different
# estimator from the point estimate. This checks the keywords, not a long
# bootstrap, and does not claim the intervals are calibrated.

using DRModels, Test

function _boot_stub()
    DRModels.DrmFit(Poisson(),
                    [:mu => 1:1],
                    [:mu => ["(Intercept)"]],
                    [0.0], fill(1.0, 1, 1), -1.0, 4, true,
                    Dict(:mu => [1.0]), Dict(:mu => [1.0]),
                    Dict{Symbol,Vector{Float64}}())
end

@testset "bootstrap refit keeps marginal and REML (#1038, #1025)" begin
    base = _boot_stub()
    plain = DRModels._bootstrap_refit_kwargs(base)
    @test !haskey(plain, :marginal)
    @test !haskey(plain, :method)

    for m in (:VA, :AGHQ, :Laplace)
        kw = DRModels._bootstrap_refit_kwargs(DRModels._withmarginal(base, m))
        @test kw.marginal === m
        @test !haskey(kw, :method)
    end

    reml = DRModels._withreml(base, -2.0, -1.0)
    @test DRModels._bootstrap_refit_kwargs(reml).method === :REML

    both = DRModels._withmarginal(reml, :VA)
    kw = DRModels._bootstrap_refit_kwargs(both)
    @test kw.marginal === :VA
    @test kw.method === :REML
end
