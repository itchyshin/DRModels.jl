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

@testset "bivariate REML bootstrap refit keeps method" begin
    # Cheap stand-in for a bivariate Gaussian REML seed. `_is_gaussian_lss`
    # requires a univariate `DrmFormula`, so this seed takes the REML branch
    # of `_bootstrap_refit_kwargs` and must still forward `method => :REML`.
    fit = DRModels.DrmFit(DRModels.Gaussian(),
                          [:mu => 1:1],
                          [:mu => ["(Intercept)"]],
                          [0.0], fill(1.0, 1, 1), -1.0, 4, true,
                          Dict(:mu => [1.0]), Dict(:mu => [1.0]),
                          Dict{Symbol,Vector{Float64}}())
    formula = DRModels.BivariateDrmFormula(:y1, :y2,
        Pair{Symbol,Any}[:mu1 => :y1, :mu2 => :y2])
    seed = DRModels._withreml(DRModels._withformula(fit, formula), -2.0, -1.0)
    @test seed.formula isa DRModels.BivariateDrmFormula
    @test !DRModels._is_gaussian_lss(seed)

    kw = DRModels._bootstrap_refit_kwargs(seed)
    @test kw.method === :REML
    @test !haskey(kw, :marginal)
    passed = DRModels._bootstrap_bivariate_refit_kwargs(kw)
    @test passed.method === :REML
    @test keys(passed) == (:method,)

    # A non-default marginal is recorded for the univariate refit and must
    # stay off the bivariate call.
    va = DRModels._withmarginal(seed, :VA)
    kw_va = DRModels._bootstrap_refit_kwargs(va)
    @test kw_va.method === :REML
    @test kw_va.marginal === :VA
    passed_va = DRModels._bootstrap_bivariate_refit_kwargs(kw_va)
    @test passed_va == (; method = :REML)
    @test !haskey(passed_va, :marginal)

    ml = DRModels._withformula(fit, formula)
    ml_kw = DRModels._bootstrap_refit_kwargs(ml)
    @test !haskey(ml_kw, :method)
    @test DRModels._bootstrap_bivariate_refit_kwargs(ml_kw) == NamedTuple()

    # The bivariate refit closure splats that helper and does not splat the
    # full `refit_options` tuple (which can contain `marginal`).
    src = read(joinpath(@__DIR__, "..", "src", "inference.jl"), String)
    branch = match(r"if formula isa BivariateDrmFormula\n(.*?\n)    else"s, src)
    @test branch !== nothing
    body = branch.captures[1]
    @test occursin("_bootstrap_bivariate_refit_kwargs(refit_options)", body)
    @test occursin(
        "drm(formula, fit.family; data=datab, K, A, tree, coords, g_tol, biv_kw...)",
        body)
    @test !occursin("refit_options...", body)
    @test !occursin("marginal", body)
    @test !occursin("algorithm", body)
end
