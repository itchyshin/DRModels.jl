# Constructor regression for bf() calls in public Julia DocString examples.
# Extracts the package documentation and checks syntax without fitting models.
using Test
using DRModels
using Markdown

const DOC_BINDINGS = (
    :drm, :Gaussian, :relmat, :Beta, :BetaBinomial, :Binomial, :BivariateDrmFormula, :CumulativeLogit,
    :Gamma, :LogNormal, :NegBinomial2, :TruncatedNegBinomial2,
    :drm_listwise, :drm_phylo_penalty, :drm_phylo_penalty_sweep,
    :Poisson, :SkewNormal, :Student, :TruncatedPoisson, :Tweedie,
    :ZeroOneBeta,
)

function _julia_code!(out, x)
    if x isa Markdown.Code
        lowercase(x.language) == "julia" && push!(out, x.code)
    elseif x isa AbstractVector
        foreach(y -> _julia_code!(out, y), x)
    elseif x isa Markdown.MD
        foreach(y -> _julia_code!(out, y), x.content)
    elseif !(x isa AbstractString || x isa Symbol || x isa Number || x === nothing)
        for i in 1:fieldcount(typeof(x))
            _julia_code!(out, getfield(x, i))
        end
    end
    return out
end

function _bf_calls!(out, ex)
    if ex isa Expr
        if ex.head === :call && !isempty(ex.args) && ex.args[1] === :bf
            push!(out, ex)
            return out # evaluate this constructor once; its nested args are formulas
        end
        foreach(x -> _bf_calls!(out, x), ex.args)
    end
    return out
end

@testset "DocString bf() examples construct formula objects without fitting" begin
    n_calls = 0
    for name in DOC_BINDINGS
        isdefined(DRModels, name) || continue
        value = getfield(DRModels, name)
        doc = Docs.doc(value)
        doc === nothing && continue
        blocks = _julia_code!(String[], doc)
        for block in blocks
            parsed = Meta.parseall(block)
            calls = _bf_calls!(Expr[], parsed)
            for call in calls
                result = Core.eval(@__MODULE__, call)
                @test result isa DRModels.DrmFormula || result isa DRModels.BivariateDrmFormula
                n_calls += 1
            end
        end
    end
    @test n_calls >= 22 # covers the bodies repaired in this documentation patch
end

@testset "Formula example negative controls" begin
    @test bf(@formula(y ~ x), @formula(sigma ~ 1)) isa DRModels.DrmFormula
    @test_throws ArgumentError bf(@formula(y1 ~ x), @formula(y2 ~ x), @formula(sigma1 ~ 1), @formula(sigma2 ~ 1), @formula(rho12 ~ 1))
end

@testset "Integrator refusal hints use route-specific defaults" begin
    blank = DrmFit(Gaussian(), Pair{Symbol,UnitRange{Int}}[],
        Pair{Symbol,Vector{String}}[], Float64[], zeros(0, 0), 0.0, 0, true,
        Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}(),
        Dict{Symbol,Vector{Float64}}())
    laplace = DRModels._withmarginal(blank, :Laplace)
    calls = [
        (() -> DRModels._marginal_compare_guard(blank, laplace, "lrtest"), "route-specific default integrator"),
        (() -> DRModels._fit_va(), "documented default integrator"),
        (() -> DRModels._va_reject(Poisson(), "multiple grouping factors"), "documented default integrator"),
        (() -> DRModels._reject_method_as_marginal(Poisson(), :VA), "documented default integrator"),
        (() -> DRModels._reject_method_as_marginal(Poisson(), :unknown), "documented default integrator"),
    ]
    for (call, wording) in calls
        err = try
            call()
            nothing
        catch caught
            caught
        end
        @test err isa Exception
        message = sprint(showerror, err)
        @test occursin(wording, message)
        @test !occursin("GHQ-32", message)
    end
end

@testset "Invalid starting vector hint names the retained coefficient labels" begin
    f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))
    data = (y = [0.1, 0.4, 0.8, 1.2], x = [0.0, 1.0, 2.0, 3.0])
    err = try
        drm(f, Gaussian(); data, control = DRMControl(start = [0.0]))
        nothing
    catch caught
        caught
    end
    @test err isa ArgumentError
    message = sprint(showerror, err)
    @test occursin("fit.coefnames", message)
    @test !occursin("coefnames(fit)", message)
end
