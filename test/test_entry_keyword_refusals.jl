# An unsupported keyword is a readable refusal, never a raw MethodError
# (docs/dev-log/parity-crosswalk-evidence-20261107/fuzz-gap-map: 729 "crash-Julia" cells, all one
# uncaught MethodError). The grid below is the keyword x family set of crash_cells.tsv:
#   LogNormal / Student / Tweedie  x  method, tree, K, coords, A + ids (and `method` with each).
# Plus the splat routes (Gaussian, Poisson), whose fitters raise the MethodError one level down.
using DRModels, Test, LinearAlgebra, StableRNGs
import Distributions

function _kw_data(; G = 8, m = 5, seed = 20261108)
    rng = StableRNG(seed)
    g = repeat(["g$i" for i in 1:G], inner = m)
    x = randn(rng, G * m)
    y = exp.(0.2 .+ 0.3 .* x .+ 0.3 .* randn(rng, G * m))
    return (; y, x, g)
end

# what each marker needs, as the fuzz-gap-map generated it
const _KW_MARKER = Dict(:method => "(1 | g)", :tree => "phylo(1 | g)", :K => "relmat(1 | g)",
                        :coords => "spatial(1 | g)", :A => "animal(1 | g)")

function _kw_formula(marker)
    marker === :method && return bf(@formula(y ~ x + (1 | g)))
    marker === :tree && return bf(@formula(y ~ x + phylo(1 | g)))
    marker === :K && return bf(@formula(y ~ x + relmat(1 | g)))
    marker === :coords && return bf(@formula(y ~ x + spatial(1 | g)))
    return bf(@formula(y ~ x + animal(1 | g)))
end

function _kw_values(kws, d)
    G = length(unique(d.g))
    nt = (;)
    :method in kws && (nt = merge(nt, (; method = :REML)))
    :tree in kws && (nt = merge(nt, (; tree = "(" * join(["g$i:1" for i in 1:G], ",") * ");")))
    :K in kws && (nt = merge(nt, (; K = Matrix(1.0 * I(G)))))
    :coords in kws && (nt = merge(nt, (; coords = Dict("g$i" => [Float64(i), 0.0] for i in 1:G))))
    :A in kws && (nt = merge(nt, (; A = Matrix(1.0 * I(G)))))
    :ids in kws && (nt = merge(nt, (; ids = ["g$i" for i in 1:G])))
    return nt
end

# keywords each route does accept (so they must NOT be named as unsupported)
const _KW_ACCEPTED = Dict(:LogNormal => [:tree, :K], :Student => [:coords],
                          :Tweedie => [:tree, :K, :A, :coords])

# (family, keyword set) cells from crash_cells.tsv, as listed in the fuzz-gap-map README
const _KW_CELLS = [
    (LogNormal(), [:A, :ids]), (LogNormal(), [:coords]), (LogNormal(), [:method]),
    (LogNormal(), [:method, :A, :ids]), (LogNormal(), [:method, :K]),
    (LogNormal(), [:method, :coords]), (LogNormal(), [:method, :tree]),
    (Student(), [:A, :ids]), (Student(), [:K]), (Student(), [:coords]), (Student(), [:method]),
    (Student(), [:method, :A, :ids]), (Student(), [:method, :K]), (Student(), [:method, :coords]),
    (Student(), [:method, :tree]), (Student(), [:tree]),
    (Tweedie(), [:A, :ids]), (Tweedie(), [:method]), (Tweedie(), [:method, :A, :ids]),
    (Tweedie(), [:method, :K]), (Tweedie(), [:method, :coords]), (Tweedie(), [:method, :tree]),
]

function _kw_marker_of(kws)
    for k in (:tree, :K, :coords, :A)
        k in kws && return k
    end
    return :method
end

@testset "unsupported keyword => ArgumentError naming keyword, route and remedy (not MethodError)" begin
    d = _kw_data()
    for (fam, kws) in _KW_CELLS
        kw = _kw_values(kws, d)
        f = _kw_formula(_kw_marker_of(kws))
        if fam isa Student && kws == [:coords]
            # Student `spatial(1 | g)` with `coords` is now a fitted route (test_student_spatial_coords.jl),
            # so this cell no longer refuses: assert it fits.
            @test drm(f, fam; data = d, kw...) isa DRModels.DrmFit
            continue
        end
        err = try
            drm(f, fam; data = d, kw...)
            nothing
        catch e
            e
        end
        label = "$(nameof(typeof(fam))) $(join(kws, ","))"
        @testset "$label" begin
            @test !(err isa MethodError)
            @test err isa ArgumentError
            msg = err isa ArgumentError ? err.msg : sprint(showerror, err)
            fam_name = string(nameof(typeof(fam)))
            @test occursin("the $fam_name route does not accept", msg)
            head = first(split(msg, "Keywords this route accepts"))
            unsupported = [k for k in kws if !(k in _KW_ACCEPTED[nameof(typeof(fam))])]
            @test !isempty(unsupported)
            for k in unsupported
                @test occursin("`$k`", head)
            end
            for k in _KW_ACCEPTED[nameof(typeof(fam))]
                k in kws && @test !occursin("keyword `$k`", head) && !occursin("keywords `$k`", head)
            end
            @test occursin("drmTMB", msg)                    # says what R does
            @test occursin("Keywords this route accepts", msg)   # says what to try
            @test occursin("refused before any fitting", msg)
        end
    end
end

@testset "the cell count of the fuzz-gap-map grid" begin
    # every (family, keyword set) cell above was a MethodError in crash_cells.tsv; none is now
    @test length(_KW_CELLS) == 22
end

@testset "splat routes (fitter-level keyword MethodError) are refused too" begin
    d = _kw_data()
    for (fam, kw) in ((Gaussian(), (; bogus = 1)), (Poisson(), (; bogus = 1)),
                      (DRModels.Gamma(), (; method = :REML, bogus = 2)))
        err = try drm(bf(@formula(y ~ x + (1 | g))), fam; data = d, kw...); nothing catch e; e end
        @test err isa ArgumentError
        @test !(err isa MethodError)
        @test occursin("`bogus`", err.msg)
    end
    # a misspelled keyword on a plain fixed-effect Gaussian call
    err = try drm(bf(@formula(y ~ x)), Gaussian(); data = d, metod = :REML); nothing catch e; e end
    @test err isa ArgumentError && occursin("`metod`", err.msg)
end

@testset "calls that work today are unchanged" begin
    d = _kw_data()
    # accepted keywords, including the `@drm_entry` ones, still fit
    f0 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = d)
    f1 = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), LogNormal(); data = d, g_tol = 1e-8, marginal = :LA,
             weights = ones(length(d.y)))
    @test f0.converged && f1.converged
    @test f0.theta ≈ f1.theta
    @test loglikelihood(f0) == loglikelihood(f1) || isapprox(loglikelihood(f0), loglikelihood(f1); atol = 1e-9)
    # Student, Tweedie fixed-effect fits
    ds = merge(d, (; y = d.y .- 1.0))
    @test drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Student(); data = ds) isa DRModels.DrmFit
    @test drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1)), Tweedie(); data = d) isa DRModels.DrmFit
    # Gaussian REML is still a supported keyword
    @test drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML) isa DRModels.DrmFit
    # a MethodError that is NOT a keyword problem is not rewritten: wrong positional types stay a MethodError
    @test_throws MethodError drm(bf(@formula(y ~ x)), "not a family"; data = d)
end
