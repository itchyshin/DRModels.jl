# #1303: CumulativeLogit() must refuse a response whose codes skip a level
# (y ∈ {1, 2, 4}) instead of allocating a cutpoint for the unobserved code.
# The guard is in `_drm_cumulative_fit`, before the fixed / random / phylo /
# crossed split, so every CumulativeLogit route shares it. The ordinal
# missing-predictor path (`_joint_finite_numeric_levels`) already requires
# contiguous 1:K. CategoricalLogit is a nominal predictor family, not an
# ordinal response family, and does not use this fit path.
using DRModels
using Test

const _X1303 = [0.3, -0.2, 1.1, 0.5, -0.7, 0.1, -1.2, 0.8]
# Issue MRE: categories 1, 2 and 4; level 3 never occurs.
const _Y1303 = [1.0, 2.0, 4.0, 4.0, 2.0, 1.0, 4.0, 2.0]

function _err1303(f)
    try
        f()
        error("expected an ArgumentError, but drm() returned a fit")
    catch e
        e isa ArgumentError && return e
        rethrow()
    end
end

@testset "Fix #1303 skipped ordinal category" begin
    @testset "MRE: y in {1, 2, 4} is refused" begin
        err = _err1303() do
            drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = _Y1303, x = _X1303))
        end
        @test occursin("missing level(s): 3", err.msg)
        @test occursin("observed levels: 1, 2, 4", err.msg)
        @test occursin("Response `y`", err.msg)
        @test occursin("appear at least once", err.msg)

        # The same hole in integer storage, not only Float64.
        err_int = _err1303() do
            drm(bf(@formula(y ~ x)), CumulativeLogit();
                data = (; y = [1, 2, 4, 4, 2, 1, 4, 2], x = _X1303))
        end
        @test occursin("missing level(s): 3", err_int.msg)
        @test occursin("observed levels: 1, 2, 4", err_int.msg)

        # Issue sibling: only level 2 is observed, so level 1 is missing.
        err_one = _err1303() do
            drm(bf(@formula(y ~ x)), CumulativeLogit();
                data = (; y = fill(2.0, length(_X1303)), x = _X1303))
        end
        @test occursin("missing level(s): 1", err_one.msg)
        @test occursin("observed levels: 2", err_one.msg)

        # Several missing codes are named (a range), not collapsed into a fit.
        err_range = _err1303() do
            drm(bf(@formula(y ~ x)), CumulativeLogit();
                data = (; y = [1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 4.0, 1.0], x = _X1303))
        end
        @test occursin("missing level(s): 2:3", err_range.msg)
        @test occursin("observed levels: 1, 4", err_range.msg)

        # Shared entry: a random-intercept formula is refused before quadrature.
        g = [1, 1, 2, 2, 3, 3, 4, 4]
        err_re = _err1303() do
            drm(bf(@formula(y ~ x + (1 | g))), CumulativeLogit();
                data = (; y = _Y1303, x = _X1303, g))
        end
        @test occursin("missing level(s): 3", err_re.msg)
        @test occursin("observed levels: 1, 2, 4", err_re.msg)
    end

    @testset "contiguous levels still fit" begin
        # The MRE predictor, with the hole filled so the codes are 1, 2, 3.
        y3 = [1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 3.0, 2.0]
        fit3 = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = y3, x = _X1303))
        @test length(coef(fit3, :cutpoints)) == 2
        @test length(fit3.scales[:ordinal_cuts]) == 2
        @test all(isfinite, coef(fit3))

        # K = 2 with both codes observed stays a fit. drmTMB asks for three
        # categories; this family does not, and the guard must not add that rule.
        y2 = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0]
        fit2 = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = y2, x = _X1303))
        @test length(coef(fit2, :cutpoints)) == 1
        @test all(isfinite, coef(fit2))
    end
end
