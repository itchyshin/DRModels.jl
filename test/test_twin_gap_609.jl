# Issue #609 item 1: "prediction parity fails for factor predictors".
#
# Root cause: `_design` (gaussian_core.jl) built the StatsModels contrasts
# schema from whatever data it was CALLED with — training data at fit time,
# but `newdata` at predict time. As long as `newdata` happened to contain
# every training level, in any row order, this was invisible (which is why
# the repo's own `tools/parity_prediction.R` "factors" fixture, whose grid
# is constructed with `levels = levels(g)` covering every level, never
# tripped it). It breaks the moment `newdata`:
#   (b) omits a training level          → wrong number of dummy columns
#                                          (DimensionMismatch against the
#                                          fitted coefficient vector), or
#   (d) contains a level never fitted   → silently different contrasts, or
#                                          an opaque StatsModels error.
#
# Fix: `DrmFormula` now carries a `schema_cache` populated at fit time (the
# first `_design` call, on training data); `predict`/`predict_parameters`
# pass the same cache key so they reuse the TRAINING schema instead of
# rebuilding it from `newdata`. An unseen level then raises a clear
# `ArgumentError` naming the parameter and the problem.
#
# This fixture cross-checks DRModels.jl's own `predict()` against R's
# `model.matrix()` (the same contrast machinery drmTMB relies on) applied to
# the SAME fitted coefficients — isolating the newdata design construction
# from any cross-engine optimizer drift (the established "adapter oracle"
# pattern already used by `tools/parity_prediction.R`).
using DRModels
using Test, DelimitedFiles

@testset "issue #609 item 1: predict with factor levels in newdata" begin
    dir = joinpath(@__DIR__, "fixtures", "twin_gap_609")

    raw, header = readdlm(joinpath(dir, "train_609.csv"), ','; header = true)
    yi = findfirst(==("y"), vec(header)); xi = findfirst(==("x"), vec(header))
    gi = findfirst(==("g"), vec(header))
    y = Float64.(raw[:, yi]); x = Float64.(raw[:, xi]); g = String.(raw[:, gi])
    data = (; y, x, g)

    fit = drm(bf(@formula(y ~ x + g), @formula(sigma ~ x + g)), Gaussian(); data)
    @test fit.converged

    read_grid(name) = begin
        raw, header = readdlm(joinpath(dir, name), ','; header = true)
        xi = findfirst(==("x"), vec(header)); gi = findfirst(==("g"), vec(header))
        (; x = Float64.(raw[:, xi]), g = String.(raw[:, gi]))
    end
    grid_a = read_grid("grid_a_609.csv")   # (a) every training level, training order
    grid_b = read_grid("grid_b_609.csv")   # (b) a SUBSET of training levels
    grid_c = read_grid("grid_c_609.csv")   # (c) every level, different row order
    grid_d = read_grid("grid_d_609.csv")   # (d) a level never seen at fit time ("z")

    expected = Dict{String,Vector{Float64}}()
    for line in eachline(joinpath(dir, "expected_609.txt"))
        k, v = split(line, "="; limit = 2)
        (startswith(k, "pred_") || startswith(k, "oracle_")) || continue
        expected[k] = parse.(Float64, split(v, ";"))
    end

    tol = 1e-8

    # (a) all training levels present, training order: must match the R
    # model.matrix() oracle (same coefficients, independent contrast build).
    pred_a = predict(fit, grid_a; type = :link)
    @test pred_a ≈ expected["pred_a_link"] atol = tol
    @test pred_a ≈ expected["oracle_r_pred_a_link"] atol = tol

    # (b) a subset of training levels: this is exactly where the pre-fix
    # `_design` built a newdata-only schema with too few dummy columns and
    # crashed with a `DimensionMismatch` against the fitted β. Every row of
    # `grid_b` also appears in `grid_a`; predictions at matching (x, g) must
    # agree with (a) and with the R oracle.
    pred_b = predict(fit, grid_b; type = :link)
    @test pred_b ≈ expected["pred_b_link"] atol = tol
    @test pred_b ≈ expected["oracle_r_pred_b_link"] atol = tol
    @test pred_b[2] ≈ pred_a[3] atol = tol   # (x=0.8, g="c") in both grids

    # (c) every level present but rows in a different order: predictions must
    # be a genuine row permutation of (a), not accidentally-correct-by-luck.
    pred_c = predict(fit, grid_c; type = :link)
    @test pred_c ≈ expected["pred_c_link"] atol = tol
    @test pred_c ≈ expected["oracle_r_pred_c_link"] atol = tol
    @test pred_c[1] ≈ pred_a[3] atol = tol   # (x=0.8, g="c")
    @test pred_c[2] ≈ pred_a[1] atol = tol   # (x=-0.7, g="a")
    @test pred_c[3] ≈ pred_a[2] atol = tol   # (x=0.0, g="b")

    # (d) an unseen level ("z"): must refuse with a clear, on-topic error
    # rather than silently mis-predicting or raising an opaque StatsModels
    # error (the pre-fix behaviour was a `DimensionMismatch`/obscure
    # `ArgumentError` unrelated to "unseen level").
    err = nothing
    try
        predict(fit, grid_d; type = :link)
    catch e
        err = e
    end
    @test err isa ArgumentError
    @test occursin("not seen when the model was fitted", sprint(showerror, err))

    # `predict_parameters` (the multi-parameter accessor) must show the same
    # fix for both `:mu` and `:sigma`.
    pp_a = predict_parameters(fit, grid_a; type = :link)
    @test pp_a[:mu] ≈ pred_a atol = tol
    pp_b = predict_parameters(fit, grid_b; type = :link)
    @test pp_b[:mu] ≈ pred_b atol = tol
    err2 = nothing
    try
        predict_parameters(fit, grid_d; type = :link)
    catch e
        err2 = e
    end
    @test err2 isa ArgumentError
    @test occursin("not seen when the model was fitted", sprint(showerror, err2))

    # Numeric-predictor behaviour is unchanged: no factor column at all.
    fit_num = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data)
    β = coef(fit_num, :mu)
    @test predict(fit_num, (; x = [0.0, 1.0])) ≈ β[1] .+ β[2] .* [0.0, 1.0]
end
