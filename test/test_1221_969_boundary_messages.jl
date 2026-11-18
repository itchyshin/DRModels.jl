# #1221: CumulativeLogit (quasi-)complete separation message (warn only).
# #969: Student ν Gaussian-limit boundary flag (warn + `show` note; SE unchanged).
using DRModels
using Test, Random, Logging

@testset "#1221 CumulativeLogit separation message" begin
    n = 40
    x = collect(range(-2, 2, length = n))
    yo = Float64.(1 .+ (x .> 0) .+ (x .> 1))          # 3 levels, perfectly separated by x
    f = @test_logs (:warn, r"\(quasi-\)complete separation") match_mode = :any begin
        drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = yo, x))
    end
    # the message names the coefficient (structured key/value on the log record)
    logs = Test.collect_test_logs() do
        drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = yo, x))
    end |> first
    rec = only(filter(r -> occursin("separation", r.message), logs))
    @test Dict(rec.kwargs)[:flagged_coefficients] == ["x"]
    # warn only: converged / estimates are NOT changed by this message
    @test f isa DRModels.DrmFit

    # a well-identified ordinal fit does not warn
    Random.seed!(11)
    xs = randn(300); ys = Float64.(clamp.(round.(2 .+ 0.8 .* xs .+ randn(300)), 1, 4))
    logs2 = Test.collect_test_logs() do
        drm(bf(@formula(y ~ x)), CumulativeLogit(); data = (; y = ys, x = xs))
    end |> first
    @test !any(r -> occursin("separation", r.message), logs2)
end

@testset "#969 Student nu at the Gaussian limit is flagged" begin
    # seed 1: Gaussian data, Student with nu ~ 1 runs to the nu -> Inf limit (eta ≈ 46)
    Random.seed!(1); xg = randn(300); yg = 1 .+ 0.5 .* xg .+ exp(-0.4) .* randn(300)
    fm = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1), @formula(nu ~ 1))
    d = (; y = yg, x = xg)
    fit = @test_logs (:warn, r"Gaussian limit") match_mode = :any drm(fm, Student(); data = d)
    @test DRModels._student_nu_boundary(fit) !== nothing
    txt = sprint(show, MIME"text/plain"(), fit)
    @test occursin("boundary (limit)", txt)

    # finite-nu fit (seed 2, eta ≈ 3.25): no warning, no note
    Random.seed!(2); x2 = randn(300); y2 = 1 .+ 0.5 .* x2 .+ exp(-0.4) .* randn(300)
    logs = Test.collect_test_logs() do
        drm(fm, Student(); data = (; y = y2, x = x2))
    end |> first
    @test !any(r -> occursin("Gaussian limit", r.message), logs)
    f2 = with_logger(NullLogger()) do
        drm(fm, Student(); data = (; y = y2, x = x2))
    end
    @test DRModels._student_nu_boundary(f2) === nothing
    @test !occursin("boundary (limit)", sprint(show, MIME"text/plain"(), f2))
end
