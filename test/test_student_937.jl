# DRModels.jl#937: a Student fit whose ν runs off to the Gaussian limit is not a
# converged fit. Owner decision (25 Oct 2026): restart from a second start, as the
# crossed-intercept route does, and make `is_converged` reject ν above a cutoff or a
# large gradient.
using DRModels
using Test, Random, LinearAlgebra
using Distributions: TDist

function _read_csv_937(name)
    lines = readlines(joinpath(@__DIR__, "fixtures", "lme4", name))
    hdr = [strip(c, '"') for c in split(lines[1], ',')]
    cols = [String[strip(split(l, ',')[j], '"') for l in lines[2:end]] for j in eachindex(hdr)]
    return Dict(Symbol(h) => c for (h, c) in zip(hdr, cols))
end

@testset "Student ν at the Gaussian limit (#937)" begin
    @testset "sleepstudy (1 | Subject), sigma ~ Days: the second start leaves the Gaussian ridge" begin
        ss = _read_csv_937("sleepstudy.csv")
        sleep = (; Reaction = parse.(Float64, ss[:Reaction]), Days = parse.(Float64, ss[:Days]),
                   Subject = ss[:Subject])
        fit = drm(bf(@formula(Reaction ~ Days + (1 | Subject)), @formula(sigma ~ Days),
                     @formula(nu ~ 1)), Student(); data = sleep)
        # The single start used to stop at nu = 1.3e12, logLik -891.912 (the Gaussian
        # fit); the heavy-tailed optimum is -887.513 at nu = 4.57.
        @test loglik(fit) > -887.6
        @test loglik(fit) ≈ -887.5133 atol = 1e-3
        @test fit.scales[:nu][1] < 100
        @test convergence_status(fit) === :converged
        @test is_converged(fit)
    end

    @testset "is_converged rejects ν above the cutoff" begin
        Random.seed!(937)
        n = 400; x = randn(n)
        gauss = (; y = 1 .+ 0.5 .* x .+ randn(n), x)             # Gaussian data: ν has no finite MLE
        f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))
        # se = false skips the covariance rules, so the Wald-matrix screens do not
        # catch the limit first.
        fit = drm(f, Student(); data = gauss, se = false)
        @test fit.scales[:nu][1] > DRModels._STUDENT_NU_CUTOFF
        @test convergence_status(fit) === :boundary
        @test !is_converged(fit)
    end

    @testset "is_converged rejects a large gradient; a tight fit passes" begin
        Random.seed!(938)
        n = 400; x = randn(n)
        heavy = (; y = 1 .+ 0.5 .* x .+ 0.8 .* rand(TDist(4), n), x)
        f = bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(nu ~ 1))
        tight = drm(f, Student(); data = heavy, se = false)
        @test convergence_status(tight) === :converged
        @test norm(gradient(tight), Inf) / n <= 1e-6
        # A loose optimiser tolerance satisfies Optim's own test (`fit.converged`)
        # with a gradient far above the bar.
        loose = drm(f, Student(); data = heavy, se = false, g_tol = 1.0)
        @test loose.converged
        @test norm(gradient(loose), Inf) / n > 1e-6
        @test convergence_status(loose) === :degenerate
        @test !is_converged(loose)
    end
end
