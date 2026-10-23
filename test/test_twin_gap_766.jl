# #766: `fe_biv_student` returns no confidence-interval endpoint from either
# `confint(fit; method = :profile)` or the parametric bootstrap — an abort that
# fires in ~0 s, even with a single explicit `parm`. Root cause: the generic
# `_simulate_once` in gaussian_core.jl special-cases bivariate GAUSSIAN fits
# (`fam isa Gaussian && haskey(fit.scales, :sigma1)`) before falling through to
# `fit.means[:mu]` / `fit.scales[:sigma]` for every other family. A bivariate
# Student-t fit (`biv_student()`) stores `:mu1`/`:mu2` and `:sigma1`/`:sigma2`
# instead — never `:mu`/`:sigma` — so that fallthrough throws
# `KeyError: key :mu not found`. The parametric bootstrap draws its replicate
# response via `simulate(fit0; rng)` (gaussian_core.jl) BEFORE any refit is
# attempted, so every replicate fails identically and immediately: exactly the
# "0 s abort, no fit work attempted" symptom, independent of `parm`.
#
# Fix: `src/bivariate_student.jl` adds `_simulate_once(fit::DrmFit{Student}, rng;
# ...)`, more specific than the generic `fit::DrmFit` method, dispatching to a
# bivariate branch (mirroring bivariate Gaussian's own `Y = mu + diag(sigma) *
# Z * sqrt(nu / chisq_nu)` draw, per this file's own docstring) when
# `fit.scales` has `:sigma1`, and to the untouched univariate Student draw
# otherwise.
#
# Profile intervals never call `simulate` at all, and were already fixed on
# this branch (claude/twin-gap-bivstudent) by the large-ν bivariate-t density
# correction — confirmed passing here as a regression guard, not a new fix.

using DRModels
using Test
using Random

@testset "#766: fe_biv_student profile and bootstrap return finite CI endpoints" begin

    rng = MersenneTwister(6401)
    n = 400
    x = randn(rng, n)
    s1, s2, rho, nu = 0.55, 0.85, 0.35, 7.0
    z1 = randn(rng, n)
    z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(rng, n)
    chi = [sum(randn(rng, Int(nu)) .^ 2) for _ in 1:n]
    sh = sqrt.(nu ./ chi)
    y1 = 0.2 .+ 0.45 .* x .+ s1 .* z1 .* sh
    y2 = -0.3 .+ (-0.25) .* x .+ s2 .* z2 .* sh
    data = (; y1 = y1, y2 = y2, x = x)
    f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1))

    fit = drm(f, Student(); data = data)
    @test is_converged(fit)

    @testset "profile: an explicit single parm returns a finite endpoint" begin
        ci = confint(fit; method = :profile, parm = :mu1)
        @test length(ci) == 2
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in ci)
    end

    @testset "profile: every coefficient returns a finite endpoint" begin
        ci = confint(fit; method = :profile)
        @test length(ci) == 8
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in ci)
    end

    @testset "simulate(fit) no longer throws KeyError (the #766 gate)" begin
        ysim = simulate(fit; rng = MersenneTwister(1))
        @test ysim isa AbstractDict
        @test haskey(ysim, :mu1) && haskey(ysim, :mu2)
        @test length(ysim[:mu1]) == n
        @test all(isfinite, ysim[:mu1]) && all(isfinite, ysim[:mu2])
    end

    @testset "bootstrap_result: replicates run and return finite endpoints" begin
        res = bootstrap_result(fit; data = data, B = 8, rng = MersenneTwister(2))
        @test res.failed == 0
        @test res.used == 8
        @test length(res.summary) == 8
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in res.summary)
    end

    @testset "bootstrap_ci: same, via the CI-only surface" begin
        ci = bootstrap_ci(fit; data = data, B = 8, rng = MersenneTwister(3))
        @test length(ci) == 8
        @test all(isfinite(r.lower) && isfinite(r.upper) for r in ci)
    end

end
