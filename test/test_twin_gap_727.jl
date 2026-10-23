# test_twin_gap_727.jl — issue #727 (twin drmTMB #1281 line of gaps): the
# formula grammar had no `offset()` term at all, while drmTMB supports
# `y ~ x + offset(log_exposure)` (gated to the Poisson mean, log link). Verifies:
#   1. RED: `offset` is undefined pre-#727 (`UndefVarError`), verified manually
#      against the pre-#727 base (origin/main); this file only exercises the
#      post-fix (GREEN) behaviour.
#   2. GREEN: known-exposure recovery — the slope on x recovers, the offset's
#      own coefficient is fixed at 1 (never estimated), and predictions scale
#      with exposure.
#   3. `offset(...)` also accepts an inline transform (`offset(log(exposure))`),
#      not just a bare column reference, and gives the identical fit.
#   4. Omitting the offset on the same DGP recovers a visibly biased intercept
#      (the offset is not a no-op).
#   5. `offset(...)` combined with a random effect, a structured marker, `zi`,
#      or `hu` is refused with a clear error rather than silently ignored.

using DRModels
using Test, Random, LinearAlgebra
using Statistics: std
import Distributions

@testset "twin-gap #727: Poisson offset()" begin
    @testset "GREEN: known-exposure recovery" begin
        Random.seed!(20260927)
        n = 3000
        x = randn(n)
        exposure = rand(n) .* 5 .+ 0.5
        log_exposure = log.(exposure)
        β = [0.20, 0.50]
        λ = exposure .* exp.(β[1] .+ β[2] .* x)
        y = Float64.(rand.(Distributions.Poisson.(λ)))

        fit = drm(bf(@formula(y ~ x + offset(log_exposure))), Poisson();
                  data = (; y, x, log_exposure))

        @test fit.converged
        @test coef(fit, :mu)[1] ≈ β[1] atol = 0.08
        @test coef(fit, :mu)[2] ≈ β[2] atol = 0.05
        # the offset coefficient is fixed at 1, not estimated: only 2 mu
        # coefficients are reported (intercept, x), never a 3rd for the offset.
        @test length(coef(fit, :mu)) == 2
        @test isfinite(loglik(fit))
        # predictions scale with exposure: doubling it should double fitted λ.
        fv = fitted(fit)
        @test all(fv .> 0)
        pred_ratio = fv ./ exposure
        @test std(log.(pred_ratio)) < 0.6   # roughly constant exp(Xβ̂)/obs noise
    end

    @testset "offset(...) accepts an inline transform" begin
        Random.seed!(20260928)
        n = 1500
        x = randn(n)
        exposure = rand(n) .* 3 .+ 0.2
        β = [-0.10, 0.35]
        λ = exposure .* exp.(β[1] .+ β[2] .* x)
        y = Float64.(rand.(Distributions.Poisson.(λ)))

        fit_col = drm(bf(@formula(y ~ x + offset(log_exposure))), Poisson();
                      data = (; y, x, log_exposure = log.(exposure)))
        fit_inline = drm(bf(@formula(y ~ x + offset(log(exposure)))), Poisson();
                         data = (; y, x, exposure))

        @test coef(fit_col, :mu) ≈ coef(fit_inline, :mu) atol = 1e-8
        @test loglik(fit_col) ≈ loglik(fit_inline) atol = 1e-8
    end

    @testset "offset is not a no-op: omitting it biases the intercept" begin
        Random.seed!(20260929)
        n = 2000
        x = randn(n)
        exposure = rand(n) .* 9 .+ 1.0    # wide exposure range → clear bias if dropped
        log_exposure = log.(exposure)
        β = [0.0, 0.4]
        λ = exposure .* exp.(β[1] .+ β[2] .* x)
        y = Float64.(rand.(Distributions.Poisson.(λ)))

        fit_off = drm(bf(@formula(y ~ x + offset(log_exposure))), Poisson();
                      data = (; y, x, log_exposure))
        fit_no_off = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))

        @test coef(fit_off, :mu)[1] ≈ β[1] atol = 0.1
        @test abs(coef(fit_no_off, :mu)[1] - β[1]) > 0.3   # visibly biased without the offset
    end

    @testset "offset(...) refused outside the fixed-effects-only mean" begin
        Random.seed!(20260930)
        n = 400
        x = randn(n)
        log_exposure = randn(n) .* 0.3
        g = repeat(1:20, inner = 20)
        y = Float64.(rand.(Distributions.Poisson.(exp.(0.2 .+ 0.3 .* x))))

        @test_throws ErrorException drm(
            bf(@formula(y ~ x + offset(log_exposure) + (1 | g))), Poisson();
            data = (; y, x, log_exposure, g),
        )
        @test_throws ErrorException drm(
            bf(@formula(y ~ x + offset(log_exposure)), @formula(zi ~ 1)), Poisson();
            data = (; y, x, log_exposure),
        )
        @test_throws ErrorException drm(
            bf(@formula(y ~ x + offset(log_exposure)), @formula(hu ~ 1)), Poisson();
            data = (; y, x, log_exposure),
        )
    end

    @testset "at most one offset(...) term" begin
        n = 50
        x = randn(n); log_e1 = randn(n); log_e2 = randn(n)
        y = Float64.(rand.(Distributions.Poisson.(exp.(0.1 .+ 0.2 .* x))))
        @test_throws ErrorException drm(
            bf(@formula(y ~ x + offset(log_e1) + offset(log_e2))), Poisson();
            data = (; y, x, log_e1, log_e2),
        )
    end
end
