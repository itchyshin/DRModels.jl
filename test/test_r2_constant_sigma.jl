# test_r2_constant_sigma.jl — R² where it means one thing, and refusals elsewhere.
#
# The value test is a CROSS-CHECK against OLS computed from first principles in
# this file, not against a number copied from a previous run: on a constant-σ
# Gaussian fit, 1 - RSS/TSS must equal what `lm()` reports for the same mean model.
#
# The refusal tests are the point of the function. Each one is a RED CONTROL for a
# denominator that would otherwise be picked silently.
using DRModels, Test, Random

@testset "r2_constant_sigma" begin
    @testset "constant σ: equals the OLS R² exactly" begin
        Random.seed!(20260906)
        n = 300
        x = randn(n)
        y = 1.0 .+ 2.0 .* x .+ 1.5 .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        r = r2_constant_sigma(fit)

        X = hcat(ones(n), x)
        b = X \ y
        ols = 1 - sum(abs2, y .- X * b) / sum(abs2, y .- sum(y) / n)

        @test isfinite(r)
        @test 0 < r < 1
        @test r ≈ ols atol = 1e-12          # the same quantity, not merely close
    end

    @testset "REFUSES a modelled σ, naming the offending term" begin
        Random.seed!(20260906)
        n = 400
        x = randn(n)
        sex = rand(["female", "male"], n)
        sd = [s == "male" ? 1.6 : 0.6 for s in sex]
        y = 1.0 .+ 2.0 .* x .+ sd .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ sex)), Gaussian(); data = (; y, x, sex))
        @test_throws ArgumentError r2_constant_sigma(fit)
        msg = try
            r2_constant_sigma(fit)
        catch e
            sprint(showerror, e)
        end
        @test occursin("MODELLED", msg)
        @test occursin("sex: male", msg)     # names WHICH term, not just "some term"
        @test occursin("sigma ~ 1", msg)     # says what to do instead
    end

    @testset "REFUSES random effects (marginal vs conditional is a different question)" begin
        Random.seed!(3)
        G = 30; nper = 10; n = G * nper
        g = repeat(1:G, inner = nper)
        x = randn(n)
        b = 0.7 .* randn(G)
        y = 1.0 .+ 0.5 .* x .+ b[g] .+ 0.4 .* randn(n)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; y, x, g))
        @test_throws ArgumentError r2_constant_sigma(fit)
        msg = try
            r2_constant_sigma(fit)
        catch e
            sprint(showerror, e)
        end
        @test occursin("MARGINAL", msg) && occursin("CONDITIONAL", msg)
    end

    @testset "REFUSES a non-Gaussian family" begin
        Random.seed!(11)
        n = 200; x = randn(n)
        yc = rand(0:5, n)
        fit = drm(bf(@formula(yc ~ x)), DRModels.Poisson(); data = (; yc, x))
        @test_throws ArgumentError r2_constant_sigma(fit)
    end

    @testset "REFUSES a mean model with no intercept" begin
        Random.seed!(5)
        n = 200; x = randn(n)
        y = 2.0 .* x .+ randn(n)
        fit = drm(bf(@formula(y ~ 0 + x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        @test_throws ArgumentError r2_constant_sigma(fit)
    end
end

# --- issue #752: natural-scale residual SD + dof_residual in show() -------------
@testset "residual SD (response scale) and dof_residual in show (#752)" begin
    Random.seed!(20260906)
    n = 400
    x = randn(n)
    sex = rand(["female", "male"], n)
    sd = [s == "male" ? 1.6 : 0.6 for s in sex]
    y = 1.0 .+ 2.0 .* x .+ sd .* randn(n)

    @testset "constant σ prints ONE residual SD" begin
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x))
        out = sprint(show, MIME("text/plain"), fit)
        @test occursin("Residual SD (response scale) = ", out)
        @test occursin("dof_residual = ", out)
        @test !occursin("varies with the σ model", out)
        # the printed number is the fitted scale, not something re-derived
        @test occursin(string(round(first(fit.scales[:sigma]), digits = 4)), out)
    end

    @testset "modelled σ prints a RANGE, never one number" begin
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ sex)), Gaussian(); data = (; y, x, sex))
        out = sprint(show, MIME("text/plain"), fit)
        @test occursin("varies with the σ model", out)
        # RED CONTROL: collapsing a modelled σ to a single "Residual SD = x" would
        # be read as "the" residual SD. That form must NOT appear here.
        @test !occursin("Residual SD (response scale) = ", out)
        lo, hi = extrema(fit.scales[:sigma])
        @test hi > lo
        @test 0.4 < lo < 0.9 && 1.2 < hi < 1.9      # brackets the simulated 0.6 / 1.6
    end

    @testset "GATED to Gaussian: no Residual SD label on other families" begin
        # scales[:sigma] holds a shape for Gamma and a dispersion for NB2, so the
        # label would be wrong there. Absence is the assertion.
        cnt = drm(bf(@formula(yc ~ x)), DRModels.Poisson(); data = (yc = rand(0:5, n), x = x))
        @test !occursin("Residual SD", sprint(show, MIME("text/plain"), cnt))
    end
end
