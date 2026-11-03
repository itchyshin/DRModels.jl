# Missing-data handling (issue #49).
#
# Two jobs:
#   (A) DOCUMENT current behaviour — a raw `missing`/`NaN` response or predictor
#       makes `drm` ERROR (it does not silently produce a garbage fit). This pins
#       the status quo so a future FIML/imputation path is a deliberate change.
#   (B) ANCHOR the new `drm_listwise` complete-case preprocessing: it drops the
#       right rows, warns, the cleaned data fits, and a listwise fit on
#       MAR-deleted data recovers parameters ≈ the complete-data fit.
#
# Out of scope (follow-up under #49): FIML for missing responses; multiple
# imputation for missing predictors. See report/fiml-missing-data-design.md.
using DRModels
using Test, Random

@testset "Missing data — listwise deletion path (#49)" begin

    @testset "(A) response missingness → warn + observed-rows fit; predictor missingness → error" begin
        # Reconciliation of #241 (auto-fit on observed rows) with #258 ("not SILENTLY
        # fit"): a missing/NaN RESPONSE is now dropped (observed-rows fit) WITH A
        # WARNING — glmmTMB's default na.action behaviour, made non-silent. A missing
        # PREDICTOR still errors (out of scope, both PRs agree).
        Random.seed!(20260610)
        n = 200; x = randn(n)
        ybase = 1.0 .+ 0.5 .* x .+ 0.5 .* randn(n)

        # `missing` AND `NaN` in the response → warn + fit on the observed rows.
        # Captured via a logger (robust; avoids the @test_logs failure-recording quirk).
        for bad in (missing, NaN)
            yb = bad === missing ? Vector{Union{Missing,Float64}}(ybase) : copy(ybase)
            yb[5] = bad
            io = IOBuffer()
            fit = Base.CoreLogging.with_logger(Base.CoreLogging.SimpleLogger(io, Base.CoreLogging.Warn)) do
                drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y = yb, x))
            end
            @test occursin("dropped", String(take!(io)))   # not SILENT — it warns
            @test isfinite(loglik(fit))                     # not garbage — the bad row was dropped
        end

        # `NaN` in a predictor → still errors (predictor missingness is out of scope).
        xnan = copy(x); xnan[5] = NaN
        @test_throws Exception drm(bf(@formula(y ~ x), @formula(sigma ~ 1)),
                                   Gaussian(); data = (; y = ybase, x = xnan))
    end

    @testset "(B1) drm_listwise drops the right rows + warns" begin
        Random.seed!(1)
        n = 50; x = randn(n)
        y = Vector{Union{Missing,Float64}}(1.0 .+ 0.5 .* x .+ 0.3 .* randn(n))
        y[3] = missing; y[7] = missing      # 2 missing responses
        xx = copy(x); xx[7] = NaN; xx[20] = NaN   # 7 overlaps a missing y; 20 is new ⇒ 3 distinct rows

        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        clean = @test_logs (:warn,) drm_listwise(f, (; y, x = xx))
        @test length(clean.y) == n - 3            # rows 3, 7, 20 removed
        @test length(clean.x) == n - 3
        @test !any(ismissing, clean.y)
        @test all(isfinite, clean.x)
        @test eltype(clean.y) <: Float64          # Missing union narrowed away after the drop

        # The cleaned table fits through the normal API without error.
        fit = drm(f, Gaussian(); data = clean)
        @test isfinite(loglik(fit))
        @test nobs(fit) == n - 3

        # verbose = false suppresses the warning but still drops.
        clean2 = @test_logs drm_listwise(f, (; y, x = xx); verbose = false)
        @test length(clean2.y) == n - 3
    end

    @testset "(B2) no missing ⇒ no warning, columns unchanged" begin
        Random.seed!(2)
        n = 30; x = randn(n)
        y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        clean = @test_logs drm_listwise(f, (; y, x))   # asserts NO log messages
        @test clean.y == y
        @test clean.x == x
    end

    @testset "(B3) unrelated missing column does not drop rows" begin
        Random.seed!(3)
        n = 40; x = randn(n)
        y = 1.0 .+ 0.5 .* x .+ 0.3 .* randn(n)
        junk = Vector{Union{Missing,Float64}}(randn(n)); junk[1:10] .= missing  # not in the model
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        clean = @test_logs drm_listwise(f, (; y, x, junk))   # junk ignored ⇒ no warning, no drop
        @test length(clean.y) == n
    end

    @testset "(B4) guard: formula column absent from data errors clearly" begin
        n = 10
        f = bf(@formula(y ~ z), @formula(sigma ~ 1))   # references `z`
        @test_throws ArgumentError drm_listwise(f, (; y = randn(n), x = randn(n)))  # no `z`
    end

    @testset "(B5) bivariate: complete-case drops a row if EITHER trait is missing" begin
        Random.seed!(4)
        n = 60; x = randn(n)
        y1 = Vector{Union{Missing,Float64}}(0.5 .+ 0.4 .* x .+ 0.5 .* randn(n))
        y2 = Vector{Union{Missing,Float64}}(-0.2 .+ 0.3 .* x .+ 0.7 .* randn(n))
        y1[4] = missing       # only trait 1 missing
        y2[9] = missing       # only trait 2 missing
        y1[15] = missing; y2[15] = missing   # both missing (one row)
        bivf = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
                  sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                  rho12 = @formula(rho12 ~ 1))
        clean = @test_logs (:warn,) drm_listwise(bivf, (; y1, y2, x))
        @test length(clean.y1) == n - 3       # rows 4, 9, 15 all dropped (complete-case)
        fit = drm(bivf, Gaussian(); data = clean)
        @test isfinite(loglik(fit))
    end

    @testset "(B6) recovery: listwise fit ≈ complete-data fit under MAR deletion" begin
        Random.seed!(20260610)
        n = 4000
        x = randn(n)
        βμ = [0.5, -0.8]
        βσ = [-0.3, 0.4]
        μ = βμ[1] .+ βμ[2] .* x
        logσ = βσ[1] .+ βσ[2] .* x
        ytrue = μ .+ exp.(logσ) .* randn(n)

        f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x))

        # Complete-data baseline.
        fit_full = drm(f, Gaussian(); data = (; y = ytrue, x))

        # MAR deletion: probability of a missing response depends on the (observed)
        # predictor x, NOT on y itself ⇒ listwise stays consistent here.
        pmiss = 1.0 ./ (1.0 .+ exp.(-(−1.0 .+ 0.8 .* x)))   # ~20–40% missing, x-driven
        y = Vector{Union{Missing,Float64}}(ytrue)
        for i in 1:n
            rand() < pmiss[i] && (y[i] = missing)
        end
        ndrop = count(ismissing, y)
        @test 0 < ndrop < n                                  # genuinely partial

        clean = drm_listwise(f, (; y, x); verbose = false)
        fit_lw = drm(f, Gaussian(); data = clean)

        @test nobs(fit_lw) == n - ndrop
        # Recovers the truth (looser tol than complete-data: fewer effective rows).
        @test coef(fit_lw, :mu)    ≈ βμ atol = 0.10
        @test coef(fit_lw, :sigma) ≈ βσ atol = 0.10
        # And lands close to the complete-data estimates.
        @test coef(fit_lw, :mu)    ≈ coef(fit_full, :mu)    atol = 0.10
        @test coef(fit_lw, :sigma) ≈ coef(fit_full, :sigma) atol = 0.10
    end
end

# #1111: a numeric predictor typed `Vector{Union{Missing,Float64}}` that holds no
# `missing` must be fitted as continuous, not dummy-coded as a factor (one
# coefficient per distinct value). `drm_listwise` on complete data must hand back
# a narrowed column too. A `Vector{Any}` column (the bridge's R `factor(x)`)
# stays categorical.
@testset "#1111: Union{Missing,Float64} predictor with no missing is continuous" begin
    n = 120; G = 12
    g = repeat(1:G, inner = n ÷ G)
    r = Random.MersenneTwister(2111)
    x = randn(r, n)
    U(v) = Vector{Union{Missing,eltype(v)}}(v)
    η = 0.2 .+ 0.3 .* x .+ 0.3 .* randn(r, G)[g]

    @testset "$(nameof(typeof(fam))) (1 | g)" for fam in (Gaussian(), Poisson(), Gamma())
        y = fam isa Gaussian ? η .+ 0.5 .* randn(r, n) :
            fam isa Poisson  ? Float64.([rand(r, 0:max(0, round(Int, 2exp(e)))) for e in η]) :
                               exp.(η) .* (0.5 .+ rand(r, n))
        mk() = bf(@formula(y ~ x + (1 | g)))
        ref = drm(mk(), fam; data = (; y, x, g))
        fit = drm(mk(), fam; data = (; y, x = U(x), g))
        @test length(coef(fit, :mu)) == 2
        @test loglik(fit) ≈ loglik(ref)
        @test coef(fit, :mu) ≈ coef(ref, :mu)
    end

    @testset "sigma ~ x: one sigma slope, not one per value" begin
        y = 0.5 .+ 0.4 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(r, n)
        mk() = bf(@formula(y ~ x), @formula(sigma ~ x))
        ref = drm(mk(), Gaussian(); data = (; y, x))
        fit = drm(mk(), Gaussian(); data = (; y, x = U(x)))
        @test length(coef(fit, :sigma)) == 2
        @test loglik(fit) ≈ loglik(ref)
    end

    @testset "few distinct values (the plausible-looking failure)" begin
        xi = Float64.(round.(x))
        y = 0.5 .+ 0.4 .* xi .+ 0.5 .* randn(r, n)
        mk() = bf(@formula(y ~ xi), @formula(sigma ~ 1))
        ref = drm(mk(), Gaussian(); data = (; y, xi))
        fit = drm(mk(), Gaussian(); data = (; y, xi = U(xi)))
        @test length(coef(fit, :mu)) == 2
        @test loglik(fit) ≈ loglik(ref)
    end

    @testset "drm_listwise on complete data narrows the column" begin
        y = 0.5 .+ 0.4 .* x .+ 0.5 .* randn(r, n)
        f = bf(@formula(y ~ x), @formula(sigma ~ 1))
        clean = @test_logs drm_listwise(f, (; y = U(y), x = U(x)))
        @test eltype(clean.x) === Float64
        @test eltype(clean.y) === Float64
        @test clean.x == x
        # rows are dropped here; every kept column is narrowed, an unrelated one too
        xm = U(x); xm[3] = missing
        junk = U(x)
        clean2 = drm_listwise(f, (; y, x = xm, junk); verbose = false)
        @test eltype(clean2.junk) === Float64          # kept rows hold no missing
        clean3 = drm_listwise(f, (; y, x, junk))        # nothing dropped: still narrowed
        @test eltype(clean3.junk) === Float64
    end

    @testset "a Vector{Any} factor column stays categorical" begin
        xf = Vector{Any}(Float64.(repeat(1:3, inner = n ÷ 3)))
        y = 0.5 .+ 0.4 .* Float64.(xf) .+ 0.5 .* randn(r, n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data = (; y, x = xf))
        @test length(coef(fit, :mu)) == 3              # intercept + 2 dummies
    end
end
