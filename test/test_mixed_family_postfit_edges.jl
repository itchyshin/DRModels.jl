using DRModels
using Test, Random, Statistics, LinearAlgebra

# Edge-case coverage for src/mixed_family_postfit.jl (mf_coef, mf_aic, mf_bic,
# mf_fitted, mf_summary, _mf_print_ci). Complements test_mixed_family_postfit.jl.

# Knuth Poisson sampler (no Distributions dependency in the test).
function _rpois_pf(rng, λ)
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1
        p *= rand(rng)
        p <= L && return k - 1
    end
end

@testset "cross-family post-fit accessors: edge cases" begin
    rng = MersenneTwister(20260610)
    n = 300
    x = randn(rng, n)
    X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
    β1 = [0.5, 0.8]; β2 = [0.3, -0.5]
    λ1 = 0.7; λ2 = 0.6; σ1 = 0.5
    u = randn(rng, n)
    y1 = X1 * β1 .+ λ1 .* u .+ σ1 .* randn(rng, n)
    η2 = X2 * β2 .+ λ2 .* u
    y2 = Float64[_rpois_pf(rng, exp(clamp(η2[i], -20.0, 20.0))) for i in 1:n]

    fit = DRModels.fit_mixed_family(y1 = y1, X1 = X1, fam1 = Gaussian(),
                               y2 = y2, X2 = X2, fam2 = DRModels.Poisson(),
                               confint = false)
    @test fit.converged
    k = DRModels._mf_nparams(fit)

    warnmsg = "cross-family fit is non-converged or on the objective sentinel; information criterion is NaN"
    bad_conv = merge(fit, (; converged = false))
    bad_sent = merge(fit, (; loglik = -1e18))       # the objective sentinel (nll = 1e18)
    bad_nan  = merge(fit, (; loglik = NaN))

    @testset "mf_aic / mf_bic NaN guard" begin
        for bad in (bad_conv, bad_sent, bad_nan)
            @test (@test_logs (:warn, warnmsg) mf_aic(bad)) |> isnan
            @test (@test_logs (:warn, warnmsg) mf_bic(bad; nobs = n)) |> isnan
        end
        # a healthy fit is silent
        @test_logs mf_aic(fit)
        @test_logs mf_bic(fit; nobs = n)
        # the sentinel boundary: loglik exactly -1e9 is rejected, just above is accepted
        @test isnan(@test_logs (:warn, warnmsg) mf_aic(merge(fit, (; loglik = -1e9))))
        @test isfinite(mf_aic(merge(fit, (; loglik = -1e9 + 1))))
    end

    @testset "mf_bic nobs validation" begin
        @test_throws ArgumentError mf_bic(fit; nobs = 0)
        @test_throws ArgumentError mf_bic(fit; nobs = -3)
        err = try mf_bic(fit; nobs = -3); nothing catch e; e end
        @test err isa ArgumentError && occursin("nobs must be positive", err.msg)
        # nobs is validated before the convergence guard (no warning, throws)
        @test_throws ArgumentError mf_bic(bad_conv; nobs = 0)
        # nobs = 1 gives log(1) = 0 penalty
        @test mf_bic(fit; nobs = 1) ≈ -2 * fit.loglik
    end

    @testset "mf_fitted without design args" begin
        @test_throws ArgumentError mf_fitted(fit)
        err = try mf_fitted(fit); nothing catch e; e end
        @test err isa ArgumentError && occursin("does not carry its design matrices", err.msg)
        # only one of the two present is still an error
        @test_throws ArgumentError mf_fitted(merge(fit, (; X1 = X1)))
        @test_throws ArgumentError mf_fitted(merge(fit, (; X2 = X2)))
        # both present: works and equals the explicit-design method
        fitX = merge(fit, (; X1 = X1, X2 = X2))
        a = mf_fitted(fitX)
        b = mf_fitted(fit, X1, X2)
        @test a.mu1 == b.mu1 && a.mu2 == b.mu2
        @test a.mu1 ≈ X1 * fit.β1
        @test a.mu2 ≈ exp.(X2 * fit.β2)
        # the stored-design method still checks widths
        @test_throws DimensionMismatch mf_fitted(merge(fit, (; X1 = X1[:, 1:1], X2 = X2)))
    end

    @testset "mf_summary" begin
        tbl = mf_coef(fit)
        buf = IOBuffer()
        @test mf_summary(fit; nobs = n, io = buf) === fit
        s = String(take!(buf))
        for t in tbl.term
            @test occursin(t, s)
        end
        @test occursin("converged: true", s)
        @test occursin("logLik = ", s) && occursin("AIC = ", s) && occursin("BIC = ", s)
        @test occursin("AIC = " * string(round(mf_aic(fit); digits = 4)), s)
        @test occursin("BIC = " * string(round(mf_bic(fit; nobs = n); digits = 4)), s)

        buf = IOBuffer()
        mf_summary(fit; io = buf)
        s0 = String(take!(buf))
        @test occursin("AIC = ", s0) && !occursin("BIC", s0)

        # missing CIs (confint = false: NaN tuples): no CI rows
        @test !occursin("95% CI", s)

        # present CIs: only the finite ones print
        fitci = merge(fit, (; rho_ci_wald = (0.1, 0.5), rho_ci_profile = (0.12346, 0.56789),
                              rho_ci_boot = (NaN, 0.9)))
        buf = IOBuffer()
        mf_summary(fitci; nobs = n, io = buf)
        sc = String(take!(buf))
        @test occursin("Wald     95% CI: (0.1, 0.5)", sc)
        @test occursin("profile  95% CI: (0.1235, 0.5679)", sc)
        @test !occursin("bootstrap", sc)            # half-NaN interval is skipped

        # non-converged fit: summary still prints, AIC shown as NaN (with the warning)
        buf = IOBuffer()
        @test_logs (:warn, warnmsg) mf_summary(bad_conv; io = buf)
        @test occursin("AIC = NaN", String(take!(buf)))
    end

    @testset "_mf_print_ci" begin
        buf = IOBuffer()
        DRModels._mf_print_ci(buf, "lbl", (0.25, 0.75))
        @test String(take!(buf)) == "lbl: (0.25, 0.75)\n"
        for ci in ((NaN, NaN), (NaN, 1.0), (0.0, Inf), (-Inf, 1.0))
            @test DRModels._mf_print_ci(buf, "lbl", ci) === nothing
            @test isempty(String(take!(buf)))
        end
    end

    @testset "mf_coef names and order" begin
        tbl = mf_coef(fit)
        expect_terms = vcat(["b1[$j]" for j in eachindex(fit.β1)],
                            ["b2[$j]" for j in eachindex(fit.β2)],
                            ["bsig1[$j]" for j in eachindex(fit.βσ1)],
                            ["bsig2[$j]" for j in eachindex(fit.βσ2)],
                            ["lambda1", "lambda2", "rho"])
        @test tbl.term == expect_terms
        @test tbl.estimate == vcat(fit.β1, fit.β2, fit.βσ1, fit.βσ2,
                                   [fit.λ1, fit.λ2, fit.rho_latent])
        @test tbl.axis == vcat(fill(:y1, length(fit.β1)), fill(:y2, length(fit.β2)),
                               fill(:y1, length(fit.βσ1)), fill(:y2, length(fit.βσ2)),
                               [:shared, :shared, :shared])
        @test eltype(tbl.estimate) == Float64
        # both axes carry dispersion: bsig2 rows appear too, in order after bsig1
        fit2 = merge(fit, (; βσ1 = [0.1, 0.2], βσ2 = [0.3]))
        t2 = mf_coef(fit2)
        @test t2.term == ["b1[1]", "b1[2]", "b2[1]", "b2[2]", "bsig1[1]", "bsig1[2]",
                          "bsig2[1]", "lambda1", "lambda2", "rho"]
        @test t2.estimate[5:7] == [0.1, 0.2, 0.3]
        @test t2.axis[5:7] == [:y1, :y1, :y2]
        @test length(t2.term) == DRModels._mf_nparams(fit2) + 1
    end
end
