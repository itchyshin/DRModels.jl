# test_repeatability_logsigma_re.jl — a random intercept on log σ (`sigma ~ (1 | g)`)
# is NOT a variance component of the response. Its SD is reported under
# `<group>_logsigma` and lives on the log-σ scale, so `repeatability` / `icc` /
# `heritability` must never put it in a numerator or add exp(2·log ω) to a
# denominator. Twin of drmTMB's rule (R/heritability.R): the ratio uses the
# MEAN components only, and a sigma random intercept turns the residual entry
# into the marginal residual variance E[σ²] = exp(2 b₀ + 2 Σ ω_k²).
#
# Fixture: test/fixtures/musigma_ranef_745 (see test_twin_gap_745.jl for its
# provenance). `native_repeatability.tsv` holds drmTMB's own repeatability() on
# that fixture (generated outputs only): drmTMB 0eb0467851 (main, 0.7.1), R 4.6.1,
# `repeatability(drmTMB(bf(y ~ x + (1 | g), sigma ~ (1 | g)), stats::gaussian(), d))`.
using DRModels
using Test

const _RLS_DIR = joinpath(@__DIR__, "fixtures", "musigma_ranef_745")

function _rls_readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    col(n) = cols[findfirst(==(n), hdr)]
    return (y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = col("g"))
end

function _rls_native(file)
    d = Dict{String,Float64}()
    for l in readlines(joinpath(_RLS_DIR, file))[2:end]
        term, est = split(l, '\t')[1:2]
        d[term] = parse(Float64, est)
    end
    return d
end

@testset "repeatability: a log-σ random intercept is not a variance component" begin
    d = _rls_readfix(joinpath(_RLS_DIR, "data.csv"))

    @testset "mean + sigma random intercepts: marginal residual variance" begin
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ (1 | g))), Gaussian(); data = d)
        @test fit.converged
        sb = re_sd(fit)[:g]; ω = re_sd(fit)[:g_logsigma]; b0 = coef(fit, :sigma)[1]
        R_hand = sb^2 / (sb^2 + exp(2b0 + 2ω^2))

        r = repeatability(fit)                  # one MEAN component ⇒ no `component`
        @test r.method === :delta
        @test r.estimate ≈ R_hand atol = 1e-10
        @test isfinite(r.se) && r.se > 0
        @test 0 <= r.ci.lower <= r.estimate <= r.ci.upper <= 1
        @test icc(fit).estimate ≈ R_hand atol = 1e-10
        @test heritability(fit).estimate ≈ R_hand atol = 1e-10   # sole mean component

        # The log-σ SD is not a component that can be chosen.
        @test_throws ErrorException repeatability(fit; component = :g_logsigma)

        # Twin: drmTMB's repeatability() on the same fixture (marginal E[σ²] residual).
        nat = _rls_native("native_repeatability.tsv")
        @test abs(r.estimate - nat["estimate"]) <= 1e-6
        @test abs(r.se - nat["se"]) <= 1e-6

        # The profile CI uses the same marginal residual term.
        p = repeatability(fit; method = :profile)
        @test p.estimate ≈ R_hand atol = 1e-10
        @test p.ci.lower < p.estimate < p.ci.upper
    end

    @testset "variance-boundary check ignores the log-σ modes" begin
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ (1 | g))), Gaussian(); data = d)
        vb = DRModels._variance_boundary(fit)
        @test vb !== nothing
        @test collect(keys(vb.structured_ratios)) == [:g]          # no :g_logsigma
        sb_rms = sqrt(sum(abs2, ranef(fit)[:g]) / length(ranef(fit)[:g]))
        σe = exp(coef(fit, :sigma)[1] + re_sd(fit)[:g_logsigma]^2)  # sqrt(E[σ²])
        @test vb.residual_ratio ≈ σe / sb_rms rtol = 1e-10
    end

    @testset "a mean grouping column named *_logsigma stays a mean component" begin
        d2 = (; y = d.y, x = d.x, g_logsigma = d.g)
        fit = drm(bf(@formula(y ~ x + (1 | g_logsigma)), @formula(sigma ~ 1)), Gaussian(); data = d2)
        sb = re_sd(fit)[:g_logsigma]; se = exp(coef(fit, :sigma)[1])
        @test repeatability(fit).estimate ≈ sb^2 / (sb^2 + se^2) atol = 1e-10
    end

    @testset "sigma-only random intercept: refused, not misread as a variance" begin
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ (1 | g))), Gaussian(); data = d)
        for f in (repeatability, icc, heritability)
            err = try f(fit); nothing catch e; e end
            @test err isa ErrorException
            @test occursin("log σ", sprint(showerror, err))
        end
    end
end
