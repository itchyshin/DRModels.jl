# test_twin_gap_745.jl — TDD for #745 (twin drmTMB #1287): DRModels.jl refused
# `bf(y ~ x + (1 | g), sigma ~ (1 | g))` (simultaneous mean + sigma random
# intercepts) with "a random effect on `sigma` must be the only random
# structure", while drmTMB admits it (its TMB Laplace integrates the full
# stacked (u_mu, u_sigma) vector, independent `dnorm(u,0,1)` priors — see
# `src/drmTMB.cpp` model_type 1).
#
# Route implemented: `_fit_musigma_ranef_gaussian` (src/gaussian_ranef.jl). The
# two random effects here share ONE grouping factor, so drmTMB's joint Hessian
# over the whole (u_mu, u_sigma) vector is block-diagonal in 2×2 blocks — one
# per group. Summing an independent per-group 2-D Laplace approximation over
# those blocks IS the whole-model nested Laplace, so no approximation is lost
# doing it group-by-group; each group's own random intercept on the mean is
# NOT profiled out in closed form first (the sigma random effect makes the
# per-row variance itself latent, so the exact-Woodbury trick
# `_fit_ranef_gaussian` uses for `sigma ~ x` fixed effects does not apply once
# `sigma` also carries `(1 | g)`) — instead both scalars (b_mu,k, delta_sigma,k)
# are found and Laplace-integrated together per group. `marginal` is always
# `:Laplace` on this route: there is no closed-form or quadrature alternative.
#
# Fixture provenance: `test/fixtures/musigma_ranef_745/data.csv` was simulated
# in R (seed 20344905, G = 60 groups, n_g = 10, β_μ = (1.2, -0.6), fixed-effect
# log σ intercept = -0.2, sd_mu = 0.6, sd_sigma = 0.3) and fit with drmTMB
# 0.7.1 (`drmTMB(bf(y ~ x + (1 | g), sigma ~ (1 | g)), family =
# stats::gaussian(), data = d)`, engine = "tmb", i.e. native TMB Laplace); the
# reference numbers are in `native.tsv`.
using DRModels
using Test, Random, LinearAlgebra
import ForwardDiff

const _MSR_DIR = joinpath(@__DIR__, "fixtures", "musigma_ranef_745")

function _msr_readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    col(n) = cols[findfirst(==(n), hdr)]
    return (y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = col("g"))
end

function _msr_native()
    lines = readlines(joinpath(_MSR_DIR, "native.tsv"))
    d = Dict{String,Float64}()
    for l in lines[2:end]
        term, est, _ = split(l, '\t')
        d[term] = parse(Float64, est)
    end
    return d
end

@testset "twin #745: simultaneous mean + sigma random intercepts" begin
    d = _msr_readfix(joinpath(_MSR_DIR, "data.csv"))
    fμ = @formula(y ~ x + (1 | g))
    fσ = @formula(sigma ~ (1 | g))

    @testset "admitted (no longer refuses; #745 base error is gone)" begin
        fit = drm(bf(fμ, fσ), Gaussian(); data = d)
        @test fit isa DrmFit
        @test fit.converged
        @test fit.marginal === :Laplace
    end

    fit = drm(bf(fμ, fσ), Gaussian(); data = d)
    nat = _msr_native()

    @testset "matches native drmTMB (TMB Laplace) on the committed fixture" begin
        # Both sides use Laplace over the identical block-diagonal likelihood,
        # so agreement is far tighter than the ~1e-4 the twin ledger asks for.
        @test abs(loglik(fit) - nat["logLik"]) <= 1e-4
        @test dof(fit) == Int(nat["df"])
        @test nobs(fit) == Int(nat["nobs"])
        cmu = coef(fit, :mu); csig = coef(fit, :sigma)
        @test abs(cmu[1] - nat["mu:(Intercept)"]) <= 1e-4
        @test abs(cmu[2] - nat["mu:x"]) <= 1e-4
        @test abs(csig[1] - nat["sigma:(Intercept)"]) <= 1e-4
        sds = re_sd(fit)
        @test abs(sds[:g] - nat["sd:mu:(1 | g)"]) <= 1e-3
        @test abs(sds[:g_logsigma] - nat["sd:sigma:(1 | g)"]) <= 1e-3
    end

    @testset "known-DGM recovery (truth: β_μ=(1.2,-0.6), log σ₀=-0.2, sd_mu=0.6, sd_sigma=0.3)" begin
        cmu = coef(fit, :mu); csig = coef(fit, :sigma); sds = re_sd(fit)
        se = stderror(fit)
        @test abs(cmu[1] - 1.2) <= 5 * se[1]
        @test abs(cmu[2] - (-0.6)) <= 5 * se[2]
        @test abs(csig[1] - (-0.2)) <= 5 * se[3]
        @test abs(sds[:g] - 0.6) <= 0.25
        @test abs(sds[:g_logsigma] - 0.3) <= 0.15
    end

    @testset "vc() / ranef() surface both axes distinctly" begin
        v = vc(fit)
        @test v[:g] ≈ fill(re_sd(fit)[:g]^2, 1, 1)
        @test v[:g_logsigma] ≈ fill(re_sd(fit)[:g_logsigma]^2, 1, 1)
        rr = ranef(fit)
        @test length(rr[:g]) == 60
        @test length(rr[:g_logsigma]) == 60
    end

    @testset "AD gradient is exact through the per-group Newton refinement" begin
        g0 = ForwardDiff.gradient(fit.nll, fit.theta)
        @test norm(g0, Inf) < 1e-5
        θp = fit.theta .+ [0.05, -0.03, 0.1, -0.05, 0.02]
        gad = ForwardDiff.gradient(fit.nll, θp)
        h = 1e-6
        gfd = [(fit.nll(θp .+ h .* (1:5 .== k)) - fit.nll(θp .- h .* (1:5 .== k))) / 2h for k in 1:5]
        @test gad ≈ gfd rtol = 1e-5
    end

    @testset "refusals: scope of the new route" begin
        dh = merge(d, (h = d.g,))   # a second grouping column, same levels
        @test_throws r"a single mean random INTERCEPT" drm(
            bf(@formula(y ~ x + (1 | g) + (1 | h)), fσ), Gaussian(); data = dh)
        @test_throws r"a single `sigma` random INTERCEPT" drm(
            bf(fμ, @formula(sigma ~ (1 | g) + (1 | h))), Gaussian(); data = dh)
        @test_throws r"SAME grouping factor" drm(
            bf(fμ, @formula(sigma ~ (1 | h))), Gaussian(); data = dh)
        @test_throws r"structured .* mean marker or `meta_V" drm(
            bf(@formula(y ~ x + (1 | g) + meta_V(v)), fσ), Gaussian();
            data = merge(d, (v = fill(1.0, length(d.y)),)))
        @test_throws r"method = :REML is not implemented" drm(bf(fμ, fσ), Gaussian(); data = d,
                                                              method = :REML)
        # A random SLOPE on the mean (not just an intercept) is still refused,
        # not silently routed through this intercept-only Laplace pair.
        @test_throws r"a single mean random INTERCEPT" drm(
            bf(@formula(y ~ (0 + x | g)), fσ), Gaussian(); data = d)
    end
end
