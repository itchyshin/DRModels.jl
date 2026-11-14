# simulate(fit; data) for several ordinary random intercepts on distinct groups.
# Before: `simulate` refused ("no marginal ... simulator is available") for
# `(1 | g) + (1 | g2) [+ (1 | h)]`, although drmTMB 0.8.1 draws one fresh N(0, sd^2) vector per term.
# R reference: mean per-observation variance (and variance of g group means) of 4000 `simulate(fit, nsim = 4000, seed = 20261110)`
# draws from drmTMB 0.8.1 on the same data (Monte Carlo error about 1.5% at 2000 draws).
using DRModels
using Test
using Random
using Statistics
using DelimitedFiles
using StatsModels: @formula

const _SMR_DIR = joinpath(@__DIR__, "fixtures", "gap_v2_crashes")
const _SMR_FACTORS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])
const _SMR = let (m, h) = readdlm(joinpath(_SMR_DIR, "data_v2.csv"), ',', Any; header = true)
    h = String.(vec(h))
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(h)
        c = m[:, j]
        v = if n in _SMR_FACTORS
            any(==("NA"), c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(==("NA"), c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    (; cols...)
end

@testset "simulate: several ordinary random intercepts (drmTMB parity)" begin
    N = 2000
    cases = [
        ("Gaussian 3 intercepts", () -> drm(bf(@formula(y_g ~ x1 + (1 | g) + (1 | g2) + (1 | h)),
                                               @formula(sigma ~ 1)), Gaussian(); data = _SMR), 1.2477, 0.2507),
        ("Poisson 3 intercepts", () -> drm(bf(@formula(y_pois ~ x1 + (1 | g) + (1 | g2) + (1 | h))),
                                           Poisson(); data = _SMR), 1.9352, 0.3521),
        ("CumulativeLogit 2 intercepts", () -> drm(bf(@formula(y_ord ~ x1 + (1 | g) + (1 | g2))),
                                                   CumulativeLogit(); data = _SMR), 1.2335, 0.1593),
    ]
    for (label, mk, r_meanvar, r_between) in cases
        @testset "$label" begin
            fit = mk()
            S = simulate(fit; nsim = N, data = _SMR, rng = MersenneTwister(20261110))
            @test size(S) == (fit.nobs, N)
            @test all(isfinite, S)
            @test isapprox(mean(var(S; dims = 2)), r_meanvar; rtol = 0.05)
            # between-group variance: variance of the `g` group means, averaged over draws
            # (R: 4000 draws, same definition; Julia agreed within 0.5% at 4000 draws)
            idx = [findall(==(l), _SMR.g) for l in unique(_SMR.g)]
            B = mean(var([mean(view(S, i, s)) for i in idx]) for s in 1:N)
            @test isapprox(B, r_between; rtol = 0.08)
            # the default without `data` is still a clear refusal, never a b = 0 draw
            @test_throws ArgumentError simulate(fit)
            # the refusal prints the exact call to use, and names the grouping columns (maintainer decision of 10 Nov)
            msg = try; simulate(fit); "" ; catch e; sprint(showerror, e); end
            @test occursin("simulate(fit; data = <the data frame you fitted with>)", msg)
            @test occursin("simulate(fit; re = :zero)", msg)
            @test occursin("grouping columns (g, g2", msg)
            # population-level draws carry no between-group signal, so they are less variable
            Z = simulate(fit; nsim = 200, re = :zero, rng = MersenneTwister(1))
            @test mean(var(Z; dims = 2)) < mean(var(S; dims = 2))
        end
    end
    @testset "uncovered structure stays a refusal" begin
        fit = drm(bf(@formula(y_g ~ x1 + (1 | g) + (0 + x1 | g) + (0 + x2 | g2)), @formula(sigma ~ 1)),
                  Gaussian(); data = _SMR)
        @test_throws ArgumentError simulate(fit; data = _SMR)
    end
    @testset "meta_V plus two fields stays a refusal" begin
        fit = drm(bf(@formula(yi ~ x1 + (1 | g) + (1 | g2) + meta_V(vi)), @formula(sigma ~ 1)),
                  Gaussian(); data = _SMR)
        @test_throws ArgumentError simulate(fit; data = _SMR)
    end
    @testset "bootstrap now redraws both fields (was a conditional draw)" begin
        fit = drm(bf(@formula(y_g ~ x1 + (1 | g) + (1 | g2)), @formula(sigma ~ 1)), Gaussian(); data = _SMR)
        sim = DRModels._marginal_simulator(fit, _SMR)
        @test sim isa Function
        res = bootstrap_result(fit; data = _SMR, B = 12, level = 0.95, rng = MersenneTwister(5),
                               failures = :skip, check_converged = false)
        sdrows = [r for r in res.summary if String(r.param) == "resd"]
        @test length(sdrows) == 2
        # the conditional fallback collapsed this interval to nearly zero width (#459)
        @test all(r -> r.upper - r.lower > 0.05, sdrows)
    end
end
