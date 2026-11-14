# test_cumlogit_ordinary_laplace_parity.jl -- CumulativeLogit with one ordinary `(1 | g)` and
# `marginal = :Laplace` vs drmTMB 0.8.1. References: `test/fixtures/cumlogit_laplace_parity/`
# written by `test/parity/gen_cumlogit_laplace_parity.R` (drmTMB 0.8.1; max|grad| < 1e-6, all SEs
# finite; numbers and simulated data only). Bounds are the lane's and are never loosened:
#   logLik 1e-4 (absolute) | fixed effects and thresholds 1e-3 R SE | log SD 5e-3 |
#   SEs 1e-3 relative | category probabilities 1e-6.
using DRModels, Test, TOML, Random, LinearAlgebra, Statistics

const _CLL_DIR = joinpath(@__DIR__, "fixtures", "cumlogit_laplace_parity")

function _cll_data(route)
    L = readlines(joinpath(_CLL_DIR, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    cols[:y] = round.(Int, cols[:y])
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

function _cll_named(fit)
    θ = coef(fit); V = vcov(fit)
    est = Dict{String,Float64}(); se = Dict{String,Float64}(); i = 0
    for (param, r) in fit.blocks
        for (nm, v) in zip(Dict(fit.coefnames)[param], θ[r])
            i += 1
            est["$(param)_$(nm)"] = v
            se["$(param)_$(nm)"] = V[i, i] > 0 ? sqrt(V[i, i]) : NaN
        end
    end
    return est, se
end

const _CLL_FORM = bf(@formula(y ~ x + (1 | g)))

@testset "CumulativeLogit (1 | g) marginal = :Laplace vs drmTMB 0.8.1, four data sets" begin
    for route in ("cl_1", "cl_2", "cl_3", "cl_4")
        d = _cll_data(route)
        ref = TOML.parsefile(joinpath(_CLL_DIR, route, "ref.toml"))
        K = ref["K"]; n = length(d.y)
        @test ref["max_grad"] < 1e-6 && ref["drmTMB_version"] == "0.8.1"
        fit = drm(_CLL_FORM, CumulativeLogit(); data = d, marginal = :Laplace)
        est, se = _cll_named(fit)
        @testset "$route (K = $K, $(length(unique(d.g))) groups)" begin
            @test is_converged(fit)
            @test fit.marginal === :Laplace
            @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
            for (nm, r, s) in zip(ref["names"], ref["theta"], ref["se"])
                @test haskey(est, nm)
                tol = startswith(nm, "resd_") ? 5e-3 : 1e-3 * s
                @test abs(est[nm] - r) <= tol
                @test isfinite(se[nm]) && abs(se[nm] / s - 1) <= 1e-3
            end
            # category probabilities at the population level (random effect at zero)
            η = predict(fit, d)
            cuts = DRModels._cumulative_cuts(coef(fit, :cutpoints))
            cdf = hcat(zeros(n), [1 ./ (1 .+ exp.(-(c .- η))) for c in cuts]..., ones(n))
            P = cdf[:, 2:end] .- cdf[:, 1:end-1]
            Pref = permutedims(reshape(Float64.(ref["prob"]), K, n))
            @test maximum(abs.(P .- Pref)) <= 1e-6
            @test maximum(abs.(fitted(fit; re = :zero) .- Pref * collect(1.0:K))) <= 1e-6   # population value
            # fitted() is conditional (random-effect modes included), as drmTMB's fitted()
            rf = [split(l, ',') for l in readlines(joinpath(_CLL_DIR, route, "fitted_r.csv"))[2:end]]
            @test length(rf) == n
            @test maximum(abs.(fitted(fit) .- parse.(Float64, getindex.(rf, 2)))) <= 1e-8
            @test maximum(abs.(fitted(fit; re = :zero) .- parse.(Float64, getindex.(rf, 3)))) <= 1e-6
            # the default integrator is unchanged and is a different number
            fla = drm(_CLL_FORM, CumulativeLogit(); data = d)
            @test fla.marginal === :LA
            @test abs(loglik(fla) - loglik(fit)) > 1e-3
            # explicit :LA is the default, bit for bit
            fex = drm(_CLL_FORM, CumulativeLogit(); data = d, marginal = :LA)
            @test coef(fex) == coef(fla) && loglik(fex) == loglik(fla)
        end
    end
end

@testset "CumulativeLogit :Laplace post-fit functions" begin
    d = _cll_data("cl_1"); n = length(d.y)
    fit = drm(_CLL_FORM, CumulativeLogit(); data = d, marginal = :Laplace)
    fla = drm(_CLL_FORM, CumulativeLogit(); data = d)
    p = length(coef(fit))
    @test nobs(fit) == n
    @test loglikelihood(fit) == loglik(fit)
    V = vcov(fit)
    @test size(V) == (p, p) && issymmetric(V) && minimum(eigvals(Symmetric(V))) > 0
    @test stderror(fit) ≈ sqrt.(diag(V))
    @test aic(fit) ≈ -2 * loglik(fit) + 2p
    # the objective stored on the fit is the Laplace one, not the quadrature one
    θ̂ = coef(fit)
    @test fit.nll(θ̂) ≈ -loglik(fit) atol = 1e-9
    @test abs(fla.nll(θ̂) - fit.nll(θ̂)) > 1e-3
    # population-level predictions and fitted values: same as the default route's definition
    @test predict(fit, d) ≈ d.x .* coef(fit, :mu)[1]
    @test predict(fit, merge(d, (g = fill("NEW", n),))) ≈ predict(fit, d)
    @test length(fitted(fit)) == n && all(1 .<= fitted(fit) .<= 3)
    @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == n
    @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == n
    @test length(residuals(fit; type = :quantile)) == n
    @test re_sd(fit)[:g] ≈ exp(coef(fit, :resd)[1])
    # random-effect modes are not stored (as on the default route): refused by name
    @test_throws ArgumentError ranef(fit)
    # Wald and profile intervals; the profile uses the Laplace objective (it differs from :LA's)
    ci = confint(fit)
    @test all(c -> c.lower < c.estimate < c.upper, ci)
    cp = confint(fit; method = :profile)
    cpl = confint(fla; method = :profile)
    @test all(c -> c.lower < c.estimate < c.upper, cp)
    @test maximum(abs.([a.lower - b.lower for (a, b) in zip(cp, cpl)])) > 1e-4
    # lrtest: against the fixed-effects model works; against a :LA fit it is refused
    f0 = drm(bf(@formula(y ~ x)), CumulativeLogit(); data = d)
    lr = @test_logs (:warn,) match_mode = :any lrtest(f0, fit)
    @test lr.statistic ≈ 2 * (loglik(fit) - loglik(f0))
    @test_throws ArgumentError lrtest(fla, fit)
    # bootstrap: every replicate is refitted with :Laplace (replay each seed and refit both ways)
    br = bootstrap_result(fit; data = d, B = 3, rng = Random.Xoshiro(1))
    @test br.used == 3 && br.marginal === :Laplace && br.refit_marginal === :Laplace
    sim = DRModels._marginal_simulator(fit, d)
    reps = map(br.seeds) do seed
        datab = DRModels._bootstrap_data(fit.formula, d, sim(Random.MersenneTwister(seed)))
        (lap = drm(_CLL_FORM, CumulativeLogit(); data = datab, marginal = :Laplace),
         la = drm(_CLL_FORM, CumulativeLogit(); data = datab))
    end
    @test all(r -> r.lap.marginal === :Laplace, reps)
    rep_sd(k) = vec(Statistics.std(permutedims(reduce(hcat, [coef(getproperty(r, k)) for r in reps])); dims = 1))
    boot_sd = [row.std_error for row in br.summary]
    @test boot_sd ≈ rep_sd(:lap) rtol = 1e-8
    @test !isapprox(boot_sd, rep_sd(:la); rtol = 1e-6)
    # a response with a missing value drops that row, as on the default route
    ym = Union{Missing,Int}[i == 1 ? missing : v for (i, v) in enumerate(d.y)]
    fm = @test_logs (:warn,) match_mode = :any drm(_CLL_FORM, CumulativeLogit(); data = merge(d, (y = ym,)), marginal = :Laplace)
    @test nobs(fm) == n - 1 && fm.marginal === :Laplace
    # the bridge forwards the option and reports the integrator used
    outL = drm_bridge(; formula = "y ~ x + (1 | g)", family = "cumulative_logit", data = d,
                      options = Dict(:marginal => "Laplace"))
    outD = drm_bridge(; formula = "y ~ x + (1 | g)", family = "cumulative_logit", data = d)
    @test outL["marginal"] == "Laplace" && outD["marginal"] == "LA"
    @test outL["loglik"] ≈ loglik(fit) atol = 1e-8
    @test abs(outL["loglik"] - outD["loglik"]) > 1e-3
end

@testset "CumulativeLogit :Laplace refusals" begin
    d = _cll_data("cl_4"); n = length(d.y)
    dh = merge(d, (h = repeat(["a", "b", "c", "d"], cld(n, 4))[1:n],))
    lap(f; data = dh) = drm(f, CumulativeLogit(); data = data, marginal = :Laplace)
    for (what, bad) in (
            "fixed effects only" => bf(@formula(y ~ x)),
            "random slope" => bf(@formula(y ~ x + (0 + x | g))),
            "correlated block" => bf(@formula(y ~ x + (1 + x | g))),
            "offset" => bf(@formula(y ~ x + offset(x) + (1 | g))),
            "crossed" => bf(@formula(y ~ x + (1 | g) + (1 | h))),
            "sd formula" => bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ x)),
            "sigma formula" => bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)))
        @testset "$what" begin
            @test_throws ArgumentError lap(bad)
        end
    end
    e = try lap(bf(@formula(y ~ x + (0 + x | g)))) catch err; sprint(showerror, err) end
    @test occursin("marginal = :Laplace", e) && occursin("CumulativeLogit", e)
    @test_throws ArgumentError lap(bf(@formula(y ~ x + phylo(1 | g))))
    for m in (:VA, :AGHQ, :bogus)
        @test_throws ArgumentError drm(_CLL_FORM, CumulativeLogit(); data = d, marginal = m)
    end
    @test_throws ArgumentError drm(_CLL_FORM, CumulativeLogit(); data = d, marginal = :Laplace, method = :REML)
end
