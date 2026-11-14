# `residuals(fit; type = :pearson)`: (y - E[Y]) / sqrt(Var[Y]) from the fitted distribution of each
# observation, for fixed-effect Gaussian, Poisson, NegBinomial2, Gamma and Beta fits. The values are
# pinned from drmTMB 0.8.1 `residuals(fit, type = "pearson")` (generating script and data in
# test/fixtures/pearson/). Other families and routes are refused by name.
using DRModels, Test, Random, DelimitedFiles

const _PEARSON_DIR = joinpath(@__DIR__, "fixtures", "pearson")

function _pearson_fixture(name)
    m, _ = readdlm(joinpath(_PEARSON_DIR, "pearson_$name.csv"), ','; header = true)
    return (; y = Float64.(m[:, 1]), x = Float64.(m[:, 2])), Float64.(m[:, 3])
end

_pmsg(f) = try; f(); ""; catch e; e isa ArgumentError ? e.msg : rethrow(); end

@testset "Pearson residuals match drmTMB 0.8.1 (1e-6), one cell per supported family" begin
    cells = [
        ("gaussian", () -> (bf(@formula(y ~ x), @formula(sigma ~ x)), DRModels.Gaussian())),
        ("poisson", () -> (bf(@formula(y ~ x)), DRModels.Poisson())),
        ("nbinom2", () -> (bf(@formula(y ~ x), @formula(sigma ~ 1)), NegBinomial2())),
        ("gamma", () -> (bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Gamma())),
        ("beta", () -> (bf(@formula(y ~ x), @formula(sigma ~ 1)), DRModels.Beta())),
    ]
    for (name, spec) in cells
        data, rr = _pearson_fixture(name)
        f, fam = spec()
        fit = drm(f, fam; data)
        @test fit.converged
        r = residuals(fit; type = :pearson)
        @test length(r) == length(rr)
        @test maximum(abs.(r .- rr)) < 1e-6
    end
end

@testset "Pearson residuals: definition (y - mean) / sd of the fitted distribution" begin
    data, _ = _pearson_fixture("poisson")
    fit = drm(bf(@formula(y ~ x)), DRModels.Poisson(); data)
    mu = fitted(fit)
    @test residuals(fit; type = :pearson) ≈ (data.y .- mu) ./ sqrt.(mu) rtol = 1e-12
end

@testset "Pearson residuals: other types are unchanged" begin
    data, _ = _pearson_fixture("gaussian")
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), DRModels.Gaussian(); data)
    @test residuals(fit) == residuals(fit; type = :response)
    @test residuals(fit; type = :quantile, rng = Random.Xoshiro(1)) ==
          residuals(fit; type = :quantile, rng = Random.Xoshiro(1))
    @test occursin(":pearson", _pmsg(() -> residuals(fit; type = :nope)))
end

@testset "Pearson residuals: unsupported families and routes are refused by name" begin
    r = Random.Xoshiro(7); n = 150
    x = randn(r, n); g = string.("g", rand(r, 1:15, n))
    yc = 1.0 .+ 0.5 .* x .+ randn(r, n)
    ypos = exp.(0.3 .+ 0.4 .* x .+ 0.3 .* randn(r, n))
    ycount = Float64.(rand(r, 0:6, n))
    s = Float64.(rand(r, 0:8, n))
    d = (; x, g, y = yc, yp = ypos, yn = ycount, s, f = 8.0 .- s)
    student = drm(bf(@formula(y ~ x)), Student(); data = d)
    lognorm = drm(bf(@formula(yp ~ x)), LogNormal(); data = d)
    binom = drm(bf(@formula(cbind(s, f) ~ x)), Binomial(); data = d)
    for (label, fit) in (("Student()", student), ("LogNormal()", lognorm), ("Binomial()", binom))
        msg = _pmsg(() -> residuals(fit; type = :pearson))
        @test occursin("not implemented for $label", msg)
        @test occursin("type = :quantile", msg)
    end
    re = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gaussian(); data = d)
    msg = _pmsg(() -> residuals(re; type = :pearson))
    @test occursin("fixed-effect fits only", msg)
    @test occursin("Gaussian()", msg)
    pre = drm(bf(@formula(yn ~ x + (1 | g))), DRModels.Poisson(); data = d)
    @test occursin("fixed-effect fits only", _pmsg(() -> residuals(pre; type = :pearson)))
end

# Diagonal meta_V (a known per-row sampling variance, no random effect) is a fixed-effect fit and is
# supported: the variance of the fitted distribution is v + sigma^2, as in drmTMB 0.8.1.
@testset "Pearson residuals: diagonal meta_V matches drmTMB 0.8.1" begin
    m, _ = readdlm(joinpath(_PEARSON_DIR, "pearson_gaussian_metav.csv"), ','; header = true)
    data = (; y = Float64.(m[:, 1]), x = Float64.(m[:, 2]), v = Float64.(m[:, 3]))
    rr = Float64.(m[:, 4])
    fit = drm(bf(@formula(y ~ x + meta_V(v)), @formula(sigma ~ 1)), DRModels.Gaussian(); data)
    @test fit.converged
    r = residuals(fit; type = :pearson)
    @test length(r) == length(rr)
    @test maximum(abs.(r .- rr)) < 1e-6
end

# One entry per used row; rows dropped for a missing response are not returned, as for `:response` and
# `:quantile`, and as in drmTMB, so the R values are compared row by row on the kept rows.
@testset "Pearson residuals: rows dropped for a missing response are not returned" begin
    m, _ = readdlm(joinpath(_PEARSON_DIR, "pearson_poisson_miss.csv"), ','; header = true)
    ymiss = Union{Missing,Float64}[v isa Number ? Float64(v) : missing for v in m[:, 1]]
    rr = Float64[v isa Number ? Float64(v) : NaN for v in m[:, 3]]
    data = (; y = ymiss, x = Float64.(m[:, 2]))
    fit = drm(bf(@formula(y ~ x)), DRModels.Poisson(); data)
    r = residuals(fit; type = :pearson)
    dropped = ismissing.(ymiss)
    @test length(r) == count(!, dropped)
    @test !any(isnan, r)
    @test maximum(abs.(r .- rr[.!dropped])) < 1e-6
    # the other residual types follow the same convention
    @test length(residuals(fit)) == count(!, dropped) && !any(isnan, residuals(fit))
    @test length(residuals(fit; type = :quantile, rng = Random.Xoshiro(1))) == count(!, dropped)
end

# Each clause of the fixed-effect guard is pinned by its own fit.
@testset "Pearson residuals: each refusal clause, separately" begin
    r = Random.Xoshiro(11); n = 200
    x = randn(r, n); g = string.("g", rand(r, 1:20, n)); v = 0.1 .+ rand(r, n) ./ 5
    yc = 1.0 .+ 0.5 .* x .+ randn(r, n)
    yz = Float64[rand(r) < 0.3 ? 0.0 : rand(r, 1:6) for _ in 1:n]
    ypos = Float64.(rand(r, 1:6, n))
    d = (; x, g, v, yc, yz, ypos)
    fixed = ("fixed-effect fits only", "type = :quantile")
    cases = [
        ("zero inflation", () -> drm(bf(@formula(yz ~ x), @formula(zi ~ 1)), DRModels.Poisson(); data = d)),
        ("hurdle", () -> drm(bf(@formula(yz ~ x), @formula(hu ~ 1)), DRModels.Poisson(); data = d)),
        ("scale random effect", () -> drm(bf(@formula(yc ~ x), @formula(sigma ~ 1 + (1 | g))), DRModels.Gaussian(); data = d)),
        ("meta_V with a random effect", () -> drm(bf(@formula(yc ~ x + meta_V(v) + (1 | g)), @formula(sigma ~ 1)), DRModels.Gaussian(); data = d)),
        ("mean random effect", () -> drm(bf(@formula(yc ~ x + (1 | g))), DRModels.Gaussian(); data = d)),
    ]
    for (label, mk) in cases
        fit = mk()
        msg = _pmsg(() -> residuals(fit; type = :pearson))
        @testset "$label" begin
            @test occursin("fixed-effect fits only", msg)
            @test occursin("type = :quantile", msg)
        end
    end
    # a truncated family is refused by its name before the route check
    tp = drm(bf(@formula(ypos ~ x)), TruncatedPoisson(); data = d)
    @test occursin("not implemented for TruncatedPoisson()", _pmsg(() -> residuals(tp; type = :pearson)))
end
