# test_offset_fixed_coef_938.jl — issue #938: `offset(z)` must enter the linear
# predictor with its coefficient fixed at 1, on every route that accepts it, and
# every route that cannot hold it fixed must REFUSE with an error. Before the
# fix only fixed-effects `Poisson()` stripped the marker; every other route sent
# `offset(z)` into the design matrix as an ordinary column with a free,
# estimated coefficient (silently wrong model).
#
# Forcing structure (each block fails on the pre-fix base):
#   * `length(coef(...))` — the offset is never a parameter (Wave L routes 1-3);
#   * the CONSTANT-offset identity — `offset(c)` with a constant column `c` must
#     reproduce the offset-free fit with the intercept shifted by `-c`, exactly,
#     because a fixed offset is an additive constant on the log scale. A free
#     coefficient on a constant column is collinear with the intercept instead;
#   * an independent hand-coded likelihood (Distributions.jl) is evaluated at
#     the fitted θ̂ and must equal `loglik(fit)`, and θ̂ must be a local optimum
#     of that offset-fixed objective;
#   * Gaussian identity: the offset fit equals the fit of `y - offset`;
#   * refusals name the offending formula part.
#
# #1097 shift test: moving the offset by log 2 (location, and the NB2 sigma
# formula) moves the matching intercept by exactly -log 2 and leaves logLik and
# every other coefficient unchanged. `offset =` is the same as `offset(<column>)`.
#
# Data are simulated here (own RNG): no external or AGPL data enter the MIT tree.
# The Owls parity row lives in test_offset_owls_parity_938.jl (gated).

using DRModels
using Test, Random, LinearAlgebra
import Distributions

_off938_names(fit) = reduce(vcat, [String.(p.second) for p in fit.coefnames])
_off938_has_offset_name(fit) = any(n -> occursin("offset", n), _off938_names(fit))

# Counts with a VARYING offset `lo` (log exposure), a covariate `x`, and
# optional group effects.
function _off938_counts(; n = 360, G = 24, seed = 20261016, family = :nb2, re = 0.4)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    lo = log.(rand(rng, n) .* 4 .+ 0.5)
    g = repeat(1:G, inner = cld(n, G))[1:n]
    b = re .* randn(rng, G)
    μ = exp.(0.4 .+ 0.3 .* x .+ lo .+ b[g])
    y = if family === :nb2
        r = 2.0
        Float64[rand(rng, Distributions.NegativeBinomial(r, r / (r + m))) for m in μ]
    else
        Float64[rand(rng, Distributions.Poisson(m)) for m in μ]
    end
    return (; y, x, lo, lc = fill(0.7, n), g = string.("g", g))
end

# Direct NB2 negative log-likelihood with the offset FIXED at 1 (independent of
# the package's kernels): η = Xβ + offset, size r = exp(-2 η_σ).
function _off938_nb2_nll(y, X, Xs, off, θ)
    pμ = size(X, 2)
    η = X * θ[1:pμ] .+ off
    ησ = Xs * θ[(pμ+1):end]
    s = 0.0
    for i in eachindex(y)
        μ = exp(η[i]); r = exp(-2 * ησ[i])
        s -= Distributions.logpdf(Distributions.NegativeBinomial(r, r / (r + μ)), round(Int, y[i]))
    end
    return s
end

@testset "#938: offset() holds a fixed coefficient of 1" begin
    d = _off938_counts()
    n = length(d.y)
    X = hcat(ones(n), d.x)

    @testset "Wave L route 3: NB2 fixed effects, sigma ~ x" begin
        fit = drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1 + x)),
                  NegBinomial2(); data = d)
        @test length(coef(fit, :mu)) == 2
        @test length(coef(fit, :sigma)) == 2
        @test length(coef(fit)) == 4
        @test !_off938_has_offset_name(fit)
        # independent likelihood at θ̂, and θ̂ a local optimum of that objective
        θ̂ = coef(fit)
        nll = θ -> _off938_nb2_nll(d.y, X, X, d.lo, θ)
        @test -nll(θ̂) ≈ loglik(fit) atol = 1e-6
        for k in eachindex(θ̂), δ in (-1e-3, 1e-3)
            θp = copy(θ̂); θp[k] += δ
            @test nll(θp) ≥ nll(θ̂) - 1e-7
        end
        # the offset is not a no-op
        fit0 = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)),
                   NegBinomial2(); data = d)
        @test abs(loglik(fit) - loglik(fit0)) > 1.0
    end

    @testset "constant-offset identity: NB2 fixed effects" begin
        fa = drm(bf(@formula(y ~ 1 + x + offset(lc)), @formula(sigma ~ 1)),
                 NegBinomial2(); data = d)
        fb = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        @test length(coef(fa)) == length(coef(fb))
        @test loglik(fa) ≈ loglik(fb) atol = 1e-5
        @test coef(fa, :mu)[1] ≈ coef(fb, :mu)[1] - 0.7 atol = 1e-3
        @test coef(fa, :mu)[2] ≈ coef(fb, :mu)[2] atol = 1e-3
        @test coef(fa, :sigma) ≈ coef(fb, :sigma) atol = 1e-3
    end

    @testset "Wave L route 1: NB2 + (1 | g), default :LA and :Laplace" begin
        for kw in ((;), (; marginal = :Laplace))
            fit = drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g)), @formula(sigma ~ 1)),
                      NegBinomial2(); data = d, kw...)
            @test length(coef(fit, :mu)) == 2
            @test length(coef(fit)) == 4          # 2 mu + 1 sigma + 1 group SD
            @test !_off938_has_offset_name(fit)
            fa = drm(bf(@formula(y ~ 1 + x + offset(lc) + (1 | g)), @formula(sigma ~ 1)),
                     NegBinomial2(); data = d, kw...)
            fb = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)),
                     NegBinomial2(); data = d, kw...)
            @test length(coef(fa)) == length(coef(fb))
            @test loglik(fa) ≈ loglik(fb) atol = 1e-4
            @test coef(fa, :mu)[1] ≈ coef(fb, :mu)[1] - 0.7 atol = 5e-3
            @test coef(fa, :mu)[2] ≈ coef(fb, :mu)[2] atol = 5e-3
        end
    end

    @testset "Wave L route 2: Poisson + (1 | g), :Laplace" begin
        dp = _off938_counts(family = :poisson)
        fit = drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g))), Poisson();
                  data = dp, marginal = :Laplace)
        @test length(coef(fit, :mu)) == 2
        @test length(coef(fit)) == 3              # 2 mu + 1 group SD
        @test !_off938_has_offset_name(fit)
        fa = drm(bf(@formula(y ~ 1 + x + offset(lc) + (1 | g))), Poisson();
                 data = dp, marginal = :Laplace)
        fb = drm(bf(@formula(y ~ 1 + x + (1 | g))), Poisson();
                 data = dp, marginal = :Laplace)
        @test length(coef(fa)) == length(coef(fb))
        @test loglik(fa) ≈ loglik(fb) atol = 1e-4
        @test coef(fa, :mu)[1] ≈ coef(fb, :mu)[1] - 0.7 atol = 5e-3
    end

    @testset "Gaussian identity: offset fit == fit of y - offset" begin
        rng = MersenneTwister(20261017)
        x = randn(rng, n); lo = log.(rand(rng, n) .* 4 .+ 0.5)
        y = 0.5 .+ 0.8 .* x .+ lo .+ 0.6 .* randn(rng, n)
        dg = (; y, x, lo, yadj = y .- lo)
        fo = drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1)), Gaussian(); data = dg)
        fa = drm(bf(@formula(yadj ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = dg)
        @test length(coef(fo)) == length(coef(fa)) == 3
        @test !_off938_has_offset_name(fo)
        @test coef(fo) ≈ coef(fa) atol = 1e-6
        @test loglik(fo) ≈ loglik(fa) atol = 1e-6
        @test fitted(fo) ≈ fitted(fa) .+ lo atol = 1e-8
        # random intercept: the same equivalence on the mixed-model route
        gg = string.("g", repeat(1:24, inner = cld(n, 24))[1:n])
        dm = (; y = y .+ 0.5 .* randn(rng, 24)[repeat(1:24, inner = cld(n, 24))[1:n]], x, lo, g = gg)
        dm = (; dm..., yadj = dm.y .- lo)
        fm = drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dm)
        fn = drm(bf(@formula(yadj ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = dm)
        @test coef(fm) ≈ coef(fn) atol = 1e-5
        @test loglik(fm) ≈ loglik(fn) atol = 1e-6
    end

    @testset "offset absent from coef, vcov and summary (native Julia)" begin
        fits = (
            drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1 + x)), NegBinomial2(); data = d),
            drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d),
            drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                data = d, marginal = :Laplace),
        )
        for fit in fits
            @test size(vcov(fit)) == (length(coef(fit)), length(coef(fit)))
            @test !_off938_has_offset_name(fit)
            @test !occursin("offset", sprint(show, MIME("text/plain"), fit))
            @test !occursin("offset", sprint(show, fit))
        end
    end

    @testset "predict(newdata) uses the NEW offset" begin
        fit = drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1)),
                  NegBinomial2(); data = d)
        nd0 = (; y = zeros(5), x = collect(range(-1, 1, length = 5)), lo = fill(0.2, 5))
        nd1 = (; y = zeros(5), x = nd0.x, lo = nd0.lo .+ 1.0)
        p0 = predict(fit, nd0); p1 = predict(fit, nd1)
        @test p1 ≈ exp(1.0) .* p0 rtol = 1e-8
        β = coef(fit, :mu)
        @test p0 ≈ exp.(β[1] .+ β[2] .* nd0.x .+ nd0.lo) rtol = 1e-8
        # a missing offset column in newdata is an error, never a silent zero
        @test_throws ArgumentError predict(fit, (; y = zeros(5), x = nd0.x))
    end

    @testset "log 2 shift test (#1097): intercept moves by exactly -log 2" begin
        l2 = log(2.0)
        ds = (; d..., lo2 = d.lo .+ l2)
        # `a` is the fit with the offset, `b` the fit with the offset moved by log 2;
        # `k` is the index of the matching intercept in θ.
        function shifted(a, b, k)
            @test length(coef(a)) == length(coef(b))
            @test loglik(b) ≈ loglik(a) atol = 1e-5
            ca = coef(a); cb = coef(b)
            @test cb[k] ≈ ca[k] - l2 atol = 5e-4
            keep = setdiff(eachindex(ca), (k,))
            @test cb[keep] ≈ ca[keep] atol = 5e-4
        end
        sg = @formula(sigma ~ 1)
        # NB2 fixed effects, location offset (Wave L route 3)
        shifted(drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1 + x)), NegBinomial2(); data = ds),
                drm(bf(@formula(y ~ 1 + x + offset(lo2)), @formula(sigma ~ 1 + x)), NegBinomial2(); data = ds), 1)
        # NB2 fixed effects, SIGMA offset: log σ = γ + offset (θ = [β0 β1 γ0 γ1])
        shifted(drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x + offset(lo))), NegBinomial2(); data = ds),
                drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x + offset(lo2))), NegBinomial2(); data = ds), 3)
        # NB2 (1 | g), location offset, :LA and :Laplace (Wave L route 1)
        for kw in ((;), (; marginal = :Laplace))
            shifted(drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g)), sg), NegBinomial2(); data = ds, kw...),
                    drm(bf(@formula(y ~ 1 + x + offset(lo2) + (1 | g)), sg), NegBinomial2(); data = ds, kw...), 1)
        end
        # NB2 (1 | g), sigma offset (default :LA only; θ = [β0 β1 γ0 σ_b])
        shifted(drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + offset(lo))), NegBinomial2(); data = ds),
                drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + offset(lo2))), NegBinomial2(); data = ds), 3)
        # Poisson fixed effects, and (1 | g) under :LA and :Laplace (Wave L route 2)
        dp = _off938_counts(family = :poisson)
        dps = (; dp..., lo2 = dp.lo .+ l2)
        shifted(drm(bf(@formula(y ~ 1 + x + offset(lo))), Poisson(); data = dps),
                drm(bf(@formula(y ~ 1 + x + offset(lo2))), Poisson(); data = dps), 1)
        for kw in ((;), (; marginal = :Laplace))
            shifted(drm(bf(@formula(y ~ 1 + x + offset(lo) + (1 | g))), Poisson(); data = dps, kw...),
                    drm(bf(@formula(y ~ 1 + x + offset(lo2) + (1 | g))), Poisson(); data = dps, kw...), 1)
        end
        # Gaussian identity
        rng = MersenneTwister(20261018)
        xg = randn(rng, n); log_ = log.(rand(rng, n) .* 4 .+ 0.5)
        dg = (; y = 0.5 .+ 0.8 .* xg .+ log_ .+ 0.6 .* randn(rng, n), x = xg, lo = log_, lo2 = log_ .+ l2)
        shifted(drm(bf(@formula(y ~ 1 + x + offset(lo)), sg), Gaussian(); data = dg),
                drm(bf(@formula(y ~ 1 + x + offset(lo2)), sg), Gaussian(); data = dg), 1)
    end

    @testset "offset = argument is the same as offset(<column>) (#1097)" begin
        fa = drm(bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1)), NegBinomial2(); data = d)
        fb = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2(); data = d, offset = d.lo)
        @test coef(fb) ≈ coef(fa) atol = 1e-8
        @test loglik(fb) ≈ loglik(fa) atol = 1e-8
        @test length(coef(fb, :mu)) == 2
        @test !_off938_has_offset_name(fb)
    end

    @testset "refusals name the route (no silent free coefficient)" begin
        # an offset in sigma where the route cannot hold it fixed
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + offset(lo))),
                                       Gaussian(); data = d)
        err = try
            drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + offset(lo))),
                NegBinomial2(); data = d, marginal = :Laplace); nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("sigma", err.msg) && occursin("offset", err.msg)
        # NB2 and Poisson routes that do not apply the offset must refuse, not fit it as a column
        dh = (; d..., h = string.("h", repeat(1:6, outer = 60)))
        @test_throws ArgumentError drm(
            bf(@formula(y ~ 1 + x + offset(lo) + (1 | g) + (1 | h)), @formula(sigma ~ 1)),
            NegBinomial2(); data = dh)
        @test_throws ArgumentError drm(
            bf(@formula(y ~ 1 + x + offset(lo) + (1 | g) + (1 | h))), Poisson(); data = dh)
        @test_throws ArgumentError drm(
            bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1), @formula(zi ~ 1)),
            NegBinomial2(); data = d)
        @test_throws ArgumentError drm(
            bf(@formula(y ~ 1 + x + offset(lo)), @formula(zi ~ 1)), Poisson(); data = d)
        # other families: refuse, never fit `offset(lo)` as a covariate; and refuse `offset =`
        pos = (; d..., y = d.y .+ 1.0, y01 = clamp.((d.y .+ 0.5) ./ (maximum(d.y) + 1), 0.01, 0.99))
        for (fam, resp) in ((Gamma(), :y), (LogNormal(), :y), (Student(), :y), (Beta(), :y01))
            f = resp === :y ?
                bf(@formula(y ~ 1 + x + offset(lo)), @formula(sigma ~ 1)) :
                bf(@formula(y01 ~ 1 + x + offset(lo)), @formula(sigma ~ 1))
            @test_throws ArgumentError drm(f, fam; data = pos)
            f0 = resp === :y ? bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)) :
                               bf(@formula(y01 ~ 1 + x), @formula(sigma ~ 1))
            @test_throws ArgumentError drm(f0, fam; data = pos, offset = pos.lo)
        end
    end
end
