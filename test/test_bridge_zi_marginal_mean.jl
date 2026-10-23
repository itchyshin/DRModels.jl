# The bridge's `fitted`/`residuals` for a ZERO-INFLATED count fit
# (`_bridge_fitted_marginal`, src/bridge.jl).
#
# DRModels.jl's own `fitted(fit)` is `means[:mu]`, and for a zero-inflated Poisson or
# NegBinomial2 fit that slot deliberately holds the COUNT-COMPONENT mean
# `exp(Xmu*betahat)` — `simulate`, `marginal_parameters` and `_bridge_dpars`
# all read the component mean from it. drmTMB's `fitted()` for `zi_poisson` /
# `zi_nbinom2` is the UNCONDITIONAL mean `(1 - pi) * mu`, and its `residuals()`
# is `y - fitted`. So the bridge, which is drmTMB's contract surface, must send
# the unconditional mean while `dpars["mu"]` keeps the component mean.
#
# The gap this closes was silent: measured 2026-09-05 through drmTMB on the
# fixture in its tests/testthat/test-zi-nbinom2.R (n = 1800), the two engines'
# coefficients agreed to 4.57e-13 and their logLik to 1.73e-11 while their
# `fitted()` disagreed by 1.3665229755584 — no coefficient or likelihood check
# can see that.
#
# What must NOT change: hurdle (`hu ~ ...`) fits, plain counts, and every
# non-count family keep DRModels.jl's own `fitted`, and `dpars["mu"]` stays the
# component mean for the zero-inflated fits too.
using DRModels
using Test
using Random

function _zi_marginal_data(; n = 300, seed = 20260905)
    rng = Random.MersenneTwister(seed)
    x = randn(rng, n)
    w = randn(rng, n)
    mu = exp.(0.6 .+ 0.4 .* x)
    pi_zi = 1 ./ (1 .+ exp.(-(-0.8 .+ 0.5 .* w)))
    y = [rand(rng) < pi_zi[i] ? 0.0 : float(rand(rng, 0:6)) for i in 1:n]
    return (; y = y, x = x, w = w)
end

const _ZIM_DATA = _zi_marginal_data()

_zim_bridge(formula, family) =
    drm_bridge(; formula = formula, family = family, data = _ZIM_DATA)

@testset "bridge fitted/residuals for zero-inflated count fits" begin

    @testset "(a) zero-inflated NB2: fitted is the UNCONDITIONAL mean" begin
        out = _zim_bridge(Dict("mu" => "y ~ x", "sigma" => "sigma ~ 1",
                               "zi" => "zi ~ w"), "nbinom2")
        fitted_vals = out["fitted"]
        mu = out["dpars"]["mu"]
        pz = out["dpars"]["zi"]
        @test length(fitted_vals) == length(_ZIM_DATA.y)
        # the repair itself
        @test isapprox(fitted_vals, (1 .- pz) .* mu; atol = 1e-12)
        @test isapprox(out["residuals"], _ZIM_DATA.y .- fitted_vals; atol = 1e-12)
        # ... and it is a REAL change, not a no-op: zero inflation is present,
        # so the component mean and the unconditional mean genuinely differ.
        @test maximum(pz) > 0.05
        @test maximum(abs.(fitted_vals .- mu)) > 1e-3
        # the dpar table is untouched: `mu` is still DRModels.jl's own `fitted`,
        # i.e. the count-component mean, which is the `mu` dpar drmTMB wants
        direct = drm(bf(@formula(y ~ x), @formula(sigma ~ 1), @formula(zi ~ w)),
                     NegBinomial2(); data = _ZIM_DATA)
        @test isapprox(mu, fitted(direct); atol = 1e-8)
    end

    @testset "(b) zero-inflated Poisson: the same repair" begin
        out = _zim_bridge(Dict("mu" => "y ~ x", "zi" => "zi ~ w"), "poisson")
        fitted_vals = out["fitted"]
        mu = out["dpars"]["mu"]
        pz = out["dpars"]["zi"]
        @test isapprox(fitted_vals, (1 .- pz) .* mu; atol = 1e-12)
        @test isapprox(out["residuals"], _ZIM_DATA.y .- fitted_vals; atol = 1e-12)
        @test maximum(abs.(fitted_vals .- mu)) > 1e-3
    end

    @testset "(c) plain NB2 and Poisson are byte-unchanged" begin
        for fam in ("nbinom2", "poisson")
            out = _zim_bridge(Dict("mu" => "y ~ x"), fam)
            fit = drm(bf(@formula(y ~ x)),
                      fam == "nbinom2" ? NegBinomial2() : DRModels.Poisson();
                      data = _ZIM_DATA)
            @test isapprox(out["fitted"], fitted(fit); atol = 1e-10)
            @test isapprox(out["residuals"], residuals(fit); atol = 1e-10)
            @test isapprox(out["fitted"], out["dpars"]["mu"]; atol = 1e-12)
        end
    end

    @testset "(d) hurdle fits are deliberately NOT repaired here" begin
        # drmTMB's hurdle mean divides by `1 - P(0)` as well as scaling, and
        # that route has no bridge parity receipt yet. `scales` carries `:hu`,
        # not `:zi`, so `_bridge_fitted_marginal` must leave it alone.
        out = _zim_bridge(Dict("mu" => "y ~ x", "sigma" => "sigma ~ 1",
                               "hu" => "hu ~ w"), "nbinom2")
        @test haskey(out["dpars"], "hu")
        @test !haskey(out["dpars"], "zi")
        @test isapprox(out["fitted"], out["dpars"]["mu"]; atol = 1e-12)
        @test isapprox(out["residuals"], _ZIM_DATA.y .- out["dpars"]["mu"];
                       atol = 1e-12)
    end

    @testset "(e) a non-count family is untouched" begin
        gnoise = [0.15 * sin(2.7 * i) for i in 1:length(_ZIM_DATA.x)]
        gdata = (; y = _ZIM_DATA.x .* 0.7 .+ 0.2 .+ gnoise, x = _ZIM_DATA.x)
        out = drm_bridge(; formula = Dict("mu" => "y ~ x", "sigma" => "sigma ~ 1"),
                         family = "gaussian", data = gdata)
        @test isapprox(out["fitted"], out["dpars"]["mu"]; atol = 1e-12)
        @test isapprox(out["residuals"], gdata.y .- out["dpars"]["mu"]; atol = 1e-12)
    end

    @testset "(f) the helper itself: selection is scales[:zi], and it is total" begin
        # A fit with no `:zi` scale returns DRModels.jl's own values unchanged --
        # the guard, not an accident of the families exercised above.
        fit = drm(bf(@formula(y ~ x)), DRModels.Poisson(); data = _ZIM_DATA)
        f, r = DRModels._bridge_fitted_marginal(fit)
        @test !haskey(fit.scales, :zi)
        @test f == fitted(fit)
        @test isapprox(r, residuals(fit); atol = 1e-12)
    end
end
