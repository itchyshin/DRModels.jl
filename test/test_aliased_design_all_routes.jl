# #1110 follow-up: the aliased fixed-effect design check ran only on the Gaussian
# location-scale route, so a Binomial (and some Poisson) fit with `x3 = x1 + x2` came back
# `is_converged = true` with a finite SE on the aliased coefficient. Every univariate route now
# gets the same check: a `fixed_effect_rank_deficiency` warning, `is_converged` false, the SE of
# the aliased coefficient NaN, and the log-likelihood and identified estimates unchanged.
using DRModels
using Test, LinearAlgebra, Logging

_awarns(f) = (lg = Test.TestLogger(min_level = Logging.Warn); res = with_logger(f, lg);
              (res, [string(r.message) for r in lg.logs if r.level == Logging.Warn]))
_arank(ws) = filter(w -> occursin("rank-deficient", w), ws)

@testset "aliased fixed-effect design on every route" begin
    h(i) = mod(sin(12.9898i) * 43758.5453, 1)
    N = 120
    x1 = [2h(i) - 1 for i in 1:N]; x2 = [2h(500 + i) - 1 for i in 1:N]; x3 = x1 .+ x2
    eta = [0.3 + 0.5x1[i] - 0.4x2[i] for i in 1:N]
    yp = [count(j -> h(3000 + 40i + j) < exp(eta[i]) / 40, 1:40) for i in 1:N]
    yb = [count(j -> h(9000 + 8i + j) < 1 / (1 + exp(-eta[i])), 1:8) for i in 1:N]; fb = 8 .- yb
    y01 = [h(1200 + i) < 1 / (1 + exp(-eta[i])) ? 1.0 : 0.0 for i in 1:N]
    yg = [exp(eta[i]) * (0.5 + h(7000 + i)) for i in 1:N]
    ybeta = [clamp(1 / (1 + exp(-eta[i])) * (0.7 + 0.6h(8000 + i)), 0.02, 0.98) for i in 1:N]
    ynorm = [eta[i] + 2 * (h(6000 + i) - 0.5) for i in 1:N]
    # heavy-tailed noise, so the Student fit has a finite degrees-of-freedom estimate
    yt = [eta[i] + (h(6700 + i) < 0.12 ? 4.0 : 0.6) * (h(6300 + i) + h(6400 + i) + h(6500 + i) - 1.5) for i in 1:N]
    data = (; yp, yb, fb, y01, yg, ybeta, ynorm, yt, x1, x2, x3)

    # (label, formula with the alias, same formula with x3 dropped, family, aliased block)
    cases = [
        ("Poisson", bf(@formula(yp ~ x1 + x2 + x3)), bf(@formula(yp ~ x1 + x2)), Poisson(), :mu),
        ("Binomial cbind", bf(@formula(cbind(yb, fb) ~ x1 + x2 + x3)), bf(@formula(cbind(yb, fb) ~ x1 + x2)), Binomial(), :mu),
        ("Binomial 0/1", bf(@formula(y01 ~ x1 + x2 + x3)), bf(@formula(y01 ~ x1 + x2)), Binomial(), :mu),
        ("NegBinomial2", bf(@formula(yp ~ x1 + x2 + x3)), bf(@formula(yp ~ x1 + x2)), NegBinomial2(), :mu),
        ("Gamma", bf(@formula(yg ~ x1 + x2 + x3)), bf(@formula(yg ~ x1 + x2)), Gamma(), :mu),
        ("Beta", bf(@formula(ybeta ~ x1 + x2 + x3)), bf(@formula(ybeta ~ x1 + x2)), Beta(), :mu),
        ("Student", bf(@formula(yt ~ x1 + x2 + x3)), bf(@formula(yt ~ x1 + x2)), Student(), :mu),
        ("Gamma sigma", bf(@formula(yg ~ x1), @formula(sigma ~ x1 + x2 + x3)), bf(@formula(yg ~ x1), @formula(sigma ~ x1 + x2)), Gamma(), :sigma),
    ]
    for (label, f_alias, f_drop, fam, blk) in cases
        @testset "$label" begin
            fit, ws = _awarns(() -> drm(f_alias, fam; data))
            ref, wref = _awarns(() -> drm(f_drop, fam; data))
            # the full-rank twin is untouched
            @test ref.converged
            @test check_drm(ref).fixed_effect_rank_deficiency === nothing
            @test isempty(_arank(wref))
            @test all(isfinite, ref.vcov)
            # the aliased fit is flagged, with one warning that names the block and the column
            @test !is_converged(fit)
            rw = _arank(ws)
            @test length(rw) == 1
            @test occursin("x3", rw[1]) && occursin(String(blk), rw[1])
            rd = check_drm(fit).fixed_effect_rank_deficiency
            @test rd !== nothing && rd.blocks[1].param === blk && rd.blocks[1].aliased == ["x3"]
            # SE of the aliased coefficient is NaN, the other coefficients keep finite SEs
            idx = findfirst(p -> first(p) === blk, fit.blocks)
            rng = last(fit.blocks[idx])
            se = sqrt.(abs.(diag(fit.vcov)))
            @test isnan(se[rng[end]])
            @test all(isfinite, se[setdiff(1:length(se), rng[end])])
            # the likelihood does not move
            @test isapprox(loglikelihood(fit), loglikelihood(ref); atol = 1e-5)
        end
    end

    @testset "full-rank fits give no rank warning and finite SEs" begin
        for (label, _, f_drop, fam, _) in cases
            fit, ws = _awarns(() -> drm(f_drop, fam; data))
            @test isempty(_arank(ws))
            @test all(isfinite, fit.vcov)
        end
    end

    @testset "a random intercept does not hide the alias" begin
        g = ["g" * string(1 + (i % 12)) for i in 1:N]
        d2 = merge(data, (; g))
        fit, ws = _awarns(() -> drm(bf(@formula(yp ~ x1 + x2 + x3 + (1 | g))), Poisson(); data = d2))
        @test !is_converged(fit)
        @test length(_arank(ws)) == 1
    end

    @testset "LogNormal marginal = :Laplace warns once" begin
        g = ["g" * string(1 + (i % 12)) for i in 1:N]
        yl = exp.(0.2 .* yg .+ [0.3 * sin(i % 12) for i in 1:N])
        d3 = merge(data, (; g, yl))
        f = bf(@formula(yl ~ x1 + x2 + x3 + (1 | g)), @formula(sigma ~ 1))
        for marg in (:LA, :Laplace)
            fit, ws = _awarns(() -> drm(f, LogNormal(); data = d3, marginal = marg))
            @test length(_arank(ws)) == 1
            @test !is_converged(fit)
            @test check_drm(fit).fixed_effect_rank_deficiency !== nothing
        end
    end
end
