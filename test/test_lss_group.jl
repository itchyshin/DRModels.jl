# #544 — location–scale–scale for the plain (1 | g) route: `sd(g) ~ x` puts a
# linear predictor on the log RE SD (drmTMB's `sd(group)` grammar). Cross-engine
# evidence (2026-08-28, identical CSV): logLik -417.7794 both engines, every
# coefficient block agreeing ≤ 1e-5 vs drmTMB's native TMB fit.
using Test
using DRModels
using StableRNGs
using Statistics

# The drmTMB location-scale-scale vignette design: sex moves the mean, the
# between-individual SD, and the within-individual SD, all at once.
function _lss_sim(; n_id = 80, n_each = 6)
    rng = StableRNG(20260715)
    sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
    b = randn(rng, n_id) .* [0.65, 0.40][Int.(sex) .+ 1]
    id = repeat(1:n_id, inner = n_each)
    sexl = sex[id]
    y = [0.35, 0.70][Int.(sexl) .+ 1] .+ b[id] .+
        randn(rng, n_id * n_each) .* [0.35, 0.60][Int.(sexl) .+ 1]
    return (y = y, sex = sexl, id = id)
end

@testset "lss sd(group): grammar" begin
    f = bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))
    @test any(p -> first(p) === :sd_id, f.forms)
    # duplicate sd() formula
    @test_throws ArgumentError bf(@formula(y ~ x + (1 | g)),
                                  @formula(sd(g) ~ 1), @formula(sd(g) ~ 1))
    # tau stays rejected, sd() does not relax the dpar whitelist
    @test_throws ArgumentError bf(@formula(y ~ x), @formula(tau ~ 1))
end

@testset "lss sd(group): router refusals are loud" begin
    dat = _lss_sim()
    fsd = bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex))
    # non-Gaussian family: refused, never silently dropped
    datp = (y = round.(Int, abs.(dat.y) .* 3) .* 1.0, sex = dat.sex, id = dat.id)
    @test_throws ArgumentError drm(bf(@formula(y ~ sex + (1 | id)), @formula(sd(id) ~ sex)),
                                   Poisson(); data = datp)
    # sd() names a group with no matching (1 | g)
    @test_throws ArgumentError drm(bf(@formula(y ~ sex + (1 | id)), @formula(sd(sex) ~ 1)),
                                   Gaussian(); data = dat)
    # no random effect at all
    @test_throws ArgumentError drm(bf(@formula(y ~ sex), @formula(sd(id) ~ 1)),
                                   Gaussian(); data = dat)
    # sd() predictor varying within group (an observation-level covariate)
    datv = (y = dat.y, sex = dat.sex, id = dat.id, xobs = collect(1.0:length(dat.y)))
    @test_throws ArgumentError drm(bf(@formula(y ~ sex + (1 | id)), @formula(sd(id) ~ xobs)),
                                   Gaussian(); data = datv)
end

@testset "lss sd(group): reduction invariant + fit" begin
    dat = _lss_sim()
    f1 = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ 1)),
             Gaussian(); data = dat)
    f0 = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex)), Gaussian(); data = dat)
    @test f1.converged && f0.converged
    @test abs(f1.loglik - f0.loglik) ≤ 1e-8
    @test maximum(abs.(f1.theta .- f0.theta)) ≤ 1e-6

    fit = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex)),
              Gaussian(); data = dat)
    @test fit.converged
    # cross-engine pinned values (drmTMB agrees ≤ 1e-5 on this exact fixture)
    @test isapprox(fit.loglik, -417.7794; atol = 1e-2)
    @test isapprox(coef(fit, :mu), [0.45890, 0.11795]; atol = 1e-2)
    @test isapprox(coef(fit, :sigma), [-1.01563, 0.55507]; atol = 1e-2)
    @test isapprox(coef(fit, :sd), [-0.39921, -0.34843]; atol = 1e-2)
    # dropping sex from sd() is an interior null — ordinary LRT applies and the
    # nested logLik ordering must hold
    @test fit.loglik ≥ f1.loglik - 1e-8
end

@testset "lss sd(group): REML and post-fit surfaces" begin
    dat = _lss_sim()
    fit = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex)),
              Gaussian(); data = dat)
    fr = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex)),
             Gaussian(); data = dat, method = :REML)
    @test fr.converged
    @test fr.estim_method === :REML && isfinite(fr.reml_loglik)

    # the :sd block is a first-class coefficient block
    @test length(coef(fit, :sd)) == 2
    ci = confint(fit; parm = :sd)
    @test length(ci) == 2 && all(r -> r.lower < r.estimate < r.upper, ci)

    # single-SD summaries are ill-defined and must refuse, not misreport
    @test_throws ArgumentError re_sd(fit)
    @test_throws ArgumentError vc(fit)
    @test_throws ErrorException DRModels.heritability(fit)

    # BLUPs exist per group and are centered-ish
    @test haskey(fit.ranef, :id) && length(fit.ranef[:id]) == 80
    @test abs(mean(fit.ranef[:id])) < 0.2
end
