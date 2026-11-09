using DRModels, Test
import DelimitedFiles

# Non-Gaussian ordinary correlated blocks the wave-10 gap map listed as refused while
# drmTMB 0.8.1 fits them (design: docs/dev-log/2026-11-09-nongauss-corr-routes.md):
#   Poisson and NB2 `(1 + x1 | g)` with an SD formula on ONE coefficient (`sd-first`,
#   `sd-last`); LogNormal `(1 + x1 | g)` with SD formulas on one or both coefficients;
#   Binomial `(1 + x1 + x2 [+ x3] | g)` with no SD formula.
# Data are the wave-8 fuzz data (committed): test/fixtures/nongauss_corr/ for Poisson, NB2 and
# LogNormal, test/fixtures/binomial_indep_restarts/g12_binomial.csv for Binomial (the same
# 400-row, 40-group file). The R reference log-likelihoods are drmTMB 0.8.1 ML on the same files
# (Totoro library ~/drm_campaign_20261108/Rlib; test/fixtures/nongauss_corr/run_r_nongauss_corr.R).

const _NGC_DIR = joinpath(@__DIR__, "fixtures", "nongauss_corr")

function _ngc_read(path)
    m, h = DelimitedFiles.readdlm(path, ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end
_ngc_data(fam::AbstractString) = _ngc_read(joinpath(_NGC_DIR, "data_$(fam)_ordinary.csv"))

const _NGC_MU = @formula(y ~ x1 + x2 + x3 + (1 + x1 | g))
const _NGC_SD = Dict(
    :first => @formula(sd(g, coef(intercept)) ~ z),
    :last  => @formula(sd(g, coef(x1)) ~ z),
    :one0  => @formula(sd(g, coef(intercept)) ~ 1),
    :one1  => @formula(sd(g, coef(x1)) ~ 1),
    :z0    => @formula(sd(g, coef(intercept)) ~ z),
    :z1    => @formula(sd(g, coef(x1)) ~ z),
)

@testset "Poisson and NB2 (1 + x1 | g): one SD formula, the other coefficient keeps a constant SD" begin
    # drmTMB 0.8.1 logLik: Poisson first -712.3057715513, last -711.3985694172;
    # NB2 first -696.3014043646, last -698.1310518253. Julia agrees to 1e-9 (tolerance 1e-4, the map's).
    for (fam, mk, refs) in (("poisson", () -> Poisson(), (first = -712.3057715513, last = -711.3985694172)),
                            ("nb2", () -> NegBinomial2(), (first = -696.3014043646, last = -698.1310518253)))
        d = _ngc_data(fam)
        for which in (:first, :last)
            parts = fam == "nb2" ? (_NGC_MU, @formula(sigma ~ 1), _NGC_SD[which]) : (_NGC_MU, _NGC_SD[which])
            fit = drm(bf(parts...), mk(); data = d)
            @test fit.converged
            @test loglikelihood(fit) ≈ getproperty(refs, which) atol = 1e-4
            @test all(isfinite, stderror(fit))
            # the model is the two-formula fit with `~ 1` on the other coefficient
            other = which === :first ? _NGC_SD[:one1] : _NGC_SD[:one0]
            parts2 = fam == "nb2" ? (_NGC_MU, @formula(sigma ~ 1), _NGC_SD[which], other) :
                                    (_NGC_MU, _NGC_SD[which], other)
            fit2 = drm(bf(parts2...), mk(); data = d)
            @test loglikelihood(fit) ≈ loglikelihood(fit2) atol = 1e-6
            @test length(fit.theta) == length(fit2.theta)
        end
    end
end

@testset "Poisson and NB2: refusals that must stay" begin
    d = _ngc_data("poisson")
    # a target on a coefficient the block does not have
    @test_throws ArgumentError drm(bf(_NGC_MU, @formula(sd(g, coef(x2)) ~ z)), Poisson(); data = d)
    # the same coefficient twice
    @test_throws ArgumentError drm(bf(_NGC_MU, @formula(sd(g, coef(x1)) ~ z), @formula(sd(g, coef(x1)) ~ 1)),
                                   Poisson(); data = d)
    # an NB2 q = 3 block stays refused (drmTMB refuses it too)
    dn = _ngc_data("nb2")
    mu3 = @formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 | g))
    @test_throws Exception drm(bf(mu3, @formula(sigma ~ 1), @formula(sd(g, coef(intercept)) ~ z)),
                               NegBinomial2(); data = dn)
end

@testset "LogNormal (1 + x1 | g): SD formulas on one or both coefficients" begin
    # drmTMB 0.8.1 logLik: one/one -550.8682164308, z/z -549.4290610144,
    # first -549.9083396264, last -550.4577955853. Julia agrees to 1e-9.
    d = _ngc_data("lognormal")
    cells = ((:both_one, (_NGC_SD[:one0], _NGC_SD[:one1]), -550.8682164308),
             (:both_z, (_NGC_SD[:z0], _NGC_SD[:z1]), -549.4290610144),
             (:first, (_NGC_SD[:first],), -549.9083396264),
             (:last, (_NGC_SD[:last],), -550.4577955853))
    for (name, sds, ref) in cells
        fit = drm(bf(_NGC_MU, @formula(sigma ~ 1), sds...), LogNormal(); data = d)
        @test fit.converged
        @test loglikelihood(fit) ≈ ref atol = 1e-4
        # exact identity with the Gaussian fit of log(y): logLik_LogNormal = logLik_Gaussian - sum(log y)
        dg = merge(d, (; y = log.(d.y)))
        fg = drm(bf(_NGC_MU, @formula(sigma ~ 1), sds...), Gaussian(); data = dg)
        @test loglikelihood(fit) ≈ loglikelihood(fg) - sum(log.(d.y)) atol = 1e-8
        @test fit.theta ≈ fg.theta atol = 1e-6
    end
    # q = 3 stays refused: drmTMB refuses the LogNormal q = 3, 4 blocks
    mu3 = @formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 | g))
    @test_throws Exception drm(bf(mu3, @formula(sigma ~ 1), _NGC_SD[:first]), LogNormal(); data = d)
    # fixed `sigma ~ 1` only
    @test_throws ArgumentError drm(bf(_NGC_MU, @formula(sigma ~ x1), _NGC_SD[:first]), LogNormal(); data = d)
end

@testset "Binomial (1 + x1 + x2 [+ x3] | g) with no SD formula" begin
    # drmTMB 0.8.1: q = 3 logLik -253.5747241067 (converged, pdHess TRUE); q = 4 -253.3593431487
    # (is_converged() FALSE, pdHess FALSE, so no agreement claim at q = 4).
    # R integrates by Laplace, Julia by adaptive Gauss-Hermite (5 nodes per dimension by default).
    # Measured on this file (Julia 1.10.12): q = 3 with `nq = 1` (the Laplace rule) gives
    # -253.57472, equal to R's value to 1e-5, so the 0.57 gap is the integrator; the default
    # gives -253.00244 and nq = 9 gives -253.00057. q = 4: default -252.71318 (converged).
    d = _ngc_read(joinpath(@__DIR__, "fixtures", "binomial_indep_restarts", "g12_binomial.csv"))
    mu3 = bf(@formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 | g)))
    fit3 = drm(mu3, Binomial(:logit); data = d)
    # (`fit3.converged` is true on Julia 1.10.12 and false on 1.12.6: the optimum has a flat
    # direction, rcond about 1e-11, so the flag is not asserted.)
    @test loglikelihood(fit3) ≈ -253.00244 atol = 1e-3
    @test loglikelihood(fit3) > -253.5747241067              # at or above drmTMB's Laplace value
    @test length(fit3.theta) == 4 + 3 + 3                    # 4 fixed + 3 SDs + 3 correlation parameters
    # the explicit `~ 1` SD formulas are the same model
    fit3b = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 | g)),
                   @formula(sd(g, coef(intercept)) ~ 1), @formula(sd(g, coef(x1)) ~ 1),
                   @formula(sd(g, coef(x2)) ~ 1)), Binomial(:logit); data = d)
    @test loglikelihood(fit3) ≈ loglikelihood(fit3b) atol = 1e-4   # observed 1e-5 on 1.12.6, <= 1e-6 on 1.10.12
    fit4 = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g))), Binomial(:logit); data = d)
    @test fit4.converged
    @test loglikelihood(fit4) ≈ -252.71318 atol = 1e-3
    # other links keep the old refusal
    @test_throws Exception drm(mu3, Binomial(:probit); data = d)
end
