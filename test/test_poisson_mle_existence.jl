# Poisson / NB2 MLE-existence screen (#1109; drmTMB twin #1553).
# Policy: detect and warn -- the fit is returned, SEs of flagged mu
# coefficients are Inf. The optimiser flag `fit.converged` is unchanged;
# `convergence_status` reads the fit as :degenerate, so `is_converged` is
# false (FX3b, the shared twin contract of drmTMB #1554).
using DRModels
using Test, Logging, Random
import Distributions

# sweep-oct16 t4 pois_n8_lowmean r03 (all zero) and r27 (positives at max x)
const _POIS_X0 = [
    -0.0292785730789077, -2.07527755306188, -1.26224739973712, -0.135456173606004,
    -1.05584042635037, 0.218711141103825, 0.915790979674801, 0.0391185640938337,
]
const _POIS_Y0 = zeros(8)
const _POIS_X1 = [
    0.874179376754336, -1.80802294417958, 1.07582680175277, 1.99152036375555,
    -0.34920696964089, -0.0158223689737766, 0.928431408635883, 1.09143738410038,
]
const _POIS_Y1 = [0.0, 0, 0, 2, 0, 0, 0, 0]

function _pois_capture(f)
    logger = Test.TestLogger(min_level = Logging.Warn)
    fit = with_logger(f, logger)
    recs = filter(r -> haskey(r.kwargs, :separation), logger.logs)
    return fit, recs
end

@testset "count MLE existence cone" begin
    X = [ones(6) [-3.0, -2, -1, 1, 2, 3]]
    y0 = zeros(6)
    det0 = DRModels._detect_separation(X, Float64.(y0 .> 0), 1 .+ Float64.(y0 .> 0))
    @test det0.separated
    @test det0.conclusive

    y1 = [0.0, 0, 0, 0, 0, 2]
    det1 = DRModels._detect_separation(X, Float64.(y1 .> 0), 1 .+ Float64.(y1 .> 0))
    @test det1.separated

    yok = [0.0, 1, 0, 2, 1, 3]
    detok = DRModels._detect_separation(X, Float64.(yok .> 0), 1 .+ Float64.(yok .> 0))
    @test !detok.separated
end

@testset "all-zero Poisson warns and reports Inf SE (#1109 r03)" begin
    fit, recs = _pois_capture() do
        drm(bf(@formula(y ~ 1 + x)), Poisson(); data = (; y = _POIS_Y0, x = _POIS_X0))
    end
    @test !isempty(recs)
    @test recs[1].kwargs[:separation] === :count_mle_nonexistent
    se = stderror(fit)
    @test all(isinf, se)
    @test fit.converged                          # the optimiser flag is unchanged
    @test fit.geometry === :count_mle_nonexistent
    @test convergence_status(fit) === :degenerate   # FX3b
    @test !is_converged(fit)
end

@testset "extreme-x Poisson warns and reports Inf SE (#1109 r27)" begin
    fit, recs = _pois_capture() do
        drm(bf(@formula(y ~ 1 + x)), Poisson(); data = (; y = _POIS_Y1, x = _POIS_X1))
    end
    @test !isempty(recs)
    @test recs[1].kwargs[:separation] === :count_mle_nonexistent
    @test all(isinf, stderror(fit))
    @test convergence_status(fit) === :degenerate
end

@testset "all-zero NB2 warns and reports Inf SE (sibling of #1109)" begin
    fit, recs = _pois_capture() do
        drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), NegBinomial2();
            data = (; y = _POIS_Y0, x = _POIS_X0))
    end
    @test !isempty(recs)
    @test recs[1].kwargs[:separation] === :count_mle_nonexistent
    se_mu = stderror(fit)[1:2]
    @test all(isinf, se_mu)
    @test fit.geometry === :count_mle_nonexistent
    @test convergence_status(fit) === :degenerate
end

# The geometry verdict must not depend on whether standard errors were
# computed: with `se = false` the covariance rules are skipped, so only the
# recorded cone screen can say the MLE does not exist (drmTMB: step 2 runs
# under `drm_control(se = FALSE)` too).
@testset "se = false keeps the count MLE verdict (FX3b)" begin
    for (fam, form) in ((Poisson(), bf(@formula(y ~ 1 + x))),
                        (NegBinomial2(), bf(@formula(y ~ 1 + x), @formula(sigma ~ 1))))
        fit, recs = _pois_capture() do
            drm(form, fam; data = (; y = _POIS_Y0, x = _POIS_X0), se = false)
        end
        @test !isempty(recs)
        @test fit.uncertainty === :skipped
        @test fit.geometry === :count_mle_nonexistent
        @test convergence_status(fit) === :degenerate
        @test !is_converged(fit)
    end
end

# drmTMB reads the bridge `converged` key as its step-1 optimiser flag
# (drmTMB #1564). A fit whose MLE does not exist must not send `true` there,
# even though the raw optimiser flag `fit.converged` is unchanged.
@testset "bridge converged key requires geometry === :ok (#1129)" begin
    for fam in ("poisson", "nbinom2")
        b, recs = _pois_capture() do
            drm_bridge(; formula = fam == "poisson" ? "y ~ x" : "y ~ x; sigma ~ 1",
                       family = fam, data = (; y = _POIS_Y0, x = _POIS_X0))
        end
        @test !isempty(recs)
        @test b["converged"] === false
        @test b["convergence_status"] == "degenerate"
    end
end

@testset "ordinary Poisson is not flagged" begin
    Random.seed!(20261016)
    x = randn(40)
    y = rand.(Distributions.Poisson.(exp.(0.5 .+ 0.5 .* x)))
    fit, recs = _pois_capture() do
        drm(bf(@formula(y ~ 1 + x)), Poisson(); data = (; y, x))
    end
    @test isempty(recs)
    @test all(isfinite, stderror(fit))
    @test fit.geometry === :ok
    @test convergence_status(fit) === :converged
    @test is_converged(fit)
end
