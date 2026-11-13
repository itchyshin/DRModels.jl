# LogNormal `(1 + x1 | g)`: the default HagerZhang line search used to throw an AssertionError out of
# `drm()` on data whose slope SD (or correlation) sits at a boundary. The fit now retries with a finite
# sentinel and BackTracking, and polishes from that point (`_fit_lognormal_corr_ranef`).
#
# Why it threw: as the log-Cholesky coordinate log(L22) runs below about -20, the per-group Hessian of
# the quadrature (entries near 1e18) loses positive definiteness; the objective first turns spuriously
# low and then +Inf, and `secant2!` asserts that it is finite. Which probes reach that region depends on
# last-bit rounding, so it differs between Julia versions and CPUs.
#
# Data (fixtures/lognormal_slope_linesearch/data.csv, 120 rows each, 12 groups of 10), the throwing
# versions on Totoro (AMD znver5) before the fix:
#   s2   1.10.12  `isfinite(phi_c) && isfinite(dphi_c)`; fits on 1.12.7 and 1.13.3
#   s10  1.10.12 and 1.12.7  `B > A`; fits on 1.13.3
#   b1   1.13.3  `B > A` (generated with no random slope)
#   b10  1.10.12 `B > A` (generated with no random slope)
# No single dataset throws on every version, so the test uses all four: on each of 1.10.12 and 1.13.3
# at least one of them failed before the fix. Julia 1.12.7: s10 fails, the others fit.
#
# Agreement with drmTMB 0.8.1 (reference r_reference.csv, made by gen_ref.R) under the lane rule:
# logLik 1e-4, fixed effects 1e-3 R standard errors, log SDs 5e-3. R's correlation sits at +-1 on all
# four datasets, so the correlation is not compared.
using DRModels, Test, DelimitedFiles, Logging

const _LNLS = joinpath(@__DIR__, "fixtures", "lognormal_slope_linesearch")

function _lnls_data(ds)
    raw, _ = readdlm(joinpath(_LNLS, "data.csv"), ','; header = true)
    rows = String.(raw[:, 1]) .== ds
    return (; y = Float64.(raw[rows, 2]), x1 = Float64.(raw[rows, 3]), g = String.(raw[rows, 4]))
end

function _lnls_ref(ds)
    raw, _ = readdlm(joinpath(_LNLS, "r_reference.csv"), ','; header = true)
    rows = String.(raw[:, 4]) .== ds
    term = String.(raw[rows, 1]); est = Float64.(raw[rows, 2]); se = Float64.(raw[rows, 3])
    get(t) = (i = findfirst(==(t), term); (est[i], se[i]))
    return (; ll = Float64(raw[findfirst(rows), 5]),
            mu = [get("mu:(Intercept)"), get("mu:x1")], logsigma = get("sigma:(Intercept)")[1],
            sd0 = get("sd:mu:(1 + x1 | g):(Intercept)")[1], sd1 = get("sd:mu:(1 + x1 | g):x1")[1])
end

@testset "LogNormal (1 + x1 | g): no line-search AssertionError; agrees with drmTMB" begin
    f = bf(@formula(y ~ x1 + (1 + x1 | g)), @formula(sigma ~ 1))
    for ds in ("s2", "s10", "b1", "b10")
        data = _lnls_data(ds); R = _lnls_ref(ds)
        @test length(data.y) == 120
        fit = Logging.with_logger(Logging.NullLogger()) do
            try
                drm(f, LogNormal(); data)
            catch e
                e
            end
        end
        @test !(fit isa AssertionError)
        @test fit isa DRModels.DrmFit
        if fit isa DRModels.DrmFit   # a failed fit is reported by the line above, not rethrown here
            @test abs(loglikelihood(fit) - R.ll) <= 1e-4
            @test maximum(abs.(coef(fit, :mu) .- first.(R.mu)) ./ last.(R.mu)) <= 1e-3
            @test abs(coef(fit, :sigma)[1] - R.logsigma) <= 5e-3
            a, b, cc = coef(fit, :recov)             # log L11, log L22, L21 (L = [L11 0; L21 L22])
            sd0 = exp(a); sd1 = sqrt(cc^2 + exp(2b))
            @test abs(log(sd0) - log(R.sd0)) <= 5e-3
            @test abs(log(sd1) - log(R.sd1)) <= 5e-3
        end
    end
end
