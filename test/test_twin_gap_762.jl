# test_twin_gap_762.jl -- DRModels.jl #762 / #707: Gaussian correlated random
# intercept + slope `(1 + x | g)` (`_fit_correlated_ranef_gaussian`,
# src/gaussian_ranef.jl).
#
# THE DEFECTS. #762: `DomainError` (log of a negative number) on every fit with an
# uncentred covariate and on ~35% of refits under H0: no slope variance. #707:
# `AssertionError` in LBFGS's HagerZhang line search on a sleepstudy-style cell
# whose MLE sits at the rho = 1 boundary. drmTMB fits all of them.
#
# ROOT CAUSE. Not the covariance parametrisation (log-Cholesky is PD by
# construction). The capacitance determinant was formed as m11*m22 - m21^2, and
# the quadratic as r'D^-1 r - c'M^-1 c: both are differences of large, nearly
# equal numbers when the covariate is uncentred (b22 ~ xbar^2 b11) or rho -> +-1
# (Sigma^-1 entries ~1e20). The determinant came out NEGATIVE (-3.5e41 on #707's
# cell) and `log` threw; or an Inf/NaN reached the line search. Fix:
# `_corr_re_stable` evaluates both in whitened coordinates as sums of
# non-negative terms (det A = 1 + tr P + det P, det P from a centred weighted
# sum of squares; penalised RSS at the refined conditional mode). The optimiser
# also works on (1, x - xbar) and a QR-preconditioned mean design, then maps back
# to the reported Cholesky factor exactly, and restarts once when it stops on the
# rho = +-1 edge (a stationary limit of this parametrisation).
#
# MEASURED (macOS/aarch64, Julia 1.10, 2026-09-27) on the H0 panel in
# twin_gap_762_dgm.jl (60 seeds, centred and uncentred x = 120 fits):
#   origin/main  95/120 threw (56 centred + 39 uncentred DomainError, 2 AssertionError),
#                13 non-converged, 10 OK.
#   this fix     120/120 converged; logLik is the best of {drmTMB centred, drmTMB
#                uncentred, DRModels.jl centred, DRModels.jl uncentred} on every
#                seed (max shortfall 4.9e-12); DRModels.jl centred == uncentred to 2.3e-13.
# drmTMB 0.7.1 reference values below are generated OUTPUTS on the same data; both
# sides evaluate the exact Gaussian marginal (TMB Laplace is exact here).
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_twin_gap_762.jl")'

module TestTwinGap762

using DRModels
using Test
using LinearAlgebra
using Random
using Logging

include(joinpath(@__DIR__, "twin_gap_762_dgm.jl"))

# Dense reference: V = D + Z Sigma Z', Z_k rows (1, x_i); returns (logdet V, r'V^-1 r)
# minus the log(D) part, i.e. matching `_corr_re_stable`'s (logdetA, quad).
function _dense_corr(r, invD, xs, gidx, G, L)
    n = length(r)
    Z = zeros(eltype(r), n, 2G)
    for i in 1:n
        k = gidx[i]; Z[i, 2k - 1] = 1; Z[i, 2k] = xs[i]
    end
    Σ = L * L'
    V = Diagonal(1 ./ invD) + Z * kron(Matrix{eltype(r)}(I, G, G), Σ) * Z'
    F = cholesky(Symmetric(V))
    return logdet(F) - sum(log.(1 ./ invD)), dot(r, F \ r)
end

@testset "#762/#707: stable correlated-RE determinant and quadratic" begin
    rng = MersenneTwister(762)
    G = 6; gidx = repeat(1:G, inner = 5); n = length(gidx)
    r = randn(rng, n); invD = exp.(0.3 .* randn(rng, n))
    # (a) moderate: centred covariate, well-conditioned Sigma
    xs = randn(rng, n)
    l11, l22, cc = 0.8, 0.4, 0.2
    ld, q, _, _ = DRModels._corr_re_stable(r, invD, xs, gidx, G, l11, l22, cc)
    ldr, qr_ = _dense_corr(r, invD, xs, gidx, G, [l11 0; cc l22])
    @test ld ≈ ldr rtol = 1e-12
    @test q ≈ qr_ rtol = 1e-12
    # (b) the #762/#707 geometry: covariate far from 0 and rho -> 1 (l22 tiny).
    xb = 60 .+ 2 .* randn(rng, n)
    l11, l22, cc = 20.0, 1e-9, -0.33
    ld, q, _, _ = DRModels._corr_re_stable(r, invD, xb, gidx, G, l11, l22, cc)
    ldb, qb = _dense_corr(big.(r), big.(invD), big.(xb), gidx, G, big.([l11 0; cc l22]))
    @test isfinite(ld) && isfinite(q) && q >= 0
    @test ld ≈ Float64(ldb) rtol = 1e-8
    @test q ≈ Float64(qb) rtol = 1e-8
end

# #707's cell 10 (R set.seed(20270905); 12 groups x 5, x = 0:4), drmTMB 0.7.1
# logLik -82.5232736670 with cor((Intercept), x | g) = 0.999999 (the boundary).
const Y707 = [0.68302196867714005, 0.31324978226716099, 1.96598426632917, 1.16434356964804,
    1.50666986602188, -0.22722797706810899, -1.11196254033271, -1.7536264978370399,
    -1.80677308714506, -1.2799280806211999, 1.6832575519662201, -1.3132073563064399,
    -0.14362124237451099, 0.58220202958977596, 1.5662463664483699, 0.43489762964321399,
    2.0221535859477102, 1.46766640227346, 1.4439661078749999, 2.5193052041007502,
    0.044064457296570998, -0.35389181223571198, 1.3247118705743099, 0.89846452310496105,
    1.1471576211313399, 0.099118149213429693, 1.9800480180023601, 2.4192709648969899,
    1.91911181683676, 1.06342844019426, -0.68670003607531105, 0.228355040730397,
    0.57821485555791496, 0.110878336191311, 1.5630999312190701, -1.3821586372324499,
    -1.7895208970480401, -1.70198298033356, -2.31484651754571, -2.6763463751138099,
    1.5974599434339201, 2.1980864038905201, 1.9528709916201701, 4.2189909597903696,
    2.62883804478636, 0.91350635552375004, 1.8753900206515299, -0.070902681725294694,
    1.8442160541377099, 1.8590203394390299, 0.75603963632373805, -0.85882269108606402,
    -0.055893035629247197, 1.10510340587748, 1.03496055520983, 0.72782623956292403,
    0.482651361590961, 0.92190362696647599, -0.80239367519673399, 1.6293900402539401]

_fitc(d; kw...) = with_logger(NullLogger()) do
    drm(bf(@formula(y ~ x + (1 + x | g))), Gaussian(); data = d, kw...)
end

@testset "#707: rho = 1 boundary cell fits instead of throwing" begin
    d = (y = Y707, x = Float64.(repeat(0:4, 12)), g = repeat(1:12, inner = 5))
    for shift in (0.0, 100.0)          # the uncentred twin must give the same fit
        fit = _fitc(merge(d, (x = d.x .+ shift,)))
        @test fit.converged
        @test loglik(fit) >= -82.5232736670 - 1e-6
        @test loglik(fit) ≈ -82.5232736670 atol = 1e-4
        @test coef(fit, :mu)[2] ≈ 0.1710684 atol = 1e-5
        @test coef(fit, :sigma)[1] ≈ -0.3006873 atol = 1e-5
    end
end

# (shape, seed, drmTMB logLik). On origin/main: centred 1,3,4,7,9,16 and
# uncentred 3,16 threw DomainError; uncentred 4,7,9 were non-converged.
const REF = [
    (:centred, 1, -313.4668914085), (:centred, 3, -303.9234834366),
    (:centred, 4, -341.6822454388), (:centred, 7, -297.9592680737),
    (:centred, 9, -360.8360703255), (:centred, 16, -330.3581223769),
    (:uncentred, 1, -313.4670464496), (:uncentred, 3, -303.9234835895),
    (:uncentred, 4, -341.6822454321), (:uncentred, 7, -297.9592680737),
    (:uncentred, 9, -360.8360728613), (:uncentred, 16, -330.3581223769),
]

@testset "#762: H0 (no slope variance) refits return a fit, twinning drmTMB" begin
    llc = Dict{Int,Float64}()
    for (kind, seed, ll_r) in REF
        fit = _fitc(_tg762_draw(seed; centred = kind === :centred))
        @testset "$kind seed $seed" begin
            @test fit.converged
            @test isfinite(loglik(fit))
            # at least as good as drmTMB, and not wildly better (same model)
            @test loglik(fit) >= ll_r - 1e-6
            @test loglik(fit) - ll_r < 1e-2
        end
        kind === :centred ? (llc[seed] = loglik(fit)) :
            (@test loglik(fit) ≈ llc[seed] atol = 1e-8)   # shift invariance
    end
end

end # module
