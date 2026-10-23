# test_twin_gap_747_starts.jl -- follow-up to #746/#747 (draft PR #835).
#
# THE GAP #835 LEFT OPEN. `_re_quad_stable` (#746/#747) made the Gaussian
# `(1 | g)` + `sigma ~ x` objective cancellation-free, but the historical START
# (`sigma` coefficients fixed at 0 except a residual-SD intercept; the RE
# log-SD fixed at a constant fraction of the MARGINAL residual SD) still left
# LBFGS unable to find the true optimum on two seeds of the RUNAWAY panel
# (test/test_ranef_varying_scale_convergence.jl, `sigma_slope = 10`, n = 40,
# G = 4):
#
#   seed 4:  DRModels landed on a boundary sd_g -> 0 local optimum (nll
#            179.80); drmTMB finds an interior optimum (nll 56.9710494815,
#            123 nats better). DRModels' OWN nll at drmTMB's parameters is
#            56.97105 -- the better optimum exists on this objective, LBFGS
#            from the blind start simply missed it.
#   seed 10: DRModels did not converge at all (mu intercept 177.5, nll
#            258.04) vs drmTMB's 163.4460683592, 95 nats worse.
#
# THE FIX (src/gaussian_ranef.jl, start/restart only -- the objective built by
# #746/#747 is untouched). Two pieces:
#
#   `_ranef_sigma_ols_start`: OLS of log(guarded residual^2) on Xσ, giving the
#   scale submodel a real slope instead of 0.
#   `_ranef_sdg_mom_start`: one-way random-effects ANOVA method-of-moments
#   estimate of the RE variance from the OLS mean residuals.
#   `_re_lbfgs_with_restart`: optimise from that data-driven start; if the
#   result is not gradient-converged, or the RE log-SD lands at a boundary
#   sd_g -> 0 local optimum, restart once from the historical (always
#   interior) start and keep whichever objective is lower. Deterministic, no
#   random multi-start (mirrors the boundary-restart pattern already used by
#   the correlated `(1 + x | g)` route, PR #837).
#
# WHY THIS FILE TESTS THE OPTIMISER DIRECTLY, NOT `drm(...)`. At both seed 4
# and seed 10 the corrected optimum still lands close enough to a variance
# boundary that `ForwardDiff.hessian(nll, θ̂)` contains a non-finite entry, so
# `_vcov_from_hessian` throws `ArgumentError` -- this is unchanged by this fix
# and is the CORRECT call (test_ranef_varying_scale_convergence.jl documents
# the same throw at these same seeds, both before and after #746/#747: "This
# behaviour (all 3 throwing) predates this PR"). So `drm(...)` itself cannot
# observe which optimum the start/restart step found; this file drives the
# same nll / start / restart machinery `_fit_ranef_gaussian` uses, one layer
# below the `DrmFit` + vcov step, exactly as test_twin_gap_747.jl's own
# `_dense_quad`/`_SC` helpers exercise `_re_quad_stable` one layer below `drm`.
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_twin_gap_747_starts.jl")'

module TestTwinGap747Starts

using DRModels
using Test
using LinearAlgebra
using DRModels: Optim   # Optim is a DRModels dependency, NOT in test/Project.toml
using ForwardDiff
using StableRNGs
using Statistics

# The RUNAWAY draw from test/test_ranef_varying_scale_convergence.jl (#609),
# duplicated here (not exported from that file's module) so this file can
# drive the optimiser directly instead of through `drm(...)`.
function _draw(seed; n, G, sigma_slope, sd_b)
    rng = StableRNG(seed)
    g = [string("g", 1 + (i % G)) for i in 0:(n - 1)]
    x = 2 .* rand(rng, n) .- 1
    levels = unique(g)
    b = Dict(l => sd_b * randn(rng) for l in levels)
    y = [0.3 + 0.6 * x[i] + b[g[i]] + exp(-0.5 + sigma_slope * x[i]) * randn(rng)
         for i in 1:n]
    return (y = y, x = x, g = g)
end

const RUNAWAY = (n = 40, G = 4, sigma_slope = 10.0, sd_b = 0.8)

# Reference values: drmTMB 0.7.1 (TMB Laplace, exact for this Gaussian model)
# fitting the SAME data, generated OUTPUTS (no drmTMB source vendored).
const REF = Dict(4 => -56.9710494815, 10 => -163.4460683592)

# Reconstruct exactly what `_fit_ranef_gaussian` builds from `bf(y ~ x + (1|g),
# sigma ~ x)` and drive its start/restart/objective machinery directly.
function _fit_theta_nll(seed)
    dat = _draw(seed; RUNAWAY...)
    n = length(dat.y)
    gidx, G = DRModels._group_index(dat.g)
    Xμ = hcat(ones(n), dat.x)
    Xσ = hcat(ones(n), dat.x)
    y = Vector{Float64}(dat.y)
    w = ones(n)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)

    # Byte-for-byte the `nll_ml` built inside `_fit_ranef_gaussian`
    # (src/gaussian_ranef.jl) -- the #746/#747 cancellation-free Woodbury nll.
    function nll_ml(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]; lσb = θ[pμ+pσ+1]
        ημ = Xμ * βμ; ησ = Xσ * βσ
        σb² = exp(2lσb)
        T = eltype(θ)
        S = zeros(T, G); C = zeros(T, G)
        rv = Vector{T}(undef, n); invDv = Vector{T}(undef, n)
        logdetD = zero(T)
        @inbounds for i in 1:n
            invD = exp(-2 * ησ[i])
            r = y[i] - ημ[i]
            rv[i] = r; invDv[i] = invD
            k = gidx[i]
            S[k] += w[i]^2 * invD
            C[k] += w[i] * r * invD
            logdetD += 2 * ησ[i]
        end
        logdetCap = zero(T)
        @inbounds for k in 1:G
            logdetCap += log(1 + σb² * S[k])
        end
        quad = DRModels._re_quad_stable(rv, invDv, w, gidx, fill(1 / σb², G), S, C)
        return 0.5 * (logdetD + logdetCap + quad) + 0.5 * n * log(2π)
    end

    βμ0 = Xμ \ y
    res0 = y - Xμ * βμ0
    θ0 = zeros(pμ + pσ + 1)
    θ0[1:pμ] .= βμ0
    θ0[pμ+1:pμ+pσ] .= DRModels._ranef_sigma_ols_start(Xσ, res0)
    θ0[pμ+pσ+1] = DRModels._ranef_sdg_mom_start(res0, gidx, G)

    θ0_restart = zeros(pμ + pσ + 1)
    θ0_restart[1:pμ] .= βμ0
    θ0_restart[pμ+1] = log(std(res0) + eps())
    θ0_restart[pμ+pσ+1] = log(std(res0) / 2 + eps())

    res = DRModels._re_lbfgs_with_restart(nll_ml, θ0, θ0_restart, 1e-8, pμ + pσ + 1, std(res0))
    θ̂ = Optim.minimizer(res)
    return θ̂, nll_ml(θ̂)
end

@testset "#747 follow-up: starts/restart reach drmTMB's optimum" begin
    for seed in (4, 10)
        θ̂, nll = _fit_theta_nll(seed)
        loglik = -nll
        @testset "seed $seed" begin
            @test isfinite(loglik)
            @test loglik ≈ REF[seed] atol = 1e-6
        end
    end
end

# --- seed 20: non-finite primary result must trigger the restart ----------
# On Linux / Julia 1.10.12 the data-driven start's LBFGS run reports converged
# with a NaN MINIMIZER on seed 20 (nll(theta_hat) is NaN while Optim.minimum is
# a stale finite value). Both restart triggers were NaN comparisons, so neither
# fired and `_vcov_from_hessian`'s `eigvals` threw "matrix contains Infs or
# NaNs". `_re_lbfgs_with_restart` now treats a non-finite result as a trigger.
# (On platforms where the primary run stays finite this passes trivially.)
@testset "#747 follow-up: seed 20 never returns a non-finite optimum" begin
    θ̂, nll = _fit_theta_nll(20)
    @test all(isfinite, θ̂)
    @test isfinite(nll)
end

# --- no-regression control over the seeds that already matched -------------
# On origin/claude/twin-gap-747 (pre-fix), the RUNAWAY panel's 3 throwing
# seeds were {1, 3, 10}: seed 1 threw an ArgumentError from `drm(...)` too
# (its own boundary sd_g -> 0 optimum, matching drmTMB, is a
# `_vcov_from_hessian` guarded throw exactly like seeds 3 and 10 -- unrelated
# to this fix) and seed 4 landed on the WRONG boundary optimum without
# throwing at all (the very defect this PR fixes). After this fix the
# throwing set is {3, 10} only: seed 1 now reaches the SAME optimum as before
# (loglik -159.8734455689509, unchanged to 1e-9) through the tie-break in
# `_re_lbfgs_with_restart` (see its docstring), and seed 4 moved into the
# converged set (tested above). So the "unaffected" set here is 1:20 minus
# {3, 4, 10} -- 4 is asserted separately above, 3 and 10 both throw exactly
# as they did pre-fix (a genuine boundary `_vcov_from_hessian` guard, not a
# regression -- see test_ranef_varying_scale_convergence.jl's header note).
@testset "#747 follow-up: previously-converged seeds are unaffected" begin
    nconv = 0; nerr = 0
    for seed in 1:20
        seed in (3, 4, 10) && continue
        dat = _draw(seed; RUNAWAY...)
        try
            fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), Gaussian(); data = dat)
            gi = maximum(abs, ForwardDiff.gradient(fit.nll, fit.theta))
            @test fit.converged
            @test gi <= 1e-8
            # seed 1 is the tightest control: pre-fix it ALSO reached this
            # exact boundary optimum (it just threw at the vcov step from a
            # more extreme representative of the same tie, see the header
            # note above and `_re_lbfgs_with_restart`'s docstring). Pin it so
            # a future change to the tie-break cannot silently drift the
            # answer while still avoiding the throw.
            seed == 1 && @test fit.loglik ≈ -159.8734455689509 atol = 1e-6
            fit.converged && (nconv += 1)
        catch err
            nerr += 1
            println("  seed $seed threw $(typeof(err))")
        end
    end
    @test nerr == 0
    @test nconv == 17
end

end # module
