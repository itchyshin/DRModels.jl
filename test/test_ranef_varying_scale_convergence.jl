# test_ranef_varying_scale_convergence.jl -- DRModels.jl #609 item 2 (varying-scale
# conditional cell, `bf(y ~ x + (1 | g), sigma ~ x)`, routed through
# `_fit_ranef_gaussian` in src/gaussian_ranef.jl).
#
# WHAT #609 LEFT OPEN. The issue's own diagnosis established that DRModels.jl reaches
# its optimum on this cell (sweeping `g_tol` from 1e-8 to 1e-16 moves the
# coefficients by 1.3e-11) and handed the ~1e-5 parity gap to the drmTMB lane. It
# never looked at the flag DRModels.jl reports alongside that optimum, and that flag is
# where this route does have a defect.
#
# THE DEFECT. `Optim.converged(res)` is the OR of the x, f and g criteria
# (Optim/src/api.jl:117). `Optim.Options(g_tol = g_tol)` leaves `f_reltol`,
# `f_abstol`, `x_reltol` and `x_abstol` at their 0.0 defaults, so `f_converged`
# fires whenever two successive objective values are byte-identical -- which is
# what "flat" means numerically, and which a varying-scale Gaussian surface
# reaches while running away up the unbounded sigma_i -> 0 ridge, nowhere near a
# stationary point. Measured on this route 2026-09-05 over a 6,400-cell grid
# (n in 30..400, G in 3..20, sigma slope in 0.15..8.0): 1,042 fits returned
# `converged = true` with the GRADIENT criterion false; among those the true
# gradient infinity-norm was <= 1e-6 for 840, <= 1e-3 for 107, <= 1 for 50, and
# ABOVE 1 for 45.
#
# WHY NOT A RESTART. Re-running LBFGS from the stalled point was measured on the
# same 1,042 cases: it recovered 665, left 377 still not gradient-converged, made
# the objective WORSE in 143, and produced NaN. Rejected. The change is to the
# reported condition only -- theta-hat, the ML/REML objective and logLik are
# untouched.
#
# WHY `Optim.g_converged` IS THE RIGHT CRITERION. Over the same grid,
# `Optim.g_residual(res)` equalled `maximum(abs, ForwardDiff.gradient(nll, theta))`
# EXACTLY (max absolute difference 0.0 over 5,349 gradient-converged fits), and
# the largest recomputed norm among them was 9.996e-9 -- inside `g_tol = 1e-8`.
#
# WHY THIS FILE PINS A PANEL AND NOT A NAMED STALLING SEED (2026-09-06).
# The first version of this file pinned one draw -- `StableRNG(263)`, n = 30,
# G = 3, sigma slope 8.0 -- measured on macOS/aarch64 to stall with
# ||g||_inf = 3.72e137. On the Linux/x86-64 CI runners that SAME draw is
# gradient-converged: shard 3/4 of both Julia legs measured ||g||_inf =
# 7.450580596923828e-9, i.e. INSIDE `g_tol`, so `@test !neg.converged` was red
# on CI and green locally. Which individual draw stalls is a property of the
# LBFGS/HagerZhang path through a chaotic runaway region and therefore of the
# BLAS and the architecture; it is not a property of the fix. So the RED control
# below is a PANEL over that region -- the contract is asserted on every fit the
# panel returns, and the panel is required to actually reach the defect (at
# least one stalled fit) and to still report `true` somewhere (so the fix cannot
# be satisfied by reporting `false` wholesale). Per-fit numbers are printed, so
# a future platform disagreement is legible from the CI log alone.
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_ranef_varying_scale_convergence.jl")'

module TestRanefVaryingScaleConvergence

using DRModels
using Test
using LinearAlgebra
using ForwardDiff
using StableRNGs

const GTOL = 1e-8          # the `drm(...)` default g_tol (src/gaussian_core.jl:436)

# Gaussian varying-scale random-intercept draw. StableRNGs, per test/README.md's
# preference order, so the DATA is identical on every Julia minor version and on
# every platform (the optimiser's path through it is not -- see the note above).
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

_fit(dat) = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ x)), Gaussian(); data = dat)
_gradinf(fit) = maximum(abs, ForwardDiff.gradient(fit.nll, fit.theta))

# The RUNAWAY region: sigma slope 10 over n = 40 in G = 4 groups puts the
# steepest rows many orders of magnitude apart. Measured on macOS/aarch64
# 2026-09-06 over seeds 1:20 -- 3 fits gradient-converged, 14 stalled (worst
# ||g||_inf = 4.23e124 with a POSITIVE Gaussian loglik of +1929.55), 3 threw.
#
# #746/#747 (2026-09-27): that "unbounded sigma_i -> 0 ridge" was NOT a property
# of the likelihood. It was catastrophic cancellation in the Woodbury quadratic
# q1 - q2 (both terms ~1/D_min, their rounding error far above the true value),
# which let the computed nll go to -1e124 and LBFGS chase the rounding hole. With
# the cancellation-free form (`_re_quad_stable`) the same panel measures 17 fits
# gradient-converged (all ||g||_inf <= 4.2e-9), 0 stalled, 3 threw (a boundary
# sd -> 0 local optimum whose Hessian overflows -- a separate, pre-existing
# vcov-guard outcome); every converged logLik matches drmTMB 0.7.1 (TMB Laplace,
# exact for this Gaussian model) to <= 1e-9 where both reach the same optimum.
const RUNAWAY = (n = 40, G = 4, sigma_slope = 10.0, sd_b = 0.8)
# The WELL-CONDITIONED region: the shape of #609's own 144-row fixture.
const CLEAN = (n = 144, G = 12, sigma_slope = 0.15, sd_b = 0.8)

@testset "#609: varying-scale ranef `converged` is the gradient criterion" begin

    # --- the RED control: a panel over the runaway region ------------------
    # `converged` must never be reported at a point above `g_tol`. Pre-fix
    # (origin/main), the stalled fits below all returned `converged = true`, so
    # `nviol` was NON-ZERO and this assertion was the failing one.
    #
    # A draw in this region can also THROW (the unbounded ridge reaches an
    # Inf/NaN Hessian, which `src/vcov_guard.jl` refuses). That is a different,
    # already-guarded outcome and not what this file is about, so such draws are
    # counted and printed rather than asserted on -- `nstall` and `nconv` below
    # keep the panel from being hollowed out by them.
    nrun = 0; nconv = 0; nstall = 0; nviol = 0; nerr = 0
    worst_stall = 0.0
    for seed in 1:20
        local fit, gi
        try
            fit = _fit(_draw(seed; RUNAWAY...))
            gi = _gradinf(fit)
        catch err
            nerr += 1
            println("  runaway seed $seed: threw $(typeof(err))")
            continue
        end
        nrun += 1
        println("  runaway seed $seed: converged=$(fit.converged) ",
                "||g||_inf=$(gi) loglik=$(fit.loglik)")
        if fit.converged
            nconv += 1
            gi <= GTOL || (nviol += 1)
        else
            gi > GTOL && (nstall += 1; worst_stall = max(worst_stall, gi))
        end
    end
    println("  runaway panel: $nrun fits, $nconv converged, $nstall stalled ",
            "(worst ||g||_inf $(worst_stall)), $nviol violations, $nerr threw")

    # THE CONTRACT. Red on origin/main before #609 item 2.
    @test nviol == 0
    # The former fixture guard `nstall >= 1` asserted that this region STALLS.
    # After #746/#747 it no longer does -- the stall was the cancellation
    # artifact described above -- so the panel now guards the opposite: the
    # region must be fitted, not merely flagged. (The #609 flag contract itself
    # stays exercised: `nviol == 0` above, and the boundary cases below.)
    # 17 measured on macOS/aarch64; margin for platform-dependent boundary seeds.
    @test nconv >= 15

    # --- the OTHER side of the boundary, at the boundary -------------------
    # A well-behaved draw that genuinely meets the gradient criterion and sits
    # close to `g_tol`: ||g||_inf = 7.409475699660106e-9 on macOS/aarch64, i.e.
    # 74% of `g_tol`. A change that merely reported `false` more often fails here.
    pos = _fit(_draw(5; CLEAN...))
    gpos = _gradinf(pos)
    println("  clean seed 5: converged=$(pos.converged) ||g||_inf=$(gpos)")
    @test gpos <= GTOL
    @test pos.converged

    # --- and over the whole well-conditioned panel -------------------------
    # Every draw in the benign region must still be reported converged, and be
    # genuinely gradient-converged. 8/8 on macOS/aarch64 2026-09-06.
    ncconv = 0; ncviol = 0
    for seed in 1:8
        fit = _fit(_draw(seed; CLEAN...))
        gi = _gradinf(fit)
        println("  clean seed $seed: converged=$(fit.converged) ||g||_inf=$(gi)")
        fit.converged && (ncconv += 1)
        gi <= GTOL || (ncviol += 1)
    end
    @test ncconv == 8
    @test ncviol == 0
end

end # module
