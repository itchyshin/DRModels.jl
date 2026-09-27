# #725 (twin: drmTMB #1266) — Student() with crossed random intercepts on the mean,
# `(1 | g) + (1 | h)`, fitted by the Laplace approximation as drmTMB `student()`
# does. Reference numbers were produced by drmTMB 0.7.1 on exactly this simulated
# data set (generated below with StableRNG), via
#   drmTMB(bf(y ~ x + (1 | g) + (1 | h), sigma ~ 1, nu ~ 1), family = student(), data)
using DRModels
using Test, StableRNGs
using Distributions: TDist

function _student_crossed_725_data()
    rng = StableRNG(725)
    n, G, H = 600, 30, 25
    g = rand(rng, 1:G, n); h = rand(rng, 1:H, n)
    bg = 0.4 .* randn(rng, G); bh = 0.3 .* randn(rng, H)
    x = randn(rng, n)
    y = 1.0 .+ 0.5 .* x .+ bg[g] .+ bh[h] .+ 0.5 .* rand(rng, TDist(5.0), n)
    return (; y, x, g, h)
end

# drmTMB 0.7.1 reference on `_student_crossed_725_data()`.
const REF_725 = (loglik = -617.249843983, mu = [1.057484277763, 0.540615226747],
                 logsigma = -0.672254288932, nu_eta = 1.341727959559,
                 sd_g = 0.415773684818, sd_h = 0.336005307220,
                 se_fixed = [0.1045376550590, 0.0249291013525, 0.0537592322036, 0.3798568246409])
# Same data, `sigma ~ x`.
const REF_725_SIGX = (loglik = -616.631473925, sigma = [-0.6742953407236, -0.0408332476583],
                      sd = [0.416963549849, 0.335111253050])

@testset "#725 Student crossed (1|g)+(1|h) — admitted and matches drmTMB" begin
    data = _student_crossed_725_data()
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1), @formula(nu ~ 1)),
              Student(); data = data)
    @test fit.converged
    @test isfinite(loglik(fit))
    @test loglik(fit) ≈ REF_725.loglik atol = 1e-6
    @test coef(fit, :mu) ≈ REF_725.mu atol = 1e-5
    @test coef(fit, :sigma)[1] ≈ REF_725.logsigma atol = 1e-5
    @test coef(fit, :nu)[1] ≈ REF_725.nu_eta atol = 1e-4
    @test exp.(coef(fit, :resd)) ≈ [REF_725.sd_g, REF_725.sd_h] atol = 1e-5
    se = stderror(fit)
    @test all(isfinite, se)
    @test se[1:4] ≈ REF_725.se_fixed rtol = 1e-4
end

@testset "#725 Student crossed with a sigma formula — matches drmTMB" begin
    data = _student_crossed_725_data()
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ x), @formula(nu ~ 1)),
              Student(); data = data)
    @test fit.converged
    @test loglik(fit) ≈ REF_725_SIGX.loglik atol = 1e-6
    @test coef(fit, :sigma) ≈ REF_725_SIGX.sigma atol = 1e-5
    @test exp.(coef(fit, :resd)) ≈ REF_725_SIGX.sd atol = 1e-5
end

@testset "#725 Student crossed — still refuses unsupported RE shapes" begin
    data = _student_crossed_725_data()
    @test_throws ErrorException drm(bf(@formula(y ~ x + (1 + x | g) + (1 | h))), Student(); data = data)
end

# --- Post-review must-fix regressions (v2-review-numerics.md, PR #827) ---------------
# Same generator Noether's adversarial review used (StableRNG(3), G=15, H=10, n=300),
# reproduced here rather than read from scratch files so the tests are self-contained.
# Reference numbers are drmTMB 0.7.1 on the identical simulated data.
function _student_crossed_827_data(; G = 15, H = 10, n = 300, sdg = 0.7, sdh = 0.5, ν = 3.0, outl = 0)
    rng = StableRNG(3)
    g = rand(rng, 1:G, n); h = rand(rng, 1:H, n); x = randn(rng, n)
    ug = sdg .* randn(rng, G); uh = sdh .* randn(rng, H)
    e = rand(rng, TDist(ν), n) .* 0.6
    y = 1 .+ 0.4 .* x .+ ug[g] .+ uh[h] .+ e
    for i in 1:outl
        y[i] += 25 * (isodd(i) ? 1 : -1)
    end
    return (; y, x, g, h)
end

@testset "#827 must-fix 1 — extreme-probe fallback fails closed, never throws" begin
    # 6 outliers at ±25 with ν = 2.5: before the fix, `inner_mode`'s expected-information
    # fallback cholesky used `check = true` and a line-search probe with an extreme
    # σ/ν produced a non-finite expected-information matrix, throwing `PosDefException`
    # straight out of `drm` (reviewer: "throws PosDefException"). drmTMB fits this at
    # −487.8547216761 with one crossed sd driven to a boundary (log sd ≈ −22.4), which
    # also requires must-fix 2 (the widened wall) to be reachable at all.
    data = _student_crossed_827_data(; ν = 2.5, outl = 6)
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1), @formula(nu ~ 1)),
              Student(); data = data)
    @test fit isa DrmFit
    @test fit.converged
    @test loglik(fit) ≈ -487.8547216761 atol = 1e-4
end

@testset "#827 must-fix 2 — zero crossed variance reaches drmTMB's boundary MLE" begin
    # sd_h = 0 in the data-generating process; drmTMB's MLE puts log sd_h at −12.57,
    # just outside the old |log σ| > 12 wall, so the old code stopped early at
    # log sd_h ≈ −3.48 with converged = false. The widened (and overflow-guarded)
    # bound must let the optimizer reach the true boundary.
    data = _student_crossed_827_data(; sdh = 0.0)
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1), @formula(nu ~ 1)),
              Student(); data = data)
    @test fit.converged
    @test loglik(fit) ≈ -394.2181966664 atol = 1e-6
    @test coef(fit, :resd)[2] < -10   # log sd_h at (or past) the drmTMB boundary of −12.57
end

@testset "#827 must-fix 3 — ordinary data reaches drmTMB's optimum (no premature stop)" begin
    # ν = 1e6 (effectively Gaussian), otherwise unremarkable data. The gradient is smooth
    # and correct throughout, but the first LBFGS run stopped non-converged after its line
    # search crossed the fail-closed region; a single restart from θ̂ reached drmTMB's
    # optimum to 1e-10 (reviewer). The fitter must now do that restart itself.
    data = _student_crossed_827_data(; ν = 1e6)
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1), @formula(nu ~ 1)),
              Student(); data = data)
    @test fit.converged
    @test loglik(fit) ≈ -296.2105660997 atol = 1e-6
end
