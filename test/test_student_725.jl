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
