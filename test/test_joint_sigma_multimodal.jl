# Stress case 6a (parity evidence README): Gaussian `y ~ x + (1 | id)`,
# `sigma ~ 1 + (1 | site)`, both with SD formulas, 200 x 200 crossed groups,
# n = 2400, seed 20261110. Diagnosis: the joint density in (mean effect,
# log-sigma effect) is not log-concave, and at this design the inner problem has
# several positive-definite modes (e.g. one where a single site's log-sigma effect
# collapses to about -5.6 SD); the Laplace value then depends on the warm start
# and the outer objective is discontinuous. Native drmTMB 0.8.1 reports false
# convergence (gradient ~3e8) on the same data. The fit must say so: a warning
# naming the non-unique mode, and `converged == false`. ~40 s.
using DRModels, Random, Test

function _jsm_fixture(; seed = 20261110, n_id = 200, n_site = 200, n_each = 12)
    rng = MersenneTwister(seed)
    n = n_id * n_each
    gi = [(r - 1) ÷ n_each + 1 for r in 1:n]
    gs = [(r - 1) % n_site + 1 for r in 1:n]
    w_id = collect(range(-1, 1; length = n_id)); v_site = collect(range(-1, 1; length = n_site))
    a = exp.(log(0.6) .+ 0.4 .* w_id) .* randn(rng, n_id)
    b = exp.(log(0.35) .+ 0.6 .* v_site) .* randn(rng, n_site)
    x = randn(rng, n)
    z = ones(n)
    y = 0.2 .+ 0.5 .* x .+ a[gi] .+ exp.(log(0.5) .+ b[gs] .* z) .* randn(rng, n)
    return (; y, x, id = string.("id", gi), site = string.("s", gs), w = w_id[gi], v = v_site[gs])
end

@testset "joint sigma: non-unique Laplace mode is reported, not hidden (stress case 6a)" begin
    d = _jsm_fixture()
    fm = bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | site)),
            @formula(sd(id) ~ w), @formula(sd(site, sigma) ~ v))
    fit = @test_logs (:warn, r"mode is not unique") match_mode = :any drm(fm, Gaussian(); data = d)
    @test !fit.converged
end
