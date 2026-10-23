# Bivariate Student-t (drmTMB `biv_student()` twin) — large-nu log-density
# stability, follow-up to #721/#820.
#
# `src/bivariate_student.jl` computed the p=2 and p=1 normalising constants by
# hand as `loggamma((ν+2)/2) - loggamma(ν/2)` and `loggamma((ν+1)/2) -
# loggamma(ν/2)`. Both are differences of loggamma terms of size ~ν·log ν that
# cancel catastrophically as ν grows — exactly the univariate Student bug
# #721/#820 fixed in `src/student.jl`'s `_student_logpdf_std`. Reproduced on
# branch claude/twin-gap-bivstudent: at standardized (z1, z2) = (0.5, -0.3),
# rho = 0, the naive joint term read -2.0110 at ν = 1e2 (fine) but -38.156 at
# ν = 1e16, against a bivariate-Normal / 256-bit BigFloat reference of
# -2.00788 — off by ~36 nats, the same failure mode as #721.
#
# Fix (src/bivariate_student.jl):
#   - Joint (p=2) term: `loggamma((ν+2)/2) - loggamma(ν/2)` is `Γ(ν/2+1)/Γ(ν/2)`
#     in log space, an EXACT Gamma recursion identity (Γ(x+1) = xΓ(x), x = ν/2)
#     for every ν > 0 — not an asymptotic approximation — so it collapses to
#     `log(ν/2)` with no cancellation whatsoever. Substituting the identity into
#     the surrounding constant terms collapses the whole normalising piece to
#     the ν-free constant `log(2π)`.
#   - Marginal (p=1) term: `loggamma((ν+1)/2) - loggamma(ν/2)` is NOT an
#     exact-recursion case (arguments differ by 1/2) — it is exactly the
#     univariate Student cancellation from #721/#820, so it now reuses the
#     already-verified `_student_logpdf_std` from student.jl instead of
#     re-deriving the same fix by hand.
using DRModels
using Test, Random
using SpecialFunctions: loggamma

# 256-bit BigFloat reference for the ORIGINAL (unsimplified) log-density
# formulas. BigFloat at this precision carries ~77 decimal digits, so even
# though these formulas subtract loggamma terms of size ~nu*log(nu), the
# cancellation loses at most ~36 digits at nu = 1e16 — nowhere near exhausting
# 77 digits of precision — so this is a faithful, independent ground truth,
# not a re-statement of the (fixed) Float64 code path.
_biv_t_logpdf_ref(z1, z2, rho, nu) = setprecision(BigFloat, 256) do
    z1b, z2b, rhob, nub = BigFloat(z1), BigFloat(z2), BigFloat(rho), BigFloat(nu)
    om = 1 - rhob^2
    d2 = (z1b^2 - 2 * rhob * z1b * z2b + z2b^2) / om
    Float64((loggamma((nub + 2) / 2) - loggamma(nub / 2)) - log(nub) - log(BigFloat(pi)) -
             0.5 * log(om) - ((nub + 2) / 2) * log1p(d2 / nub))
end

_uni_t_logpdf_ref(z, nu) = setprecision(BigFloat, 256) do
    zb, nub = BigFloat(z), BigFloat(nu)
    Float64((loggamma((nub + 1) / 2) - loggamma(nub / 2)) - 0.5 * log(nub) -
             0.5 * log(BigFloat(pi)) - ((nub + 1) / 2) * log1p(zb^2 / nub))
end

const NU_SWEEP = (2.5, 5.0, 10.0, 100.0, 1e3, 1e6, 1e10, 1e14, 1e16)

@testset "#bivstudent joint (p=2) term: stable across nu, matches BigFloat" begin
    for (z1, z2, rho) in ((0.5, -0.3, 0.0), (1.2, 0.4, 0.5), (-2.0, 2.0, -0.7))
        for nu in NU_SWEEP
            ref = _biv_t_logpdf_ref(z1, z2, rho, nu)
            om = 1 - rho^2
            d2 = (z1^2 - 2 * rho * z1 * z2 + z2^2) / om
            # Mirrors the fixed code path in src/bivariate_student.jl exactly
            # (ls1 = ls2 = 0 here; the Jacobian terms are additive and untouched
            # by this fix, so they are omitted from this focused check).
            fixed = -(log(2π) + 0.5 * log(om) + ((nu + 2) / 2) * log1p(d2 / nu))
            @test isfinite(fixed)
            @test fixed ≈ ref atol = 1e-9
        end
    end
    # The old, naive difference is what broke: confirm it actually diverges at
    # nu = 1e16 (guards against this regression test silently becoming a no-op
    # if someone "simplifies" it back).
    nu = 1e16
    naive = -(-(loggamma((nu + 2) / 2) - loggamma(nu / 2)) + log(nu) + log(pi) +
              0.5 * log(1.0) + ((nu + 2) / 2) * log1p(0.34 / nu))
    ref = _biv_t_logpdf_ref(0.5, -0.3, 0.0, nu)
    @test abs(naive - ref) > 10   # the historical bug: off by ~36 nats
end

@testset "#bivstudent marginal (p=1) term: stable across nu, matches BigFloat" begin
    for z in (0.0, 0.7, -2.5, 40.0)
        for nu in NU_SWEEP
            ref = _uni_t_logpdf_ref(z, nu)
            ην = log(nu - 2)
            got = DRModels._student_logpdf_std(z, ην)
            @test isfinite(got)
            @test got ≈ ref atol = 1e-9
        end
    end
end

@testset "#bivstudent fit.nll stays on the correct ridge as nu -> huge" begin
    # Fixed-effects bivariate Student fit on ordinary (non-heavy-tailed-looking)
    # data; sweep the nu block of theta far past any value an optimiser would
    # plausibly reach, and check the objective tracks an independent BigFloat
    # recomputation of the same rows rather than diverging (the #721 failure
    # mode: a flat near-Gaussian nu ridge that a fit could wander along).
    rng = MersenneTwister(8202)
    n = 6
    x = randn(rng, n)
    y1 = 0.5 .+ 0.8 .* x .+ 0.3 .* randn(rng, n)
    y2 = -0.2 .+ 0.5 .* x .+ 0.4 .* randn(rng, n)
    data = (; y1 = y1, y2 = y2, x = x)
    f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = drm(f, Student(); data = data)
    @test fit.blocks[5][1] === :nu
    nu_range = fit.blocks[5][2]

    θ = copy(fit.theta)
    b1 = θ[fit.blocks[1][2]]; b2 = θ[fit.blocks[2][2]]
    ls1 = θ[fit.blocks[3][2]][1]; ls2 = θ[fit.blocks[4][2]][1]
    ρ = DRModels.RHO_GUARD * tanh(θ[fit.blocks[6][2]][1])

    X1 = [ones(n) x]; X2 = [ones(n) x]
    η1 = X1 * b1; η2 = X2 * b2
    z1s = (y1 .- η1) .* exp(-ls1)
    z2s = (y2 .- η2) .* exp(-ls2)

    @test length(nu_range) == 1   # intercept-only `nu ~ 1` formula
    for nu in NU_SWEEP
        θ[nu_range[1]] = log(nu - 2)
        ref = sum(_biv_t_logpdf_ref(z1s[i], z2s[i], ρ, nu) - ls1 - ls2 for i in 1:n)
        @test isfinite(fit.nll(θ))
        @test fit.nll(θ) ≈ -ref atol = 1e-6
    end
end

@testset "#bivstudent heavy-tailed bivariate data (true nu = 5) still recovers nu" begin
    rng = MersenneTwister(5721)
    n = 800
    x = randn(rng, n)
    s1, s2, rho, nu_true = 0.7, 1.1, 0.4, 5.0
    z1 = randn(rng, n)
    z2 = rho .* z1 .+ sqrt(1 - rho^2) .* randn(rng, n)
    chi = [sum(randn(rng, Int(nu_true)) .^ 2) for _ in 1:n]
    sh = sqrt.(nu_true ./ chi)
    y1 = 0.5 .+ 0.8 .* x .+ s1 .* z1 .* sh
    y2 = -0.3 .+ 0.4 .* x .+ s2 .* z2 .* sh
    data = (; y1 = y1, y2 = y2, x = x)
    f = bf(mu1 = @formula(y1 ~ x), mu2 = @formula(y2 ~ x),
           sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
           nu = @formula(nu ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = drm(f, Student(); data = data)
    @test is_converged(fit)
    ν̂ = 2 + exp(coef(fit, :nu)[1])
    @test 3.0 < ν̂ < 8.0
    @test coef(fit, :mu1) ≈ [0.5, 0.8] atol = 0.15
    @test coef(fit, :mu2) ≈ [-0.3, 0.4] atol = 0.2
    @test isapprox(fit.scales[:rho12][1], rho; atol = 0.1)
end
