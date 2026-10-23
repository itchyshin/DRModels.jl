# test_xfam_external_validation.jl — INDEPENDENT validation of DRModels.jl's
# cross-family latent correlation (`fit_mixed_family`) against external references.
#
# DRModels's shared-latent correlation
#     rho = lambda1 lambda2 / sqrt((lambda1^2 + v1) (lambda2^2 + v2)),  v_k = link_residual(fam_k),
# is the SAME estimand as the trait-trait residual correlation of a 2-response,
# 1-factor GLLVM. Three independent checks, in increasing scope:
#
#  (1) EXTERNAL PACKAGE — Gaussian x Gaussian, DRModels (GHQ engine) vs `gllvm`
#      (VA/Laplace engine, an entirely separate codebase). gllvm::getResidualCor
#      on a 2-column `gllvm(num.lv = 1)` fit IS this rho (verified: it equals
#      lambda1 lambda2 / sqrt((lambda1^2+phi1^2)(lambda2^2+phi2^2)) reconstructed from
#      gllvm's own loadings). DRModels refits the IDENTICAL simulated data and must agree
#      to cross-package tolerance. GUARDED: skips (with @info) if the gllvm fixture
#      is absent, so the suite is portable to machines without R/gllvm. Regenerate
#      with `Rscript test/parity/gen_xfam_external.R`.
#
#      NOTE on scope: stock gllvm 2.0.5 fits ONE family per response matrix (a
#      per-column family vector errors), so gllvm gives an external reference for
#      the SAME-family case only. The genuinely MIXED Gaussian x Poisson rho is
#      therefore validated in (2) against an independent Monte-Carlo reference.
#
#  (2) CROSS-FAMILY, INDEPENDENT MONTE-CARLO — Gaussian x Poisson. The population
#      latent-scale rho is computed from the TRUE generative parameters using DRModels's
#      documented standardization (v1 = sigma1^2; v2 = log(1 + 1/mu-bar2) with
#      mu-bar2 = E_x[exp(X beta2)] the conditional-mean baseline, EXCLUDING the
#      latent — this is `rho_of`'s convention, mu-bar2 from X*beta2 alone). The
#      reference uses no DRModels fit. DRModels's estimate at large n must converge to it.
#
#  (3) CLOSED FORM — Gaussian x Gaussian, where the marginal is exactly bivariate
#      normal so rho = lambda1 lambda2 / sqrt((lambda1^2+sigma1^2)(lambda2^2+sigma2^2))
#      holds analytically. DRModels must recover it.
#
# License: gllvm is GPL; only its fitted NUMBERS (data, not source) are stored in
# the fixture — mirrors the drmTMB parity-fixture contract (AGENTS.md s.3).

using DRModels
using Test, Random, Statistics, Printf
using TOML
using DelimitedFiles: readdlm

# Knuth Poisson sampler — keeps the test free of a Distributions dependency
# (same convention as test_mixed_family.jl's `_rpois`).
function _xfam_rpois(rng, λ)
    L = exp(-λ); k = 0; p = 1.0
    while true
        k += 1; p *= rand(rng)
        p <= L && return k - 1
    end
end

const _XFAM_FIXDIR = joinpath(@__DIR__, "parity", "fixtures", "xfam-external-gllvm")

@testset "Cross-family latent correlation — independent validation" begin

    # -- (1) External package reference: gllvm (Gaussian x Gaussian) -----------
    @testset "External gllvm reference (Gaussian x Gaussian)" begin
        expected = joinpath(_XFAM_FIXDIR, "expected.toml")
        datafile = joinpath(_XFAM_FIXDIR, "data.csv")
        if !(isfile(expected) && isfile(datafile))
            @info "gllvm external fixture absent — skipping external cross-check " *
                  "(regenerate: Rscript test/parity/gen_xfam_external.R)" dir = _XFAM_FIXDIR
        else
            t = TOML.parsefile(expected)
            rho_gllvm = Float64(t["fit"]["rho_gllvm"])
            ll_gllvm  = Float64(t["fit"]["loglik"])
            atol_x    = Float64(get(t["tol"], "atol_rho_xpackage", 2e-2))

            raw, hdr = readdlm(datafile, ','; header = true)
            cols = Symbol.(strip.(string.(vec(hdr))))
            col(name) = Vector{Float64}(raw[:, findfirst(==(name), cols)])
            y1 = col(:y1); y2 = col(:y2); x = col(:x)
            n = length(y1)
            X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)

            fit = DRModels.fit_mixed_family(y1 = y1, X1 = X1, fam1 = Gaussian(),
                                       y2 = y2, X2 = X2, fam2 = Gaussian(),
                                       confint = false)
            @test fit.converged
            # DRModels (GHQ) and gllvm (VA) fit the SAME data → same estimand, agree to
            # cross-package tolerance. (On the calibration data the achieved gap was
            # ~1e-4; the 2e-2 band is a conservative engine-vs-engine margin.)
            @test isapprox(fit.rho_latent, rho_gllvm; atol = atol_x)
            # Same marginal bivariate-normal likelihood: DRModels's optimum is at least
            # as high as gllvm's (the rho/logLik are identified even though the
            # individual Gaussian-axis loadings sit on a flat ridge).
            @test fit.loglik >= ll_gllvm - 0.25
            @info "external gllvm cross-check" rho_DRM = fit.rho_latent rho_gllvm =
                rho_gllvm gap = abs(fit.rho_latent - rho_gllvm) loglik_DRM =
                fit.loglik loglik_gllvm = ll_gllvm
        end
    end

    # -- (2) Cross-family vs independent Monte-Carlo reference -----------------
    # Gaussian x Poisson: the literally MIXED case gllvm cannot fit. Validate the
    # rho against a population reference built ONLY from the true parameters.
    @testset "Cross-family Gaussian x Poisson vs Monte-Carlo reference" begin
        β1 = [0.5, 0.8]; β2 = [0.3, -0.5]
        λ1 = 0.8; λ2 = 0.6; σ1 = 0.5

        # Independent population reference (no DRModels fit): DRModels's standardization with
        # mu-bar2 = E_x[exp(X beta2)] (conditional-mean baseline). E_x is a 2e6-draw
        # Monte-Carlo over the TRUE covariate law.
        xref = randn(MersenneTwister(777), 2_000_000)
        μ̄2 = mean(exp.(β2[1] .+ β2[2] .* xref))
        v1_ref = σ1^2
        v2_ref = log1p(1 / μ̄2)                       # Poisson link-residual (N&S 2010)
        rho_ref = (λ1 * λ2) / sqrt((λ1^2 + v1_ref) * (λ2^2 + v2_ref))

        # DRModels fits at large n; average a few seeds to suppress Monte-Carlo scatter.
        # (n = 10_000 × 3 seeds gave |mean - ref| ≈ 8e-4, max single-seed ≈ 6e-3.)
        n = 10_000
        seeds = (101, 202, 303)
        fits = map(seeds) do seed
            rng = MersenneTwister(seed)
            x = randn(rng, n); X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
            u = randn(rng, n)
            η2 = X2 * β2 .+ λ2 .* u
            y1 = X1 * β1 .+ λ1 .* u .+ σ1 .* randn(rng, n)
            y2 = Float64[_xfam_rpois(rng, exp(clamp(η2[i], -20.0, 20.0))) for i in 1:n]
            DRModels.fit_mixed_family(y1 = y1, X1 = X1, fam1 = Gaussian(),
                                 y2 = y2, X2 = X2, fam2 = DRModels.Poisson(),
                                 confint = false)
        end
        @test all(f -> f.converged, fits)

        ρ̂s = [f.rho_latent for f in fits]
        ρ̄ = mean(ρ̂s)
        # Structural recovery (independent confirmation the latent is captured).
        @test isapprox(mean(f.λ1 for f in fits), λ1; atol = 0.05)
        @test isapprox(mean(f.λ2 for f in fits), λ2; atol = 0.05)
        @test isapprox(mean(f.σ1 for f in fits), σ1; atol = 0.05)
        # Reconstruct DRModels's own rho from its (λ, v) — pins the reported value to the
        # documented formula (self-consistency, not a coincidence).
        for f in fits
            @test isapprox(f.rho_latent,
                           (f.λ1 * f.λ2) / sqrt((f.λ1^2 + f.v1) * (f.λ2^2 + f.v2));
                           atol = 1e-10)
        end
        # The cross-family estimate converges to the INDEPENDENT reference.
        @test isapprox(ρ̄, rho_ref; atol = 2e-2)            # averaged: tight
        @test all(abs.(ρ̂s .- rho_ref) .< 3e-2)             # each fit: sensible band
        @info "cross-family Monte-Carlo check" rho_ref = rho_ref rho_DRM_mean = ρ̄ gap =
            abs(ρ̄ - rho_ref) per_seed = ρ̂s
    end

    # -- (3) Gaussian x Gaussian closed form ----------------------------------
    @testset "Gaussian x Gaussian closed-form rho" begin
        rng = MersenneTwister(2026)
        n = 4000
        x = randn(rng, n); X1 = hcat(ones(n), x); X2 = hcat(ones(n), x)
        β1 = [0.4, 0.7]; β2 = [-0.2, 0.5]
        λ1 = 0.9; λ2 = 0.7; σ1 = 0.5; σ2 = 0.6
        u = randn(rng, n)
        y1 = X1 * β1 .+ λ1 .* u .+ σ1 .* randn(rng, n)
        y2 = X2 * β2 .+ λ2 .* u .+ σ2 .* randn(rng, n)
        rho_closed = (λ1 * λ2) / sqrt((λ1^2 + σ1^2) * (λ2^2 + σ2^2))

        fit = DRModels.fit_mixed_family(y1 = y1, X1 = X1, fam1 = Gaussian(),
                                   y2 = y2, X2 = X2, fam2 = Gaussian(),
                                   confint = false)
        @test fit.converged
        @test isapprox(fit.rho_latent, rho_closed; atol = 3e-2)
        @info "Gaussian x Gaussian closed-form check" rho_closed = rho_closed rho_DRM =
            fit.rho_latent gap = abs(fit.rho_latent - rho_closed)
    end
end
