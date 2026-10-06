# Fast REML for the Gaussian σ-phylo location-scale route — the "beat ASReml" speed headline.
#
# The shipped FD-REML refit (`_glsp_reml_refit`) minimises the Patterson–Thompson restricted
# likelihood `nll_ML + 0.5·logdet S` by letting LBFGS finite-difference the whole composite —
# which made the convergence flag misfire (the penalty is itself an FD, so the optimiser saw a
# second-order FD gradient) and converged only by judging on substance. Two faster, clean-flag
# replacements: (1) the CLEAN-GRADIENT refit (`_glsp_reml_refit_clean`, ~6 LBFGS steps) and
# (2) the OBSERVED-INFORMATION Newton (`_glsp_reml_newton_asym`, ~3 steps). NB: the literal
# average-information data-quadratic was proven INVALID here (â is the shrunk BLUP) — see the
# second testset's note.
#
# CORRECTNESS ANCHOR (this file): the FD-REML refit. Both replacements must land on the SAME
# restricted-likelihood optimum — the same σ-phylo SD — on the verified asymmetric route.
using DRModels
using Test, Random, LinearAlgebra

@testset "REML σ-phylo: clean-gradient refit matches FD-REML (asymmetric route)" begin
    Random.seed!(707)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_sig = 0.5 .* (LC * randn(p))               # σ-phylo RE only (asymmetric: mean fixed)
    species = repeat(1:p, inner = m)
    x = randn(n)
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)       # pμ = 2 (non-trivial REML projection)
    βμ = [1.0, -0.4]; βψ = [log(0.5)]
    y = [ (Xμ[i, 1]*βμ[1] + Xμ[i, 2]*βμ[2]) +
          exp(Xψ[i, 1]*βψ[1] + u_sig[species[i]]) * randn() for i in 1:n ]
    pμ = size(Xμ, 2)

    kind = Val(:gaussian_mean)
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη, Zψ = DRModels._glsp_asym_loadings(n)
    obj(θ)  = DRModels._glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    grad(θ) = DRModels._glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)

    # ML estimate (the shared starting point for both refits).
    θ0 = vcat(Xμ \ y, [0.0], log(0.3))
    θ̂_ml, ml_conv = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)
    @test ml_conv

    # FD-REML (the anchor) and the clean-gradient refit, from the identical ML start.
    θ_fd, _, _, _                  = DRModels._glsp_reml_refit(obj, grad, θ̂_ml, pμ; ml_converged = ml_conv)
    θ_cg, conv_cg, _, _, nsteps    = DRModels._glsp_reml_refit_clean(obj, grad, θ̂_ml, pμ; ml_converged = ml_conv)

    sd_fd = exp(θ_fd[end]); sd_cg = exp(θ_cg[end])
    @info "clean-gradient refit vs FD-REML" sd_cg sd_fd diff = abs(sd_cg - sd_fd) nsteps conv_cg
    @test isapprox(sd_cg, sd_fd; rtol = 0.05)    # same restricted-likelihood optimum

    # The whole point of the clean-gradient refit: a CLEAN convergence flag (no substance workaround).
    @test conv_cg
end

# NOTE: the literal average-information data-quadratic was proven INVALID here (â is the
# shrunk BLUP, not a residual with covariance H⁻¹ — an adversarial derivation panel measured
# the candidate at ~0.2× the true curvature). The correct fast O(p) metric is the OBSERVED
# information (central FD of the exact O(p) score) — verified to converge in ~3 Newton steps.
@testset "REML σ-phylo: observed-information Newton (asymmetric route)" begin
    Random.seed!(808)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_sig = 0.5 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x = randn(n)
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    βμ = [1.0, -0.4]; βψ = [log(0.5)]
    y = [ (Xμ[i, 1]*βμ[1] + Xμ[i, 2]*βμ[2]) +
          exp(Xψ[i, 1]*βψ[1] + u_sig[species[i]]) * randn() for i in 1:n ]
    pμ = size(Xμ, 2)

    kind = Val(:gaussian_mean)
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη, Zψ = DRModels._glsp_asym_loadings(n)
    obj(θ)  = DRModels._glsp_asym_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    grad(θ) = DRModels._glsp_asym_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)

    θ0 = vcat(Xμ \ y, [0.0], log(0.3))
    θ̂_ml, ml_conv = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)
    @test ml_conv

    θ_fd, _, _, _ = DRModels._glsp_reml_refit(obj, grad, θ̂_ml, pμ; ml_converged = ml_conv)
    θ_nw, conv, _, _, nnewton =
        DRModels._glsp_reml_newton_asym(kind, y, Xμ, Xψ, gidx, G, Q, Zη, Zψ, θ̂_ml, pμ;
                                   ml_converged = ml_conv)

    sd_fd = exp(θ_fd[end]); sd_nw = exp(θ_nw[end])
    @info "observed-info Newton vs FD-REML" sd_nw sd_fd diff = abs(sd_nw - sd_fd) nnewton conv
    @test conv
    @test nnewton <= 5                            # ASReml-class: a handful of Newton steps
    @test isapprox(sd_nw, sd_fd; rtol = 0.05)     # same restricted-likelihood optimum
end

# SEPARATE block (both axes, K=2): mean-phylo RE + σ-phylo RE, Λ = diag(L11², L22²). The
# general `_glsp_reml_newton` must recover BOTH phylo SDs at the FD-REML optimum in a handful
# of Newton steps — the capability drmTMB lacks (the MUST-HAVE σ-phylo cell, both axes).
@testset "REML σ-phylo: observed-info Newton, separate block (both axes, K=2)" begin
    Random.seed!(909)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_mu  = 0.6 .* (LC * randn(p))
    u_sig = 0.5 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    Xμ = ones(n, 1); Xψ = ones(n, 1); pμ = 1; pψ = 1

    kind = Val(:gaussian_mean)
    Q, gidx, G = DRModels._locscale_phylo_setup(phy, species)
    Zη = DRModels._ls_canonical_Zeta(n); Zψ = DRModels._ls_canonical_Zpsi(n)
    obj(θ)  = DRModels._glsp_sep_nll(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)
    grad(θ) = DRModels._glsp_sep_grad(kind, y, Xμ, Xψ, gidx, G, Q, θ, Zη, Zψ)

    θ0 = vcat(Xμ \ y, [0.0], log(0.3), log(0.3))   # [βμ; βψ; logL11; logL22]
    θ̂_ml, ml_conv = DRModels._glsp_optimise(obj, (g, θ) -> (g .= grad(θ); g), θ0)
    @test ml_conv

    vidx = [pμ + pψ + 1, pμ + pψ + 2]              # logL11, logL22 (indices 3, 4)
    θ_fd, _, _, _ = DRModels._glsp_reml_refit(obj, grad, θ̂_ml, pμ; ml_converged = ml_conv)
    θ_nw, conv, _, _, nstep =
        DRModels._glsp_reml_newton(obj, grad, θ̂_ml, pμ, vidx; ml_converged = ml_conv)

    sd_mu_fd = exp(θ_fd[3]); sd_sig_fd = exp(θ_fd[4])
    sd_mu_nw = exp(θ_nw[3]); sd_sig_nw = exp(θ_nw[4])
    @info "separate-block observed-info Newton vs FD-REML" sd_mu_nw sd_mu_fd sd_sig_nw sd_sig_fd nstep conv
    @test conv
    @test nstep <= 6                               # K=2: a handful of Newton steps
    @test isapprox(sd_mu_nw,  sd_mu_fd;  rtol = 0.06)
    @test isapprox(sd_sig_nw, sd_sig_fd; rtol = 0.06)
end

# REGRESSION (verification 2026-06-12, BLOCKING): drm(method=:REML) crashed near the variance
# boundary (homoscedastic data ⇒ σ-phylo SD → 0) with an opaque LineSearches AssertionError
# (penalty Inf / non-finite gradient → line-search assert). The production REML (robust
# clean-gradient LBFGS: finite penalty + guarded line search) must NOT crash and must return a
# sane, near-boundary σ-phylo SD. Seeds 7 and 39 crashed before the fix.
@testset "REML σ-phylo: no crash at the variance boundary (homoscedastic data)" begin
    for seed in (7, 39, 13, 21)
        Random.seed!(seed)
        p = 20; m = 4; n = p * m
        phy = random_balanced_tree(p; branch_length = 0.3)
        C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
        u_mu = 0.5 .* (LC * randn(p))
        species = repeat(1:p, inner = m)
        y = [1.0 + u_mu[species[i]] + 0.5 * randn() for i in 1:n]   # constant σ: no σ-phylo signal
        data = (; y, species)
        form = bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ phylo(1 | species)))
        local fit
        @test (fit = drm(form, Gaussian(); data = data, tree = phy, method = :REML)) isa DRModels.DrmFit
        sd_sig = exp(coef(fit, :resd_sigma)[1])
        @test isfinite(sd_sig)
        @test sd_sig < 0.6                                          # near the boundary (true σ-phylo SD = 0)
    end
end

# REGRESSION (re-verification 2026-06-12): the 1e18 penalty sentinel (the boundary-crash fix)
# must NOT leak into the reported loglik/AIC/BIC. A rank-deficient (collinear) mean design makes
# the β_μ information non-PD ⇒ penalty = 1e18; the reported REML loglik must be NaN (degenerate),
# not a poisoned −1e18 that would silently corrupt model selection.
@testset "REML σ-phylo: collinear mean design ⇒ NaN loglik, not a 1e18-poisoned value" begin
    Random.seed!(2)
    p = 20; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_mu = 0.5 .* (LC * randn(p)); u_sig = 0.4 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x1 = randn(n); x2 = x1 .+ 1e-12 .* randn(n)          # collinear ⇒ rank-deficient mean design
    y = [1.0 + 0.5 * x1[i] + 0.5 * x2[i] + u_mu[species[i]] +
         exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    data = (; y, species, x1, x2)
    form = bf(@formula(y ~ x1 + x2 + phylo(1 | species)), @formula(sigma ~ phylo(1 | species)))
    fit = drm(form, Gaussian(); data = data, tree = phy, method = :REML)
    rll = reml_loglik(fit)
    @test !(isfinite(rll) && abs(rll) > 1e16)            # NOT a 1e18-poisoned value
    @test isnan(rll)                                     # degenerate fit ⇒ NaN, reported honestly
end

# REGRESSION (teammate report 2026-06-12): the ASYMMETRIC route (mean covariate, σ-phylo only)
# on WEAK σ-signal data threw an uncaught HagerZhang line-search AssertionError. With the
# finite penalty + guarded line search it must DEGRADE gracefully — return a fit (is_converged
# may be false), not throw. Distinct path from the both-phylo boundary test above (seed 1005).
@testset "REML σ-phylo: asymmetric route, weak σ-signal — returns a fit, does not throw" begin
    Random.seed!(1005)
    p = 64; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.30)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_sig = 0.05 .* (LC * randn(p))                       # very weak σ-phylo signal → near boundary
    species = repeat(1:p, inner = m); x = randn(n)
    y = [0.4 + 0.2 * x[i] + exp(0.1 + u_sig[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)
    form = bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species)))   # asymmetric: mean fixed, σ-phylo
    local fit
    @test (fit = drm(form, Gaussian(); data = data, tree = phy, method = :REML)) isa DRModels.DrmFit
    @test isfinite(exp(coef(fit, :resd_sigma)[1]))       # a finite σ-phylo SD (near 0) — no NaN/throw
end

# AYUMI (#2): σ-phylo REML + MISSING RESPONSES must compose — drop missing-y rows (observed-rows
# fit), KEEP the full tree, converge, and recover the σ-phylo SD. Both via drm() and the R-facing
# drm_bridge. This is the exact cell Ayumi needs (missing responses, σ-phylo, REML, from R).
# Before the fix the missing rows were NOT dropped for this route (nobs=144, σ stuck at init,
# loglik NaN). Missing PREDICTORS are a separate future track (drm_listwise / FIML).
@testset "REML σ-phylo: missing responses compose (drm + bridge), Ayumi #2" begin
    Random.seed!(42)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_sig = 0.5 .* (LC * randn(p)); species = repeat(1:p, inner = m); x = randn(n)
    y = [1.0 + 0.5 * x[i] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    ym = Vector{Union{Missing,Float64}}(y); ym[1:20] .= missing   # 20 missing responses
    data = (; y = ym, x, species)
    form = bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species)))

    fit = drm(form, Gaussian(); data = data, tree = phy, method = :REML)
    @test is_converged(fit)
    @test nobs(fit) == n - 20                            # the 20 missing rows dropped (observed-rows fit)
    @test 0.2 < exp(coef(fit, :resd_sigma)[1]) < 1.2     # σ-phylo SD recovered (true 0.5)

    res = DRModels.drm_bridge(; formula = "y ~ x; sigma ~ phylo(1 | species)", family = "gaussian",
                         data = Dict("y" => ym, "x" => x, "species" => species), tree = phy,
                         options = Dict("method" => "REML"))
    @test res["converged"]
    @test res["nobs"] == n - 20                          # the bridge sees the observed-rows fit
    @test isfinite(res["loglik"])
end

# AYUMI (#2) boundary inference: the σ-phylo SD profile CI must be reachable via BOTH
# confint(:profile) AND drm_bridge_inference. The route attaches no re-optimisable objective,
# so the boundary-aware CI (precomputed into scales when profile_ci=true, SD scale, honest
# [0,x] at the variance boundary) is surfaced. Before the fix both threw "requires the fitted
# objective". (σ-phylo locscale + REML + boundary inference = the community's most-used trio.)
@testset "REML σ-phylo: boundary inference exposed (confint + bridge), Ayumi #2" begin
    Random.seed!(42)
    p = 24; m = 6; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0); LC = cholesky(Symmetric(C)).L
    u_sig = 0.5 .* (LC * randn(p)); species = repeat(1:p, inner = m); x = randn(n)
    y = [1.0 + 0.5 * x[i] + exp(log(0.5) + u_sig[species[i]]) * randn() for i in 1:n]
    data = (; y, x, species)
    form = bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species)))

    fit = drm(form, Gaussian(); data = data, tree = phy, method = :REML, profile_ci = true)
    ci = confint(fit; method = :profile)
    row = only(filter(r -> r.param === :resd_sigma, ci))
    @test 0 <= row.lower < row.upper                                  # SD scale, ordered, ≥0
    @test row.lower < exp(coef(fit, :resd_sigma)[1]) < row.upper      # SD point inside the CI

    res = DRModels.drm_bridge_inference(; formula = "y ~ x; sigma ~ phylo(1 | species)", family = "gaussian",
            data = Dict("y" => y, "x" => x, "species" => species), tree = phy,
            options = Dict("method" => "REML"), method = "profile")
    @test res["param"] == "resd_sigma"
    @test res["status"] == "profile"
    @test 0 <= res["lower"] < res["estimate"] < res["upper"]          # SD-scale boundary CI from R

    explicit = DRModels.drm_bridge_inference(; formula = "y ~ x; sigma ~ phylo(1 | species)", family = "gaussian",
            data = Dict("y" => y, "x" => x, "species" => species), tree = phy,
            options = Dict("method" => "REML"), method = "profile", parm = "sd:sigma")
    @test explicit["param"] == "resd_sigma"
    @test explicit["estimate"] == res["estimate"]
    @test explicit["lower"] == res["lower"]
    @test explicit["upper"] == res["upper"]
end
