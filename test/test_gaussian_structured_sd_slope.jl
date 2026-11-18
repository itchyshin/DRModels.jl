# Predictor-dependent random-effect SD for Gaussian structured slope-only
# coefficients:  y_i = X_i b + x_i u_g + e_i,  u = D L z,  D = diag(exp(Z a)),
# for animal(0 + x | g, A), relmat(0 + x | g, K) and coordinate
# spatial(0 + x | g, coords), each with sd(g, <level>) ~ z.
#
# Scale choice (recorded): A and K enter RAW, matching native drmTMB; the
# diagonal is not normalised. Coordinate spatial uses the native fixed kernel
# exp(-d / median positive distance) + 1e-6 I (range fixed, not estimated).
# The marginal likelihood is exact, so the oracle is an independent dense
# MvNormal log density, built here without calling the fitter's code.
# Candidate-level evidence: one fixture per route, no calibration claim.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

# Latent-order kernels, each with a non-unit diagonal and non-Toeplitz structure.
function _gss_kernel(kind::Symbol, G::Int, rng)
    if kind === :spatial
        coords = randn(rng, G, 2)
        Dd = [norm(coords[k, :] - coords[l, :]) for k in 1:G, l in 1:G]
        pos = [Dd[k, l] for k in 1:G for l in (k+1):G]
        C = exp.(-Dd ./ median(pos)) + 1e-6 * I
        return C, coords
    end
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.6; length = G))
    return Matrix(Diagonal(s) * R * Diagonal(s)), nothing
end

function _gss_fixture(kind::Symbol; G = 20, m = 10, seed = 20261101)
    rng = MersenneTwister(seed)
    C, coords = _gss_kernel(kind, G, rng)
    labels = ["id_$(i)" for i in randperm(rng, G)]
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    alpha = [-0.4, 0.5]
    sdg = exp.(alpha[1] .+ alpha[2] .* zgroup)
    u = sdg .* (cholesky(Symmetric(C)).L * randn(rng, G))
    group_index = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    y = 0.3 .+ 0.2 .* x .+ x .* u[group_index] .+ 0.5 .* randn(rng, G * m)
    group = labels[group_index]
    # Shuffle rows; align the kernel (and coords) to the first-seen levels.
    ord = randperm(rng, length(y))
    fitted_labels = unique(group[ord])
    orig = [findfirst(==(l), labels) for l in fitted_labels]
    gi = [findfirst(==(l), fitted_labels) for l in group[ord]]
    data = (; y = y[ord], x = x[ord], z = zgroup[orig][gi], g = group[ord])
    return (; data, C = C[orig, orig], coords = coords === nothing ? nothing : coords[orig, :],
            group_index = gi, zgroup = zgroup[orig], alpha, labels = fitted_labels)
end

# Independent oracle: -log N(y; X b, sigma^2 I + L C L'), L_ig = x_i D_g.
function _gss_oracle_nll(theta, fx, C)
    y, x = fx.data.y, fx.data.x
    beta, sigma, alpha = theta[1:2], exp(theta[3]), theta[4:5]
    D = exp.(hcat(ones(length(fx.zgroup)), fx.zgroup) * alpha)
    n, G = length(y), length(D)
    L = zeros(n, G)
    for i in 1:n
        L[i, fx.group_index[i]] = x[i] * D[fx.group_index[i]]
    end
    V = sigma^2 * Matrix(I, n, n) + L * C * L'
    return -Distributions.logpdf(
        Distributions.MvNormal(hcat(ones(n), x) * beta, Symmetric(V)), y)
end

_gss_formula(kind) = bf(
    kind === :animal ? @formula(y ~ x + animal(0 + x | g)) :
    kind === :relmat ? @formula(y ~ x + relmat(0 + x | g)) :
                       @formula(y ~ x + spatial(0 + x | g)),
    kind === :animal ? @formula(sd(g, animal) ~ z) :
    kind === :relmat ? @formula(sd(g, relmat) ~ z) :
                       @formula(sd(g, spatial) ~ z))

_gss_kw(kind, fx) = kind === :animal ? (; A = fx.C) :
                    kind === :relmat ? (; K = fx.C) : (; coords = fx.coords)

@testset "Gaussian structured slope-only SD regression" begin
    for kind in (:animal, :relmat, :spatial)
        @testset "$kind" begin
            fx = _gss_fixture(kind)
            data = fx.data
            formula = _gss_formula(kind)
            fit = drm(formula, Gaussian(); data, _gss_kw(kind, fx)...)
            @test fit.converged
            @test length(coef(fit, :mu)) == 2
            @test length(coef(fit, :sd)) == 2
            @test all(isfinite, coef(fit, :sd))

            # Exact marginal likelihood against the independent dense oracle,
            # at the truth, at the fit, and at perturbed parameters.
            theta_hat = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
            theta_true = [0.3, 0.2, log(0.5), fx.alpha...]
            for th in (theta_true, theta_hat, theta_hat .+ 0.07)
                @test fit.nll(th) ≈ _gss_oracle_nll(th, fx, fx.C) rtol = 1e-9
            end
            @test fit.loglik ≈ -_gss_oracle_nll(theta_hat, fx, fx.C) rtol = 1e-9
            # The fit sits at an optimum of the oracle surface.
            base = _gss_oracle_nll(theta_hat, fx, fx.C)
            for j in eachindex(theta_hat), h in (-0.05, 0.05)
                th = copy(theta_hat); th[j] += h
                @test _gss_oracle_nll(th, fx, fx.C) > base - 1e-6
            end

            # Raw (not normalised) kernel: the cov2cor oracle is a different
            # likelihood surface, so an accidental normalisation would be caught.
            # (The coordinate kernel has a unit diagonal, so the check is vacuous there.)
            d = sqrt.(diag(fx.C))
            kind === :spatial || @test abs(fit.nll(theta_hat) -
                      _gss_oracle_nll(theta_hat, fx, fx.C ./ (d * d'))) > 0.05

            # Recovery is loose (20 groups); the SD slope is not asserted by sign.
            @test abs(coef(fit, :mu)[1] - 0.3) < 0.5
            @test abs(coef(fit, :mu)[2] - 0.2) < 0.5
            @test abs(coef(fit, :sd)[1] - fx.alpha[1]) < 1.0

            # Scales: sd_group = exp(Z alpha), expanded to rows by first-seen index.
            @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fx.zgroup)), fx.zgroup) *
                                               coef(fit, :sd))
            @test fit.scales[:sd] ≈ fit.scales[:sd_group][fx.group_index]

            # Constant SD nests inside the z model.
            constant_formula = bf(
                kind === :animal ? @formula(y ~ x + animal(0 + x | g)) :
                kind === :relmat ? @formula(y ~ x + relmat(0 + x | g)) :
                                   @formula(y ~ x + spatial(0 + x | g)),
                kind === :animal ? @formula(sd(g, animal) ~ 1) :
                kind === :relmat ? @formula(sd(g, relmat) ~ 1) :
                                   @formula(sd(g, spatial) ~ 1))
            cfit = drm(constant_formula, Gaussian(); data, _gss_kw(kind, fx)...)
            @test cfit.converged && length(coef(cfit, :sd)) == 1
            @test fit.loglik >= cfit.loglik - 1e-6

            # The fixed-effect design survives for prediction.
            @test length(predict(fit, data; type = :link)) == length(data.y)

            # Refusals.
            kw = _gss_kw(kind, fx)
            zbad = copy(data.z); zbad[1] += 1.0
            @test_throws ArgumentError drm(formula, Gaussian();
                data = merge(data, (; z = zbad)), kw...)
            @test_throws ArgumentError drm(formula, Gaussian();
                data = merge(data, (; x = zeros(length(data.x)))), kw...)
            @test_throws Exception drm(formula, Gaussian(); data)   # no kernel input
        end
    end

    # Wrong provider input is refused, not silently reinterpreted.
    fx = _gss_fixture(:relmat)
    @test_throws ArgumentError drm(_gss_formula(:relmat), Gaussian(); data = fx.data,
                                   K = fx.C, A = fx.C)
    fa = _gss_fixture(:animal)
    @test_throws ArgumentError drm(_gss_formula(:animal), Gaussian(); data = fa.data,
                                   K = fa.C)
    # A structured slope with no matching sd() target keeps its old refusal.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + animal(0 + x | g))), Gaussian();
                                   data = fa.data, A = fa.C)
end
