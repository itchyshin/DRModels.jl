# Predictor-dependent random-effect SD for a Gaussian structured INTERCEPT:
#   y_i = X_i b + u_{g(i)} + e_i,  u = D L z,  D = diag(exp(Z a)),  C = L L',
# for animal(1 | g, A), relmat(1 | g, K) and coordinate spatial(1 | g, coords),
# each with sd(g, <level>) ~ z.  Counterpart of test_gaussian_structured_sd_slope.jl
# with loading w_i = 1 instead of w_i = x_i.
#
# Scale choice (recorded): A and K enter RAW, as in native drmTMB. Coordinate
# spatial uses the native fixed kernel exp(-d / median positive distance) +
# 1e-6 I (range fixed, not estimated). The marginal likelihood is exact, so the
# oracle is an independent dense MvNormal log density built here without calling
# the fitter's code. Candidate-level evidence: one fixture per route, no
# calibration claim.
using DRModels, Test, Random, LinearAlgebra, Statistics
import Distributions

function _gsi_kernel(kind::Symbol, G::Int, rng)
    if kind === :spatial
        coords = randn(rng, G, 2)
        Dd = [norm(coords[k, :] - coords[l, :]) for k in 1:G, l in 1:G]
        pos = [Dd[k, l] for k in 1:G for l in (k+1):G]
        C = exp.(-Dd ./ median(pos)) + 1e-6 * I
        return Matrix(C), coords
    end
    R = 0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    s = collect(range(0.7, 1.6; length = G))
    return Matrix(Diagonal(s) * R * Diagonal(s)), nothing
end

function _gsi_fixture(kind::Symbol; G = 24, m = 10, seed = 20261102)
    rng = MersenneTwister(seed)
    C, coords = _gsi_kernel(kind, G, rng)
    labels = ["id_$(i)" for i in randperm(rng, G)]
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    alpha = [-0.4, 0.55]
    sdg = exp.(alpha[1] .+ alpha[2] .* zgroup)
    u = sdg .* (cholesky(Symmetric(C)).L * randn(rng, G))
    group_index = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    y = 0.3 .+ 0.2 .* x .+ u[group_index] .+ 0.5 .* randn(rng, G * m)
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

# Independent oracle: -log N(y; X b, sigma^2 I + Z D C D Z'), Z one-hot (loading 1).
# `loading` and `D` are arguments so the emulated-bug checks can corrupt them.
function _gsi_oracle_nll(theta, fx, C; loading = ones(length(fx.data.y)),
                         zvec = fx.zgroup)
    y, x = fx.data.y, fx.data.x
    beta, sigma, alpha = theta[1:2], exp(theta[3]), theta[4:5]
    D = exp.(hcat(ones(length(zvec)), zvec) * alpha)
    n, G = length(y), length(D)
    L = zeros(n, G)
    for i in 1:n
        L[i, fx.group_index[i]] = loading[i] * D[fx.group_index[i]]
    end
    V = sigma^2 * Matrix(I, n, n) + L * C * L'
    return -Distributions.logpdf(
        Distributions.MvNormal(hcat(ones(n), x) * beta, Symmetric(V)), y)
end

_gsi_formula(kind, rhs = :z) = bf(
    kind === :animal ? @formula(y ~ x + animal(1 | g)) :
    kind === :relmat ? @formula(y ~ x + relmat(1 | g)) :
                       @formula(y ~ x + spatial(1 | g)),
    rhs === :z ?
        (kind === :animal ? @formula(sd(g, animal) ~ z) :
         kind === :relmat ? @formula(sd(g, relmat) ~ z) :
                            @formula(sd(g, spatial) ~ z)) :
        (kind === :animal ? @formula(sd(g, animal) ~ 1) :
         kind === :relmat ? @formula(sd(g, relmat) ~ 1) :
                            @formula(sd(g, spatial) ~ 1)))

_gsi_kw(kind, fx) = kind === :animal ? (; A = fx.C) :
                    kind === :relmat ? (; K = fx.C) : (; coords = fx.coords)

@testset "Gaussian structured intercept SD regression" begin
    for kind in (:animal, :relmat, :spatial)
        @testset "$kind" begin
            fx = _gsi_fixture(kind)
            data = fx.data
            fit = drm(_gsi_formula(kind), Gaussian(); data, _gsi_kw(kind, fx)...)
            @test fit.converged
            @test length(coef(fit, :mu)) == 2
            @test length(coef(fit, :sd)) == 2
            @test all(isfinite, coef(fit, :sd))

            # Exact marginal likelihood against the independent dense oracle,
            # at the truth, at the fit, and at perturbed parameters.
            theta_hat = vcat(coef(fit, :mu), coef(fit, :sigma), coef(fit, :sd))
            theta_true = [0.3, 0.2, log(0.5), fx.alpha...]
            for th in (theta_true, theta_hat, theta_hat .+ 0.07)
                @test fit.nll(th) ≈ _gsi_oracle_nll(th, fx, fx.C) rtol = 1e-9
            end
            @test fit.loglik ≈ -_gsi_oracle_nll(theta_hat, fx, fx.C) rtol = 1e-9
            # The fit sits at an optimum of the oracle surface.
            base = _gsi_oracle_nll(theta_hat, fx, fx.C)
            for j in eachindex(theta_hat), h in (-0.05, 0.05)
                th = copy(theta_hat); th[j] += h
                @test _gsi_oracle_nll(th, fx, fx.C) > base - 1e-6
            end

            # Emulated bugs must be caught by the same oracle comparison.
            # (1) slope loading used for an intercept term.
            @test abs(fit.nll(theta_hat) -
                      _gsi_oracle_nll(theta_hat, fx, fx.C; loading = fx.data.x)) > 0.5
            # (2) the SD formula ignored (constant SD at the first coefficient).
            @test abs(fit.nll(theta_hat) -
                      _gsi_oracle_nll(theta_hat, fx, fx.C; zvec = zeros(length(fx.zgroup)))) > 0.05
            # (3) flipped sign of the SD covariate.
            @test abs(fit.nll(theta_hat) -
                      _gsi_oracle_nll(theta_hat, fx, fx.C; zvec = -fx.zgroup)) > 0.05
            # (4) kernel rows not aligned to the first-seen group order.
            Cbad = fx.C[reverse(1:end), reverse(1:end)]
            @test abs(fit.nll(theta_hat) - _gsi_oracle_nll(theta_hat, fx, Cbad)) > 0.05
            # (5) raw (not normalised) kernel: a cov2cor kernel is a different
            # surface. (The coordinate kernel has a unit diagonal, so vacuous there.)
            d = sqrt.(diag(fx.C))
            kind === :spatial || @test abs(fit.nll(theta_hat) -
                      _gsi_oracle_nll(theta_hat, fx, fx.C ./ (d * d'))) > 0.05

            # Recovery is loose (24 groups). The fixed intercept is confounded with
            # the mean of the structured field, so only the slope on x is asserted.
            @test isfinite(coef(fit, :mu)[1])
            @test abs(coef(fit, :mu)[2] - 0.2) < 0.2
            @test abs(coef(fit, :sd)[1] - fx.alpha[1]) < 1.2

            # Scales: sd_group = exp(Z alpha), expanded to rows by first-seen index.
            @test fit.scales[:sd_group] ≈ exp.(hcat(ones(length(fx.zgroup)), fx.zgroup) *
                                               coef(fit, :sd))
            @test fit.scales[:sd] ≈ fit.scales[:sd_group][fx.group_index]

            # Constant SD nests inside the z model.
            cfit = drm(_gsi_formula(kind, :const), Gaussian(); data, _gsi_kw(kind, fx)...)
            @test cfit.converged && length(coef(cfit, :sd)) == 1
            @test fit.loglik >= cfit.loglik - 1e-6

            # The fixed-effect design survives for prediction.
            @test length(predict(fit, data; type = :link)) == length(data.y)

            # Refusals.
            formula = _gsi_formula(kind)
            kw = _gsi_kw(kind, fx)
            zbad = copy(data.z); zbad[1] += 1.0
            @test_throws ArgumentError drm(formula, Gaussian();
                data = merge(data, (; z = zbad)), kw...)
            @test_throws Exception drm(formula, Gaussian(); data)   # no kernel input
        end
    end

    # Wrong provider input is refused, not silently reinterpreted.
    fx = _gsi_fixture(:relmat)
    @test_throws ArgumentError drm(_gsi_formula(:relmat), Gaussian(); data = fx.data,
                                   K = fx.C, A = fx.C)
    fa = _gsi_fixture(:animal)
    @test_throws ArgumentError drm(_gsi_formula(:animal), Gaussian(); data = fa.data,
                                   K = fa.C)
    # A coefficient-selected SD target that names a slope is refused for an intercept.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + relmat(1 | g)),
                                      @formula(sd(g, relmat, coef(x)) ~ z)),
                                   Gaussian(); data = fx.data, K = fx.C)
    # An ordinary random effect next to the structured intercept keeps refusing.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g) + relmat(1 | g)),
                                      @formula(sd(g, relmat) ~ z)),
                                   Gaussian(); data = fx.data, K = fx.C)
    # A structured SD target on a different level than the marker keeps refusing.
    @test_throws ArgumentError drm(bf(@formula(y ~ x + relmat(1 | g)),
                                      @formula(sd(g, animal) ~ z)),
                                   Gaussian(); data = fx.data, K = fx.C)
end

@testset "drm_bridge Gaussian relmat intercept SD" begin
    fx = _gsi_fixture(:relmat; seed = 20261103)
    labels = Dict{String,Any}("coef_labels" => Dict(
        "mu" => ["(Intercept)", "x"], "sigma" => ["(Intercept)"],
        "sd" => ["(Intercept)", "z"]))
    out = drm_bridge(; formula = "y ~ x + relmat(1 | g); sd(g, relmat) ~ z",
                     family = "gaussian", data = fx.data, K = fx.C,
                     options = labels)
    direct = drm(_gsi_formula(:relmat), Gaussian(); data = fx.data, K = fx.C)
    @test out["coefficients"] ≈ coef(direct) atol = 1e-6
    @test out["loglik"] ≈ direct.loglik atol = 1e-6
    theta = out["coefficients"]
    @test out["loglik"] ≈ -_gsi_oracle_nll(vcat(theta[1:2], theta[3], theta[4:5]), fx, fx.C) rtol = 1e-9
end
