# fitted() on two more opt-in `marginal = :Laplace` routes: the modes are kept at the fitted parameters.
#
#   * Binomial probit / cloglog with one correlated block `(1 + x1 | g)` and a `cbind(s, fl)` response;
#   * ZeroOneBeta with one random intercept `(1 | g)` and a fixed-effect `sigma ~ x1` (also its default route).
#
# `fitted(fit)` is conditional (`re = :conditional`), as on the other routes (test_laplace_fitted_modes.jl).
# Before, the call threw because these routes stored no modes. References are drmTMB 0.8.1 fitted values in
#   fixtures/laplace_more_shapes_parity/ref_binom_{probit,cloglog}_slope.toml  <- parity/gen_laplace_more_shapes_parity.R
#   fixtures/laplace_sigma_formula_parity/ref_zob_sigma_x1.toml                <- parity/gen_laplace_sigma_formula_parity.R
# The bound is 1e-6, the one test_fitted_conditional.jl and test_laplace_fitted_modes.jl use.
#
# A third group covers the Binomial `(1 | g)` + `sd(g) ~ gx` routes (logit and probit, default integrator).
# They keep their modes through the same fitter, so `fitted(fit)` is conditional there too; before it threw.
# That route has no Laplace option here, and the default quadrature differs from drmTMB's Laplace fit by about
# 3e-4 in `fitted`, so the drmTMB values (fixtures/binomial_sd_fitted_parity/, <- parity/gen_binomial_sd_fitted_parity.R)
# are used in two steps: the conditional value at drmTMB's own estimates, computed here by Newton's method on
# the joint density, equals drmTMB's `fitted` at 1e-6; and the package's `fitted` equals the same computation at
# its own estimates at 1e-8. The package value is also within 1e-3 of drmTMB's.
using DRModels, Test, TOML, LinearAlgebra, Statistics
using SpecialFunctions: erfc, loggamma
import ForwardDiff

const _LM_TOL = 1e-6

function _lm_data(dir; ints = Symbol[])
    L = readlines(joinpath(@__DIR__, "fixtures", dir, "data.csv"))
    hdr = [strip(c, '"') for c in split(L[1], ',')]
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(hdr)
        raw = [strip(split(l, ',')[j], '"') for l in L[2:end]]
        cols[Symbol(h)] = all(s -> tryparse(Float64, s) !== nothing, raw) ? parse.(Float64, raw) : String.(raw)
    end
    for k in ints
        cols[k] = round.(Int, cols[k])
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

_lm_ref(dir, id) = TOML.parsefile(joinpath(@__DIR__, "fixtures", dir, "ref_$(id).toml"))

# Per group, recover the mode (b0, b1) from the stored per-row contribution b0 + b1 x and check that the
# score of the joint log density (data given b, plus the Gaussian prior with the fitted covariance) is zero.
function _lm_slope_score(fit, d, link)
    th = fit.theta; sd = exp.(th[3:4]); rho = fit.scales[:corpars][1]
    P = inv(Diagonal(sd) * [1 rho; rho 1] * Diagonal(sd))
    pf = link === :probit ? (e -> 0.5 * erfc(-e / sqrt(2))) : (e -> -expm1(-exp(e)))
    df = link === :probit ? (e -> exp(-e^2 / 2) / sqrt(2pi)) : (e -> exp(e - exp(e)))
    stored = fit.scales[:cond_fit_eta_re]
    eta0 = th[1] .+ th[2] .* d.x1
    worst = 0.0
    for g in unique(d.g)
        idx = findall(==(g), d.g)
        Z = hcat(ones(length(idx)), d.x1[idx])
        b = Z \ stored[idx]
        @test maximum(abs.(Z * b .- stored[idx])) < 1e-9                  # the record is exactly b0 + b1 x
        e = eta0[idx] .+ Z * b
        # analytic score of log p(s, fl | b) + log N(b; 0, Sigma) in b, relative to the size of its two parts
        gdata = Z' * ((d.s[idx] ./ pf.(e) .- d.fl[idx] ./ (1 .- pf.(e))) .* df.(e))
        gprior = P * b
        worst = max(worst, maximum(abs.(gdata .- gprior)) / (1 + maximum(abs.(gdata)) + maximum(abs.(gprior))))
    end
    return worst
end

@testset "fitted(): Binomial (1 + x1 | g) Laplace keeps the modes" begin
    d = _lm_data("laplace_more_shapes_parity"; ints = [:y_bin, :s, :fl])
    for (id, link) in (("binom_probit_slope", :probit), ("binom_cloglog_slope", :cloglog))
        @testset "$id" begin
            ref = _lm_ref("laplace_more_shapes_parity", id)
            fit = drm(bf(@formula(cbind(s, fl) ~ x1 + (1 + x1 | g))), DRModels.Binomial(link);
                      data = d, marginal = :Laplace)
            @test fit.marginal === :Laplace
            @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
            cond = fitted(fit)
            @test length(cond) == nobs(fit) == length(ref["fitted"])
            @test maximum(abs.(cond .- Float64.(ref["fitted"]))) < _LM_TOL                # drmTMB fitted(fit)
            @test fitted(fit; re = :conditional) == cond
            zero = fitted(fit; re = :zero)
            @test zero == fit.means[:mu]                                                   # population value as before
            @test maximum(abs.(cond .- zero)) > 1e-3                                       # the modes are really in
            @test !haskey(fit.scales, :cond_fit_failed)
            y = fit.obs[:mu]
            @test residuals(fit) ≈ y .- cond
            @test residuals(fit; re = :zero) ≈ y .- zero
            @test _lm_slope_score(fit, d, link) < 1e-5                                    # modes solve the inner problem
        end
    end
end

@testset "fitted(): ZeroOneBeta (1 | g) with sigma ~ x1 keeps the modes" begin
    d = _lm_data("laplace_sigma_formula_parity"; ints = [:y_nb])
    form = bf(@formula(y_zoib ~ x1 + (1 | g)), @formula(sigma ~ x1))
    ref = _lm_ref("laplace_sigma_formula_parity", "zob_sigma_x1")
    @testset "Laplace" begin
        fit = drm(form, DRModels.ZeroOneBeta(); data = d, marginal = :Laplace)
        @test abs(loglik(fit) - ref["loglik"]) <= 1e-4
        cond = fitted(fit)
        @test length(cond) == nobs(fit) == length(ref["fitted"])
        @test maximum(abs.(cond .- Float64.(ref["fitted"]))) < _LM_TOL
        zero = fitted(fit; re = :zero)
        @test zero == fit.means[:mu]
        @test maximum(abs.(cond .- zero)) > 1e-3
        @test residuals(fit) ≈ d.y_zoib .- cond
        # the stored contribution solves the inner problem group by group (one value per group)
        th = fit.theta; eta = fit.scales[:cond_fit_eta_re]
        eta0 = th[1] .+ th[2] .* d.x1; phi = exp.(-2 .* (th[3] .+ th[4] .* d.x1))
        zoi = 1 / (1 + exp(-th[5])); sdb = exp(th[7])
        lbeta(m, p, y) = (a = m * p; b = (1 - m) * p; (a - 1) * log(y) + (b - 1) * log1p(-y) +
                          loggamma(p) - loggamma(a) - loggamma(b))
        for g in unique(d.g)
            idx = findall(==(g), d.g)
            @test maximum(abs.(eta[idx] .- eta[idx[1]])) < 1e-12
            lp(b) = sum(lbeta(1 / (1 + exp(-(eta0[i] + b))), phi[i], d.y_zoib[i]) for i in idx if 0 < d.y_zoib[i] < 1) -
                    0.5 * (b / sdb)^2
            h = 1e-6
            @test abs((lp(eta[idx[1]] + h) - lp(eta[idx[1]] - h)) / (2h)) < 1e-5
        end
    end
    @testset "default route (adaptive quadrature) keeps the modes too" begin
        fit = drm(form, DRModels.ZeroOneBeta(); data = d)
        @test fit.marginal === :LA
        cond = fitted(fit); zero = fitted(fit; re = :zero)
        @test length(cond) == nobs(fit)
        @test all(isfinite, cond)
        @test maximum(abs.(cond .- zero)) > 1e-3
        # a mean with the same Beta component: conditional minus zero is explained by the logit shift only
        eta = fit.scales[:cond_fit_eta_re]
        bm, zo, co = fit.scales[:beta_mu], fit.scales[:zoi], fit.scales[:coi]
        @test cond ≈ (1 .- zo) .* (1 ./ (1 .+ exp.(-(log.(bm ./ (1 .- bm)) .+ eta)))) .+ zo .* co
    end
end

# Conditional fitted values by Newton's method on the joint log density of one group's random intercept, with
# the group SD exp(th[3] + th[4] gx) and the mean eta0 = th[1] + th[2] x1.
function _lm_sdreg_conditional(th, d, link)
    pf = link === :probit ? (e -> 0.5 * erfc(-e / sqrt(2))) : (e -> 1 / (1 + exp(-e)))
    out = zeros(length(d.x1))
    for g in unique(d.g)
        idx = findall(==(g), d.g)
        tau = exp(th[3] + th[4] * d.gx[idx[1]])
        eta0 = th[1] .+ th[2] .* d.x1[idx]
        lp(b) = sum(d.s[idx] .* log.(pf.(eta0 .+ b)) .+ d.fl[idx] .* log.(1 .- pf.(eta0 .+ b))) - 0.5 * b^2 / tau^2
        b = 0.0
        for _ in 1:100
            step = ForwardDiff.derivative(lp, b) / ForwardDiff.derivative(x -> ForwardDiff.derivative(lp, x), b)
            b -= step
            abs(step) < 1e-14 && break
        end
        out[idx] .= pf.(eta0 .+ b)
    end
    return out
end

@testset "fitted(): Binomial (1 | g) with sd(g) ~ gx keeps the modes" begin
    d = _lm_data("binomial_sd_fitted_parity"; ints = [:s, :fl])
    for (id, link) in (("binom_logit_sd_gx", :logit), ("binom_probit_sd_gx", :probit))
        @testset "$id" begin
            ref = _lm_ref("binomial_sd_fitted_parity", id)
            fit = drm(bf(@formula(cbind(s, fl) ~ x1 + (1 | g)), @formula(sd(g) ~ gx)),
                      DRModels.Binomial(link); data = d)
            cond = fitted(fit)
            @test length(cond) == nobs(fit) == length(ref["fitted"])
            @test maximum(abs.(_lm_sdreg_conditional(Float64.(ref["theta"]), d, link) .- Float64.(ref["fitted"]))) < _LM_TOL
            @test maximum(abs.(cond .- _lm_sdreg_conditional(fit.theta, d, link))) < 1e-8
            @test maximum(abs.(cond .- Float64.(ref["fitted"]))) < 1e-3
            zero = fitted(fit; re = :zero)
            @test zero == fit.means[:mu]
            @test maximum(abs.(cond .- zero)) > 1e-3
            @test residuals(fit) ≈ fit.obs[:mu] .- cond
        end
    end
end
