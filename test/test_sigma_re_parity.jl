# Random effects on `sigma` outside the single-Gaussian-intercept case: parity with drmTMB 0.8.1.
# Eight cells (src/sigma_ranef_blocks_laplace.jl): LogNormal / Gamma / NB2 / BetaBinomial with
# `sigma ~ x1 + (1 | g)`; Gaussian `sigma ~ x1 + (0 + x1 | g)`, `(1 + x1 | g)`, `(1 | g) + (1 | g2)`,
# and mean `(1 | g)` with sigma `(1 | g2)` on a different group.
# Reference: fixtures/sigma_re/sigma_re_r_reference.R (drmTMB 0.8.1, Laplace = its only method here),
# data fixtures/sigma_re/sigma_re_data.csv (written by sre_write_csv in sigma_re_fixture.jl and
# stored so the numbers do not depend on a sampler version). Full R output:
# fixtures/sigma_re/r_reference_output.txt. Bounds fixed by the lane before the first run:
#   |logLik diff| <= 1e-4; fixed effects within 1e-3 R SE; log SD (and atanh correlation) within 5e-3;
#   fixed-effect SEs within 1e-3 relative.
using DRModels, Test, DelimitedFiles, LinearAlgebra, Random, Statistics, ForwardDiff
import Distributions

const _SRE_M, _SRE_H = readdlm(joinpath(@__DIR__, "fixtures", "sigma_re", "sigma_re_data.csv"), ','; header = true)
const SRE = NamedTuple(Symbol(h) => (h in ("g", "g2", "sb", "flb") ? Int.(_SRE_M[:, i]) : Float64.(_SRE_M[:, i]))
                       for (i, h) in enumerate(vec(_SRE_H)))

# name => (ll, mu, mu_se, sigma, sigma_se, logsd (R order), eta_cor)
const SRE_R = Dict(
    :lognormal => (ll = -504.0736307571, mu = [0.5071954922, 0.4542692070], mu_se = [0.03523356860, 0.03745750515],
                   sg = [-0.4158751844, 0.2219704111], sg_se = [0.07755092320, 0.04963383599], logsd = [-0.8875009884], cor = nothing),
    :gamma => (ll = -398.2419203255, mu = [0.4774233985, 0.3823135584], mu_se = [0.03696332492, 0.03892301171],
               sg = [-0.4279746251, 0.3101361277], sg_se = [0.06653618080, 0.04262039662], logsd = [-1.0815234687], cor = nothing),
    :nb2 => (ll = -692.2687668787, mu = [1.0276509470, 0.2663046444], mu_se = [0.05391189847, 0.06357823781],
             sg = [-0.3458512376, 0.1754026926], sg_se = [0.10226764680, 0.08572249961], logsd = [-1.1633274534], cor = nothing),
    :bb => (ll = -923.3590636071, mu = [0.1760137046, 0.4960090780], mu_se = [0.07215852639, 0.08214839804],
            sg = [-0.2453714275, 0.2531413691], sg_se = [0.06468175189, 0.05531834641], logsd = [-1.2207807267], cor = nothing),
    :gauss_slope0 => (ll = -339.6214955570, mu = [0.4681238993, 0.6027611431], mu_se = [0.03043063824, 0.01746198200],
                      sg = [-0.3945803604, 0.2518497279], sg_se = [0.04342752624, 0.10605857111], logsd = [-0.6151129688], cor = nothing),
    :gauss_slope => (ll = -322.3610041775, mu = [0.4748372035, 0.6041985893], mu_se = [0.02564436739, 0.01505302658],
                     sg = [-0.5025948232, 0.2695444948], sg_se = [0.07293302462, 0.09852571813],
                     logsd = [-1.0350924589, -0.7136924628], cor = 0.6153125805),
    :gauss_two => (ll = -339.4447990818, mu = [0.4909009192, 0.6020123780], mu_se = [0.03669374943, 0.03741723389],
                   sg = [-0.4383992056, 0.3583470994], sg_se = [0.09517807978, 0.05663770958],
                   logsd = [-0.9769384791, -1.7580398252], cor = nothing),
    :gauss_diff => (ll = -376.8993477936, mu = [0.5303526843, 0.6152148387], mu_se = [0.09201668753, 0.04805606262],
                    sg = [-0.3617177722, 0.2741875893], sg_se = [0.06070113547, 0.04811763691],
                    logsd = [-0.6516626244, -2.1179446944], cor = nothing))

const SRE_CELLS = Dict(
    :lognormal => (bf(@formula(y_ln ~ x1), @formula(sigma ~ x1 + (1 | g))), LogNormal()),
    :gamma => (bf(@formula(y_gam ~ x1), @formula(sigma ~ x1 + (1 | g))), Gamma()),
    :nb2 => (bf(@formula(y_nb ~ x1), @formula(sigma ~ x1 + (1 | g))), NegBinomial2()),
    :bb => (bf(@formula(cbind(sb, flb) ~ x1), @formula(sigma ~ x1 + (1 | g))), BetaBinomial()),
    :gauss_slope0 => (bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (0 + x1 | g))), Gaussian()),
    :gauss_slope => (bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (1 + x1 | g))), Gaussian()),
    :gauss_two => (bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (1 | g) + (1 | g2))), Gaussian()),
    :gauss_diff => (bf(@formula(y_gm ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2))), Gaussian()))

const SRE_FITS = Dict{Symbol,Any}()
sre_fit(cell) = get!(SRE_FITS, cell) do
    f, fam = SRE_CELLS[cell]
    drm(f, fam; data = SRE)
end

@testset "sigma random effects: parity with drmTMB 0.8.1" begin
    for cell in (:lognormal, :gamma, :nb2, :bb, :gauss_slope0, :gauss_slope, :gauss_two, :gauss_diff)
        @testset "$cell" begin
            r = SRE_R[cell]
            fit = sre_fit(cell)
            @test fit.converged
            @test abs(loglik(fit) - r.ll) <= 1e-4
            @test all(abs.(coef(fit, :mu) .- r.mu) .<= 1e-3 .* r.mu_se)
            @test all(abs.(coef(fit, :sigma) .- r.sg) .<= 1e-3 .* r.sg_se)
            @test all(abs.(coef(fit, :resd) .- r.logsd) .<= 5e-3)
            r.cor === nothing || @test abs(only(coef(fit, :cor)) - r.cor) <= 5e-3
            se = sqrt.(diag(vcov(fit)))
            @test all(abs.(se[1:2] ./ r.mu_se .- 1) .<= 1e-3)
            @test all(abs.(se[3:4] ./ r.sg_se .- 1) .<= 1e-3)
        end
    end
end

@testset "sigma random effects: per-observation kernels against ForwardDiff" begin
    k = DRModels._gsl_kernel
    for (fam, y) in ((LogNormal(), 1.7), (BetaBinomial(), (7.0, 20.0)), (BetaBinomial(), (0.0, 12.0)),
                     (BetaBinomial(), (12.0, 12.0)))
        for (lin, eta) in ((0.3, -0.4), (-1.1, 0.6), (0.0, 0.0))
            f(x) = k(fam, y, x[1], x[2])[1]
            g = ForwardDiff.gradient(f, [lin, eta]); H = ForwardDiff.hessian(f, [lin, eta])
            v = k(fam, y, lin, eta)
            @test v[2] ≈ g[1] rtol = 1e-8
            @test v[3] ≈ g[2] rtol = 1e-8
            @test v[4] ≈ H[1, 1] rtol = 1e-7
            @test v[5] ≈ H[1, 2] rtol = 1e-7
            @test v[6] ≈ H[2, 2] rtol = 1e-7
        end
    end
    # the BetaBinomial density against Distributions.jl
    v = k(BetaBinomial(), (7.0, 20.0), 0.3, -0.4)
    p = 1 / (1 + exp(-0.3)); phi = exp(0.8)
    @test v[1] ≈ -Distributions.logpdf(Distributions.BetaBinomial(20, p * phi, (1 - p) * phi), 7) rtol = 1e-10
    @test k(LogNormal(), 1.7, 0.3, -0.4)[1] ≈
          -Distributions.logpdf(Distributions.LogNormal(0.3, exp(-0.4)), 1.7) rtol = 1e-10
end

@testset "sigma random effects: post-fit paths" begin
    for cell in (:lognormal, :gamma, :nb2, :bb, :gauss_slope, :gauss_two, :gauss_diff)
        @testset "$cell" begin
            fit = sre_fit(cell)
            f, fam = SRE_CELLS[cell]
            nd = (x1 = [-1.0, 0.0, 1.0],)
            lin = coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* nd.x1
            # drmTMB's LogNormal prediction is the log-scale mu, as everywhere else in this package
            inv = fam isa Union{Gaussian,LogNormal} ? identity : fam isa BetaBinomial ? (z -> 1 / (1 + exp(-z))) : exp
            @test predict(fit, nd) ≈ inv.(lin) atol = 1e-10
            @test predict(fit, nd; type = :link) ≈ lin atol = 1e-10
            sp = predict_parameters(fit, nd)
            @test sp[:sigma] ≈ exp.(coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* nd.x1) atol = 1e-10
            # population-level simulate (random effects set to zero): means follow the fixed part
            Y0 = simulate(fit; data = SRE, nsim = 3, rng = MersenneTwister(1), re = :zero)
            @test size(Y0) == (320, 3) && all(isfinite, Y0)
            # fresh random effects
            Y = simulate(fit; data = SRE, nsim = 3, rng = MersenneTwister(2))
            @test size(Y) == (320, 3) && all(isfinite, Y)
            # refit keeps the terms (update) and the bootstrap runs without failures
            fit2 = update(fit, f; data = SRE)
            @test abs(loglik(fit2) - loglik(fit)) < 1e-6
            boot = bootstrap_result(fit; data = SRE, B = 2, rng = MersenneTwister(3), failures = :skip)
            @test isempty(boot.failures)
            @test !isempty(re_sd(fit))
            @test !isempty(ranef(fit))
        end
    end
end

# Independent check that the default simulate draws FRESH sigma random effects: the per-row variance
# of the draws must follow the marginal law, and the zero-effect draws must miss it.
@testset "sigma random effects: marginal simulate matches the marginal variance" begin
    nsim = 4000
    rowvar(Y) = vec(var(Y; dims = 2))
    # Gaussian, two sigma intercepts: Var(y_i) = exp(2 eta_i + 2 (s_g^2 + s_h^2))
    fit = sre_fit(:gauss_two)
    eta = coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* SRE.x1
    s2 = sum(exp.(2 .* coef(fit, :resd)))
    ref = exp.(2 .* eta .+ 2 * s2)
    Y = simulate(fit; data = SRE, nsim, rng = MersenneTwister(11))
    @test abs(mean(rowvar(Y)) / mean(ref) - 1) < 0.15
    Y0 = simulate(fit; data = SRE, nsim, rng = MersenneTwister(12), re = :zero)
    # zero-effect draws miss the marginal variance by the factor exp(-2 (s_g^2 + s_h^2)) = 0.71 here
    # (the first draft of this bar said 0.6, which was a wrong guess of that factor; 0.8 = 20% miss)
    @test mean(rowvar(Y0)) / mean(ref) < 0.8
    # NB2: Var(y) = mu + mu^2 exp(2 eta + 2 s^2)
    fit = sre_fit(:nb2)
    mu = exp.(coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* SRE.x1)
    eta = coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* SRE.x1
    ref = mu .+ mu .^ 2 .* exp.(2 .* eta .+ 2 * exp(2 * only(coef(fit, :resd))))
    Y = simulate(fit; data = SRE, nsim, rng = MersenneTwister(13))
    @test abs(mean(rowvar(Y)) / mean(ref) - 1) < 0.2
    @test abs(mean(Y) / mean(mu) - 1) < 0.05
    # Gamma: Var(y) = mu^2 exp(2 eta + 2 s^2)
    fit = sre_fit(:gamma)
    mu = exp.(coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* SRE.x1)
    eta = coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* SRE.x1
    ref = mu .^ 2 .* exp.(2 .* eta .+ 2 * exp(2 * only(coef(fit, :resd))))
    Y = simulate(fit; data = SRE, nsim, rng = MersenneTwister(14))
    @test abs(mean(rowvar(Y)) / mean(ref) - 1) < 0.25
    # LogNormal: log y has mean mu and variance exp(2 eta + 2 s^2)
    fit = sre_fit(:lognormal)
    mu = coef(fit, :mu)[1] .+ coef(fit, :mu)[2] .* SRE.x1
    eta = coef(fit, :sigma)[1] .+ coef(fit, :sigma)[2] .* SRE.x1
    ref = exp.(2 .* eta .+ 2 * exp(2 * only(coef(fit, :resd))))
    Y = log.(simulate(fit; data = SRE, nsim, rng = MersenneTwister(15)))
    @test abs(mean(rowvar(Y)) / mean(ref) - 1) < 0.1
    @test abs(mean(Y) - mean(mu)) < 0.05
end

@testset "sigma random effects: shapes that are not implemented refuse by name" begin
    msg(f) = try; f(); ""; catch e; sprint(showerror, e); end
    # Count / positive families: a mean random effect together with a sigma one, a sigma slope and two sigma
    # terms keep the family's own refusal.
    for (fm, fam) in ((bf(@formula(y_nb ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2))), NegBinomial2()),
                      (bf(@formula(y_gam ~ x1), @formula(sigma ~ x1 + (0 + x1 | g))), Gamma()),
                      (bf(@formula(y_ln ~ x1), @formula(sigma ~ x1 + (1 | g) + (1 | g2))), LogNormal()))
        @test occursin("only the mean formula may carry a random effect", msg(() -> drm(fm, fam; data = SRE)))
    end
    # Gaussian: an intercept and a slope on one sigma group, a slope plus an intercept on another, two sigma terms plus a mean term
    for fm in (bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (1 | g) + (0 + x1 | g))),
               bf(@formula(y_g ~ x1), @formula(sigma ~ x1 + (0 + x1 | g) + (1 | g2))),
               bf(@formula(y_g ~ x1 + (1 | g)), @formula(sigma ~ x1 + (1 | g2) + (1 | g))))
        @test !isempty(msg(() -> drm(fm, Gaussian(); data = SRE)))
    end
    # offsets and non-default marginals are refused, never dropped
    @test occursin("offset", msg(() -> drm(bf(@formula(y_nb ~ x1 + offset(x1)), @formula(sigma ~ x1 + (1 | g))), NegBinomial2(); data = SRE)))
    @test !isempty(msg(() -> drm(bf(@formula(y_nb ~ x1), @formula(sigma ~ x1 + (1 | g))), NegBinomial2(); data = SRE, marginal = :VA)))
end

# Missing responses: as on the other random-effect routes the rows are dropped with a warning, and the
# result equals fitting the complete rows.
@testset "sigma random effects: missing responses are dropped with a warning" begin
    for (cell, col) in ((:gamma, :y_gam), (:gauss_two, :y_g))
        f, fam = SRE_CELLS[cell]
        miss = collect(1:17:320)
        y = Vector{Union{Missing,Float64}}(getproperty(SRE, col)); y[miss] .= missing
        dm = merge(SRE, NamedTuple{(col,)}((y,)))
        keep = setdiff(1:320, miss)
        dc = map(v -> v[keep], SRE)
        fitm = @test_logs (:warn, r"dropped") match_mode = :any drm(f, fam; data = dm)
        fitc = drm(f, fam; data = dc)
        @test nobs(fitm) == length(keep)
        @test abs(loglik(fitm) - loglik(fitc)) < 1e-6
        @test coef(fitm) ≈ coef(fitc) atol = 1e-5
    end
end

# Quantile residuals and fitted values on these fits are population-level: they use the stored population
# mean and sigma (random effects set to zero), the convention of the single Gaussian sigma route.
@testset "sigma random effects: stored mean and sigma are population-level" begin
    fit = sre_fit(:gamma)
    Xs = hcat(ones(320), SRE.x1)
    @test fit.scales[:sigma] ≈ exp.(Xs * coef(fit, :sigma)) atol = 1e-10
    @test fit.means[:mu] ≈ exp.(Xs * coef(fit, :mu)) atol = 1e-10
    @test all(isfinite, residuals(fit; type = :quantile))
end
