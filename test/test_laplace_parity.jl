using DRModels, Test
include(joinpath(@__DIR__, "laplace_parity_helpers.jl"))

# `marginal = :Laplace` on routes whose default integrator is adaptive quadrature: one independent term
# (intercept or slope) with or without `sd(g, coef(.)) ~ ...`, independent terms on Binomial, and
# correlated blocks. Reference: drmTMB 0.8.1 (ML) on the committed gap-map data
# (test/fixtures/laplace_parity/, made by the committed gen_ref.R; test/fixtures/nongauss_indep/ for
# slope and independent cells). The rule was written before the single-term reference values were
# read (docs/dev-log/2026-11-09-laplace-parity.md): |dlogLik| <= 1e-4, mean/sigma/nu coefficients within
# 1e-3 of R's standard error, random-effect log SDs within 5e-3. Without the keyword nothing changes.

const _LP_REF = lp_reference()

function _lp_check(id)
    fit = lp_fit(id)
    c = lp_compare(id, fit, _LP_REF[id])
    @test fit.converged
    @test fit.marginal === :Laplace
    @test c.dll <= 1e-4
    @test c.fe <= 1e-3
    @test c.sdlog <= 5e-3 || c.boundary
    return fit
end

@testset "single (1 | g), $id" for id in [
        "poisson|ordinary|lab0|q1|sd-none|cm0|ML", "poisson|ordinary|lab0|q1|sd-one|cm0|ML", "poisson|ordinary|lab0|q1|sd-z|cm0|ML",
        "nb2|ordinary|lab0|q1|sd-one|cm0|ML", "nb2|ordinary|lab0|q1|sd-z|cm0|ML",
        "gamma|ordinary|lab0|q1|sd-one|cm0|ML", "gamma|ordinary|lab0|q1|sd-z|cm0|ML",
        "binomial|ordinary|lab0|q1|sd-none|cm0|ML", "binomial|ordinary|lab0|q1|sd-one|cm0|ML", "binomial|ordinary|lab0|q1|sd-z|cm0|ML",
        "beta|ordinary|lab0|q1|sd-one|cm0|ML", "beta|ordinary|lab0|q1|sd-z|cm0|ML",
        "student|ordinary|lab0|q1|sd-none|cm0|ML", "student|ordinary|lab0|q1|sd-z|cm0|ML",
        "tweedie|ordinary|lab0|q1|sd-z|cm0|ML"]
    _lp_check(id)
end

@testset "single (0 + x1 | g), $id" for id in [
        "poisson|slope|lab0|q1|sd-none|cm0|ML", "poisson|slope|lab0|q1|sd-one|cm0|ML", "poisson|slope|lab0|q1|sd-z|cm0|ML",
        "nb2|slope|lab0|q1|sd-one|cm0|ML", "gamma|slope|lab0|q1|sd-z|cm0|ML",
        "binomial|slope|lab0|q1|sd-one|cm0|ML", "binomial|slope|lab0|q1|sd-z|cm0|ML",
        "beta|slope|lab0|q1|sd-one|cm0|ML", "student|slope|lab0|q1|sd-none|cm0|ML", "student|slope|lab0|q1|sd-z|cm0|ML",
        "tweedie|slope|lab0|q1|sd-none|cm0|ML"]
    _lp_check(id)
end

@testset "Binomial independent terms, $id" for id in [
        "binomial|indep|lab0|q2|sd-one|cm0|ML", "binomial|indep|lab0|q2|sd-last|cm0|ML", "binomial|indep|lab0|q3|sd-first|cm0|ML",
        "binomial|indep|lab0|q2|sd-none|cm0|ML"]
    haskey(_LP_REF, id) && _lp_check(id)
end

@testset "correlated block (1 + x1 + ... | g), $id" for id in [
        "poisson|corr|lab0|q2|sd-none|cm0|ML", "nb2|corr|lab0|q2|sd-none|cm0|ML",
        "binomial|corr|lab0|q2|sd-none|cm0|ML", "binomial|corr|lab0|q2|sd-one|cm0|ML",
        "binomial|corr|lab0|q2|sd-last|cm0|ML", "binomial|corr|lab0|q2|sd-first|cm0|ML"]
    _lp_check(id)
end

@testset "boundary correlation: binomial|corr|lab0|q2|sd-last (Julia 1.13.3 regression)" begin
    # drmTMB ends this fit with rho = -0.99999994 (a flat boundary ridge). Before the correlation-parameter clamp
    # (`_binomial_bounded_partial`) the Laplace objective was rounding noise at |atanh rho| > ~12 and, on Julia 1.13.3,
    # the optimiser ran there (not converged, logLik -255.17978, 887 iterations); 1.10.12 stopped at atanh rho = -8.29.
    id = "binomial|corr|lab0|q2|sd-last|cm0|ML"
    fit = lp_fit(id)
    @test fit.converged
    @test abs(loglik(fit) - _LP_REF[id].loglik) <= 1e-4
    @test abs(only(fit.scales[:corpars])) > 0.99999          # at the boundary, as in drmTMB
    @test abs(coef(fit, :recov)[end]) < 12                   # the optimiser did not run into the noise region
end

@testset "Poisson correlated block with a mean offset, c ~ x + offset(logb) + (1 + x | g)" begin
    # Reference: drmTMB 0.8.1 (ML) on data_poisson_offset_corr.csv, made by gen_data_offset_corr.jl / gen_ref_offset_corr.R.
    dir = joinpath(@__DIR__, "fixtures", "laplace_parity")
    m, _ = readdlm(joinpath(dir, "data_poisson_offset_corr.csv"), ','; header = true)
    d = (c = Float64.(m[:, 1]), x = Float64.(m[:, 2]), logb = Float64.(m[:, 3]), g = String.(m[:, 4]))
    R, _ = readdlm(joinpath(dir, "r_reference_offset_corr.csv"), ','; header = true)
    fit = drm(bf(@formula(c ~ x + offset(logb) + (1 + x | g))), DRModels.Poisson(); data = d, marginal = :Laplace)
    @test fit.converged
    @test fit.marginal === :Laplace
    @test abs(loglikelihood(fit) - Float64(R[1, 4])) <= 1e-4
    @test maximum(abs.(coef(fit, :mu) .- Float64.(R[1:2, 2])) ./ Float64.(R[1:2, 3])) <= 1e-3
    @test maximum(abs.([coef(fit, :sd_intercept)[1], coef(fit, :sd_slope)[1]] .- log.(Float64.(R[3:4, 2])))) <= 5e-3
    @test abs(coef(fit, :recov)[1] - atanh(Float64(R[5, 2]))) <= 5e-3
end

@testset "default behaviour is unchanged (matches the base commit values)" begin
    # Values recorded on f6dab1698 (this branch's base) with no keyword; the keyword must not touch them.
    base = Dict(
        "poisson|ordinary|lab0|q1|sd-one|cm0|ML" => (-744.6180810484707,
            [0.4553791285172813, 0.45012852108177276, 0.3189147139849959, -0.15439680244810508, -0.542047699722908]),
        "binomial|ordinary|lab0|q1|sd-none|cm0|ML" => (-256.0589052810935,
            [0.5180096148049461, 0.1891360371963989, 0.24876933412250615, -0.29102788315956096, -0.7539064608322625]),
        "gamma|slope|lab0|q1|sd-z|cm0|ML" => (-617.039073555438,
            [0.5597932916913014, 0.413645609757624, 0.3232743553671103, -0.19169301553519638, -0.2634725285563333, -1.15172654667056, 0.3880128006899087]))
    for (id, (ll, theta)) in base
        fit = lp_fit(id; marginal = :LA)
        @test fit.marginal === :LA
        @test isapprox(loglik(fit), ll; rtol = 1e-9)
        @test isapprox(collect(fit.theta), theta; rtol = 1e-8, atol = 1e-9)
    end
end

@testset "the Laplace and quadrature objectives differ as documented" begin
    id = "poisson|ordinary|lab0|q1|sd-one|cm0|ML"
    la, lap = lp_fit(id; marginal = :LA), lp_fit(id)
    @test lap.marginal === :Laplace && la.marginal === :LA
    @test loglik(lap) < loglik(la) - 0.05            # Poisson `(1 | g)`: Laplace sits below the exact (quadrature) value
end

@testset "shapes the keyword does not cover stay refused" begin
    d = ngi_data("poisson")
    @test_throws r"`marginal = :Laplace` with `sd\(\)` formulas covers independent" drm(
        bf(@formula(y ~ x1 + (1 | g)), @formula(sd(g, sigma) ~ z)), Poisson(); data = d, marginal = :Laplace)
    @test_throws ArgumentError drm(bf(@formula(y ~ x1 + (1 | g)), @formula(sd(g) ~ z)), Binomial(link = :probit);
                                   data = ngi_data("binomial"), marginal = :Laplace)
end
