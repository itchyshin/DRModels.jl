# test_cov_1125n5_comparison_labels.jl — lrt / AIC label helpers (src/comparison.jl).
using DRModels
using Test, Random

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : rethrow()
end

function _tiny_gaussian_fit()
    Random.seed!(1125)
    n = 40
    g = repeat(1:4, outer = 10)
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ randn(n) .* 0.3
    return drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = (; y, x, g))
end

@testset "cov 1125n5: _cholesky_role and _string_counts" begin
    @test DRModels._cholesky_role("recov:L11") === :sd
    @test DRModels._cholesky_role("recov:L21") === :cor
    @test DRModels._cholesky_role("rho12:atanh") === :cor
    @test DRModels._cholesky_role("sd_phylo:log_sd_g") === :sd
    @test DRModels._cholesky_role("mu:x") === :unknown
    d = DRModels._string_counts(["a", "b", "a"])
    @test d["a"] == 2 && d["b"] == 1
    @test DRModels._same_string_multiset(["a", "b"], ["b", "a"])
    extra = DRModels._added_string_labels(["a"], ["a", "b", "c"])
    @test extra == ["b", "c"] || sort(extra) == ["b", "c"]
    @test DRModels._added_string_labels(["a", "x"], ["a", "b"]) === nothing
end

@testset "cov 1125n5: _push_vc_labels! recov block" begin
    sd = String[]
    cor = String[]
    names = ["recov:L11", "recov:L21", "recov:L22"]
    DRModels._push_vc_labels!(sd, cor, :recov, names, 3)
    @test length(sd) == 2
    @test length(cor) == 1
    @test all(occursin("recov:", s) for s in vcat(sd, cor))
end

@testset "cov 1125n5: _sd_and_correlation_counts on fit" begin
    fit = _tiny_gaussian_fit()
    nsd, ncor = DRModels._sd_and_correlation_counts(fit)
    @test nsd ≥ 1
    @test ncor ≥ 0
    slopes = DRModels._logsd_slope_names(fit)
    @test slopes isa Vector{String}
    sd_lab, cor_lab = DRModels._vc_sd_and_cor_labels(fit)
    @test sd_lab isa Vector{String}
    @test cor_lab isa Vector{String}
end

@testset "cov 1125n5: _canonical_sd_label" begin
    fit = _tiny_gaussian_fit()
    for n in ["resd:g", "resd:(Intercept)", "sd:g: (Intercept)"]
        lbl = DRModels._canonical_sd_label(fit, n)
        @test !isempty(lbl)
    end
    @test DRModels._canonical_sd_label(fit, "mu:x") == "mu:x"
end

@testset "cov 1125n5: _obs_mask_vals" begin
    y = [1.0, NaN, 3.0]
    m1, v1 = DRModels._obs_mask_vals(y)
    @test m1 == BitVector([true, false, true])
    @test v1 == [1.0, 3.0]
    y2 = [1.0, 2.0, 3.0]
    m2, v2 = DRModels._obs_mask_vals(y2)
    @test !DRModels._same_obs_rowset(m1, v1, m2, v2)
    m1b, v1b = DRModels._obs_mask_vals(y)
    @test DRModels._same_obs_rowset(m1, v1, m1b, v1b)
    @test !DRModels._same_obs_rowset(m1, v1, m2, [1.0, 2.0, 4.0])
end

@testset "cov 1125n5: _sentinel_compare_guard" begin
    fit = _tiny_gaussian_fit()
    bad = DrmFit(fit.family, fit.blocks, fit.coefnames, fit.theta, fit.vcov, -1e18, fit.nobs,
        fit.converged, fit.means, fit.obs, fit.scales, fit.formula, fit.nll, fit.nllgrad,
        fit.ranef, fit.estim_method, fit.reml_loglik, fit.ml_loglik, fit.marginal,
        fit.phylo_penalty, fit.penalty, fit.iterations, fit.phylo_scale, fit.uncertainty,
        fit.geometry)
    msg = _msg(() -> DRModels._sentinel_compare_guard(bad, fit, "lrtest"))
    @test occursin("log-likelihood", msg) || occursin("likelihood", msg)
end