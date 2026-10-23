# test_parity_harness.jl — ALWAYS-ON smoke test for the R-parity MACHINERY.
#
# This is the CI verification that compare.jl + loadfixture.jl actually work: no
# R, no committed drmTMB fixtures, fast. It does NOT claim drmTMB parity — it
# proves the harness round-trips a DRModels.jl fit against itself (self-consistency)
# and, critically, that it DETECTS a deliberate mismatch. Real drmTMB parity is
# the gated `runparity.jl` suite (DRM_PARITY_TESTS=1), fed by maintainer-
# generated fixtures (GENERATING.md).

using DRModels
using Test, Random
using TOML
using LinearAlgebra: I

include("parity/compare.jl")
include("parity/loadfixture.jl")

@testset "parity harness (self-consistency, no R)" begin
    # --- tiny Gaussian location–scale fit (mirrors test_family_accessor.jl) ----
    Random.seed!(20260603)
    n = 400
    x = randn(n)
    y = 0.5 .- 0.8 .* x .+ exp.(-0.3 .+ 0.4 .* x) .* randn(n)
    data = (; y, x)

    fit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian(); data = data)

    # --- drm_coef_named bridge: block layout → flat names --------------------
    named = drm_coef_named(fit)
    @test Set(keys(named)) ==
        Set(["mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x"])

    # --- self-consistency expected (round-trips against itself) --------------
    self_expected = ParityExpected(;
        family = "gaussian",
        coef = named,
        loglik = loglik(fit),
        aic = -2 * loglik(fit) + 2 * dof(fit),
        df = dof(fit),
        n = nobs(fit))

    res_ok = compare_fit(fit, self_expected)
    @test res_ok.passed == true
    @test isempty(res_ok.failures)

    # --- perturbed expected: harness MUST detect the mismatch ----------------
    bad_coef = copy(named)
    bad_coef["mu_x"] += 1.0                       # bump one coefficient
    perturbed = ParityExpected(;
        family = "gaussian",
        coef = bad_coef,
        loglik = loglik(fit) + 10.0,              # bump loglik
        aic = -2 * loglik(fit) + 2 * dof(fit),    # leave aic at the true value
        df = dof(fit),
        n = nobs(fit))

    res_bad = compare_fit(fit, perturbed)
    @test res_bad.passed == false
    @test !isempty(res_bad.failures)
    # The offending quantities are named in the failure strings.
    @test any(f -> occursin("coef[mu_x]", f), res_bad.failures)
    @test any(f -> occursin("loglik", f), res_bad.failures)

    # --- round-trip the loaders (covers loadfixture.jl) ----------------------
    mktempdir() do dir
        # Write expected.toml in the README format.
        open(joinpath(dir, "expected.toml"), "w") do io
            println(io, "[fit]")
            println(io, "family = \"gaussian\"")
            println(io, "formula = \"y ~ 1 + x; sigma ~ 1 + x\"")
            println(io, "loglik = ", loglik(fit))
            println(io, "aic = ", -2 * loglik(fit) + 2 * dof(fit))
            println(io, "df = ", dof(fit))
            println(io, "n = ", nobs(fit))
            println(io)
            println(io, "[coef]")
            for k in sort!(collect(keys(named)))
                println(io, "\"", k, "\" = ", named[k])
            end
        end
        # Write data.csv with a header row.
        open(joinpath(dir, "data.csv"), "w") do io
            println(io, "y,x")
            for i in 1:n
                println(io, y[i], ",", x[i])
            end
        end

        loaded_exp = load_expected(dir)
        @test loaded_exp.family == "gaussian"
        @test loaded_exp.df == dof(fit)
        @test loaded_exp.n == nobs(fit)

        loaded_data = load_data(dir)
        @test Set(keys(loaded_data)) == Set((:y, :x))
        @test length(loaded_data.y) == n

        # Re-fit from the loaded data and compare against the loaded expected.
        refit = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian();
            data = loaded_data)
        res_rt = compare_fit(refit, loaded_exp)
        @test res_rt.passed == true
    end

    # --- compare_bridge: flattened drm_bridge Dict vs same self-expected ------
    bridged = drm_bridge(; formula = "y ~ 1 + x; sigma ~ 1 + x",
                         family = "gaussian", data = data)
    res_br = compare_bridge(bridged, self_expected)
    @test res_br.passed == true
    @test isempty(res_br.failures)

    bad_bridge = copy(bridged)
    bad_bridge["coef"] = Dict(bridged["coef"])  # shallow copy of nested dict
    bad_bridge["coef"] = Dict{String,Float64}(String(k) => Float64(v) for (k, v) in bridged["coef"])
    bad_bridge["coef"]["mu_x"] += 1.0
    res_br_bad = compare_bridge(bad_bridge, self_expected)
    @test res_br_bad.passed == false
    @test any(f -> occursin("coef[mu_x]", f), res_br_bad.failures)
end

@testset "parity harness: group-level covariance (ranef) comparison" begin
    # Synthetic location–scale DrmFit with a KNOWN 2×2 Λ (no engine run) — Λ from
    # the log-Cholesky :recov block [log L11, log L22, L21]. This CI-verifies the
    # new [ranef] comparison + loader without R or a real fit.
    L11 = 0.5; L22 = sqrt(0.15); L21 = 0.1            # → Λ = [0.25 0.05; 0.05 0.16]
    theta = [0.3, 0.2, log(L11), log(L22), L21]
    fit = DrmFit(NegBinomial2(),
                 [:mu => 1:1, :sigma => 2:2, :recov => 3:5],
                 [:mu => ["(Intercept)"], :sigma => ["(Intercept)"],
                  :recov => ["species:L11", "species:L22", "species:L21"]],
                 theta, Matrix{Float64}(I, 5, 5), -10.0, 10, true,
                 Dict(:mu => fill(1.0, 10)), Dict(:mu => fill(1.0, 10)),
                 Dict(:sigma => fill(1.0, 10)))

    @test vc(fit)[:species] ≈ [0.25 0.05; 0.05 0.16] atol = 1e-12

    df = dof(fit); aic = -2 * (-10.0) + 2 * df
    re = Dict("sd_mu" => 0.5, "sd_sigma" => 0.4, "cor" => 0.25)
    self_re = ParityExpected(; family = "nbinom2",
        coef = Dict("mu_(Intercept)" => 0.3, "sigma_(Intercept)" => 0.2),
        loglik = -10.0, aic = aic, df = df, n = 10,
        ranef_group = "species", ranef = re)
    res_ok = compare_fit(fit, self_re)
    @test res_ok.passed
    @test isempty(res_ok.failures)

    # Perturb the correlation → the harness MUST detect it.
    bad_re = ParityExpected(; family = "nbinom2",
        coef = Dict("mu_(Intercept)" => 0.3, "sigma_(Intercept)" => 0.2),
        loglik = -10.0, aic = aic, df = df, n = 10,
        ranef_group = "species",
        ranef = Dict("sd_mu" => 0.5, "sd_sigma" => 0.4, "cor" => 0.80))
    res_bad = compare_fit(fit, bad_re)
    @test !res_bad.passed
    @test any(f -> occursin("ranef[cor]", f), res_bad.failures)

    # Loader round-trips a [ranef] block.
    mktempdir() do dir
        open(joinpath(dir, "expected.toml"), "w") do io
            println(io, "[fit]\nfamily = \"nbinom2\"\nformula = \"y ~ 1; sigma ~ 1\"")
            println(io, "loglik = -10.0\naic = ", aic, "\ndf = ", df, "\nn = 10\n")
            println(io, "[coef]\n\"mu_(Intercept)\" = 0.3\n\"sigma_(Intercept)\" = 0.2\n")
            println(io, "[ranef]\ngroup = \"species\"\nsd_mu = 0.5\nsd_sigma = 0.4\ncor = 0.25")
        end
        le = load_expected(dir)
        @test le.ranef_group == "species"
        @test le.ranef["cor"] ≈ 0.25
        @test compare_fit(fit, le).passed
    end
end

@testset "load_data: a numeric column survives a mixed-type CSV (#474)" begin
    # readdlm returns a Matrix{Any} the moment ANY column in the file is
    # non-numeric, so a per-column eltype check on that Matrix{Any} sees `Any`
    # even where a column's own values are all numbers. Guard against a
    # numeric predictor silently becoming categorical when it shares a CSV
    # with a string column.
    mktempdir() do dir
        open(joinpath(dir, "data.csv"), "w") do io
            println(io, "y,x,grp")
            println(io, "1.1,10,a")
            println(io, "1.9,10,b")
            println(io, "3.2,20,a")
            println(io, "3.8,20,b")
            println(io, "5.3,30,a")
            println(io, "4.7,30,b")
            println(io, "7.1,40,a")
            println(io, "6.9,40,b")
        end
        dat = load_data(dir)
        @test dat.x isa Vector{Float64}
        @test dat.x == [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 40.0, 40.0]

        # Modelled as continuous: one "mu_x" coefficient, not a dummy per
        # distinct value (what a Vector{Any} column produces).
        fit = drm(bf(@formula(y ~ 1 + x)), Gaussian(); data = dat)
        named = drm_coef_named(fit)
        @test Set(keys(named)) == Set(["mu_(Intercept)", "mu_x", "sigma_(Intercept)"])
    end
end
