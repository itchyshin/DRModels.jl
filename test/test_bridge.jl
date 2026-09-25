using DRModels
using Test, Random, LinearAlgebra

@testset "drm_bridge primitive R boundary" begin
    Random.seed!(20260608)
    n = 80
    x = range(-1, 1; length = n)
    y = 0.3 .+ 0.8 .* x .+ exp.(-0.4 .+ 0.2 .* x) .* randn(n)
    data = (; y = collect(y), x = collect(x))

    native = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = data)
    bridged = drm_bridge(;
        formula = "y ~ x; sigma ~ x",
        family = "gaussian",
        data = data,
    )

    @test bridged["family"] == "gaussian"
    @test bridged["coef_names"] ==
        ["mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_x"]
    @test bridged["coefficients"] ≈ coef(native)
    @test bridged["vcov"] ≈ vcov(native)
    @test bridged["loglik"] ≈ loglik(native)
    @test bridged["aic"] ≈ aic(native)
    @test bridged["bic"] ≈ bic(native)
    @test bridged["df"] == dof(native)
    @test bridged["nobs"] == nobs(native)
    @test bridged["converged"] == is_converged(native)
    @test bridged["fitted"] ≈ fitted(native)
    @test bridged["residuals"] ≈ residuals(native)
    @test bridged["sigma"] ≈ sigma(native)
    @test isempty(bridged["corpairs"])

    # Per-observation distributional parameters — what drmTMB's post-fit surface
    # consumes. `fitted_distribution()` (the hub for qq_plot / worm_plot /
    # centile_chart / exceedance) builds its d/p/q from
    # `fitted_distribution_params()`, which needs one response-scale column per
    # dpar and aborts if the columns have inconsistent lengths.
    dpars = bridged["dpars"]
    @test Set(keys(dpars)) == Set(["mu", "sigma"])
    @test length(unique(length.(values(dpars)))) == 1
    @test all(length(v) == bridged["nobs"] for v in values(dpars))
    @test all(all(isfinite, v) for v in values(dpars))
    @test dpars["mu"] ≈ fitted(native)
    @test dpars["sigma"] ≈ sigma(native)

    # A dpar is not `fitted()`. For zero-one-inflated beta, drmTMB's `mu` dpar is
    # the INTERIOR beta component mean plogis(eta_mu) -- it feeds
    # drm_beta_shapes(mu, sigma) -- while `fitted()` is the unconditional mean
    # (1 - zoi) * mu + zoi * coi. Both lie in (0, 1), so shipping the wrong one
    # yields a wrong density with no error anywhere. Guard the distinction.
    # Local RNG: `Random.seed!` here would reseed the GLOBAL stream and silently
    # change the data every later test in this file draws.
    rngz = MersenneTwister(20260814)
    nz = 200
    xz = randn(rngz, nz)
    yz = clamp.(0.5 .+ 0.15 .* randn(rngz, nz), 0.0, 1.0)
    yz[1:20] .= 0.0            # real boundary mass, so zoi > 0 and the
    yz[21:40] .= 1.0           # unconditional mean genuinely differs from `mu`
    zob = drm_bridge(; formula = "y ~ x", family = "zeroonebeta",
                     data = (; y = yz, x = xz))
    zdp = zob["dpars"]
    @test Set(keys(zdp)) == Set(["mu", "sigma", "zoi", "coi"])   # drmTMB's dpar set
    @test !haskey(zdp, "beta_mu")                                # not a drmTMB dpar name
    @test all(0 .< zdp["mu"] .< 1)
    # the dpar must be the interior beta mean, NOT the unconditional fitted mean
    unconditional = (1 .- zdp["zoi"]) .* zdp["mu"] .+ zdp["zoi"] .* zdp["coi"]
    @test zob["fitted"] ≈ unconditional
    @test !(zdp["mu"] ≈ zob["fitted"])
    @test length(unique(length.(values(zdp)))) == 1

    # `trials` is per-row CONTEXT, not a dpar. drmTMB's binomial dpar set is
    # `mu` alone; `fitted_distribution_params()` attaches `params$trials` itself.
    rngb = MersenneTwister(5)
    nb, ntr = 120, 10
    xb = randn(rngb, nb)
    prb = 1 ./ (1 .+ exp.(-(0.3 .+ 0.6 .* xb)))
    sb = Float64.([count(_ -> rand(rngb) < prb[i], 1:ntr) for i in 1:nb])
    fb = Float64.(ntr .- sb)
    bino = drm_bridge(; formula = "cbind(s, f) ~ x", family = "binomial",
                      data = (; s = sb, f = fb, x = xb))
    @test Set(keys(bino["dpars"])) == Set(["mu"])     # drmTMB's binomial dpar set
    @test !haskey(bino["dpars"], "trials")            # context, not a dpar
    @test haskey(bino, "trials")                      # ships as its own key
    @test length(bino["trials"]) == bino["nobs"]
    @test all(bino["trials"] .== ntr)
    # families with no binomial denominator must not carry the key at all
    @test !haskey(bridged, "trials")

    # Meta-analysis: drmTMB's meta `sigma` dpar is the HETEROGENEITY tau, while
    # `scales[:sigma]` holds the TOTAL sqrt(V + tau^2) that simulate() needs.
    # Shipping the total as `sigma` AND a V_known would double-count the sampling
    # variance. Both are recovered exactly, with no change to sigma()'s contract.
    rngm = MersenneTwister(2)
    nm = 60
    xm = randn(rngm, nm)
    vm = 0.05 .+ 0.4 .* rand(rngm, nm)
    taum = 0.35
    ym = 0.4 .+ 0.8 .* xm .+ sqrt.(vm .+ taum^2) .* randn(rngm, nm)
    dm = (; y = ym, x = xm, v = vm)
    meta = drm_bridge(; formula = Dict(:mu => "y ~ x + meta_V(v)", :sigma => "sigma ~ 1"),
                      family = "gaussian", data = dm)
    @test haskey(meta, "V_known")
    @test length(meta["V_known"]) == nm
    @test maximum(abs.(meta["V_known"] .- vm)) < 1e-10      # recovered EXACTLY
    # the dpar is tau (constant here), NOT the per-row total
    @test length(unique(round.(meta["dpars"]["sigma"], digits=10))) == 1
    @test !(meta["dpars"]["sigma"] ≈ meta["sigma"])          # dpar != total SD
    @test all(meta["dpars"]["sigma"] .< meta["sigma"])       # tau < sqrt(V + tau^2)
    # sigma()'s public contract is untouched: still a bare vector
    @test meta["sigma"] isa AbstractVector
    # and a non-meta fit gains nothing
    @test !haskey(bridged, "V_known")

    # Fresh-data dpars. `fitted_distribution(object, newdata = ...)` needs
    # `predict_parameters(..., type = "response")` for EVERY dpar; the
    # julia-engine vignette records the gap this closes ("fresh-data Julia
    # prediction is currently limited to location parameters").
    @test !haskey(bridged, "dpars_newdata")           # absent unless asked for
    nd = (; x = [-1.0, 0.0, 1.5])
    withnd = drm_bridge(; formula = "y ~ x; sigma ~ x", family = "gaussian",
                        data = data, newdata = nd)
    ndp = withnd["dpars_newdata"]
    @test Set(keys(ndp)) == Set(["mu", "sigma"])
    @test all(length(v) == 3 for v in values(ndp))
    @test all(ndp["sigma"] .> 0)                      # response scale, not log
    @test all(isfinite, ndp["mu"])
    # in-sample block is unchanged by asking for fresh rows
    @test withnd["dpars"]["mu"] ≈ bridged["dpars"]["mu"]

    keyed = drm_bridge(;
        formula = Dict(:mu => "y ~ x", :sigma => "sigma ~ x"),
        family = "gaussian",
        data = Dict("y" => collect(y), "x" => collect(x)),
    )
    @test keyed["coefficients"] ≈ bridged["coefficients"]
    @test keyed["loglik"] ≈ bridged["loglik"]

    @test_throws ArgumentError drm_bridge(;
        formula = Dict(:sigma => "sigma ~ x"),
        family = "gaussian",
        data = data,
    )
    @test_throws ArgumentError drm_bridge(;
        formula = "y ~ x; sigma ~ x",
        family = "not_a_real_family",
        data = data,
    )

    y2 = -0.2 .+ 0.4 .* x .+ 0.5 .* randn(n)
    bdata = (; y1 = collect(y), y2 = collect(y2), x = collect(x))
    bnative = drm(
        bf(;
            mu1 = @formula(y1 ~ x),
            mu2 = @formula(y2 ~ x),
            sigma1 = @formula(sigma1 ~ 1),
            sigma2 = @formula(sigma2 ~ 1),
            rho12 = @formula(rho12 ~ 1),
        ),
        Gaussian();
        data = bdata,
    )
    bbridged = drm_bridge(;
        formula = Dict(
            :mu1 => "y1 ~ x",
            :mu2 => "y2 ~ x",
            :sigma1 => "sigma1 ~ 1",
            :sigma2 => "sigma2 ~ 1",
            :rho12 => "rho12 ~ 1",
        ),
        family = "biv_gaussian",
        data = bdata,
    )
    @test bbridged["family"] == "biv_gaussian"
    @test bbridged["coef_names"] == [
        "mu1_(Intercept)", "mu1_x", "mu2_(Intercept)", "mu2_x",
        "sigma1_(Intercept)", "sigma2_(Intercept)", "rho12_(Intercept)",
    ]
    @test bbridged["coefficients"] ≈ coef(bnative)
    @test bbridged["vcov"] ≈ vcov(bnative)
    @test bbridged["loglik"] ≈ loglik(bnative)
    @test bbridged["aic"] ≈ aic(bnative)
    @test bbridged["bic"] ≈ bic(bnative)
    @test bbridged["df"] == dof(bnative)
    @test bbridged["nobs"] == nobs(bnative)
    @test bbridged["converged"] == is_converged(bnative)
    @test bbridged["fitted"]["mu1"] ≈ fitted(bnative)[:mu1]
    @test bbridged["fitted"]["mu2"] ≈ fitted(bnative)[:mu2]
    @test bbridged["residuals"]["mu1"] ≈ residuals(bnative)[:mu1]
    @test bbridged["residuals"]["mu2"] ≈ residuals(bnative)[:mu2]
    @test bbridged["sigma"]["sigma1"] ≈ sigma(bnative)[:sigma1]
    @test bbridged["sigma"]["sigma2"] ≈ sigma(bnative)[:sigma2]
    @test bbridged["corpairs"] ≈ corpairs(bnative)

    newick = "((sp_1:0.3,sp_2:0.3):0.3,(sp_3:0.3,sp_4:0.3):0.3);"
    empty!(DRModels._BRIDGE_TREE_CACHE)
    cached_phy1 = DRModels._bridge_tree(newick)
    cached_phy2 = DRModels._bridge_tree(newick)
    @test cached_phy1 === cached_phy2
    @test length(DRModels._BRIDGE_TREE_CACHE) == 1

    G = 16
    m = 4
    phy = random_balanced_tree(G; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C))
    K = C ./ (d * d')
    species = repeat(1:G, inner = m)
    xphy = randn(G * m)
    uphy = 0.6 .* (cholesky(Symmetric(K)).L * randn(G))
    yphy = 0.1 .+ 0.5 .* xphy .+ uphy[species] .+ 0.4 .* randn(G * m)
    pdata = (; y = yphy, x = xphy, species = species)
    pnative = drm(
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
        Gaussian();
        data = pdata,
        tree = phy,
    )
    pbridged = drm_bridge(;
        formula = Dict(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ 1"),
        family = "gaussian",
        data = pdata,
        tree = phy,
    )
    @test pbridged["family"] == "gaussian"
    @test pbridged["coefficients"] ≈ coef(pnative)
    # Phylo vcov can contain NaN blocks for random-effect rows; isequal treats NaN==NaN.
    @test isequal(pbridged["vcov"], Matrix{Float64}(vcov(pnative)))
    @test pbridged["loglik"] ≈ loglik(pnative)
    @test pbridged["aic"] ≈ aic(pnative)
    @test pbridged["bic"] ≈ bic(pnative)
    @test pbridged["df"] == dof(pnative)
    @test pbridged["nobs"] == nobs(pnative)
    @test pbridged["converged"] == is_converged(pnative)
    @test pbridged["fitted"] ≈ fitted(pnative)
    @test pbridged["residuals"] ≈ residuals(pnative)
    @test pbridged["sigma"] ≈ sigma(pnative)
    @test isempty(pbridged["corpairs"])

    Gls = 10
    mls = 3
    phyls = random_balanced_tree(Gls; branch_length = 0.25)
    Cls = sigma_phy_dense(phyls; σ²_phy = 1.0)
    Kls = Cls ./ (sqrt.(diag(Cls)) * sqrt.(diag(Cls))')
    species_ls = repeat(1:Gls, inner = mls)
    xls = randn(Gls * mls)
    uls = 0.35 .* (cholesky(Symmetric(Kls)).L * randn(Gls))
    wls = 0.25 .* (cholesky(Symmetric(Kls)).L * randn(Gls))
    yls = 0.1 .+ 0.5 .* xls .+ uls[species_ls] .+
          exp.(-0.4 .+ wls[species_ls]) .* randn(Gls * mls)
    lsdata = (; y = yls, x = xls, species = species_ls)
    lsformula = Dict(
        :mu => "y ~ x + phylo(1 | species)",
        :sigma => "sigma ~ phylo(1 | species)",
    )
    lsnative = drm(
        bf(@formula(y ~ x + phylo(1 | species)),
           @formula(sigma ~ phylo(1 | species))),
        Gaussian();
        data = lsdata,
        tree = phyls,
        phylo_coupled = true,
        g_tol = 1e-5,
    )
    lsbridged = drm_bridge(;
        formula = lsformula,
        family = "gaussian",
        data = lsdata,
        tree = phyls,
        options = Dict(:phylo_coupled => true, :g_tol => 1e-5),
    )
    @test any(startswith("recov_"), lsbridged["coef_names"])
    @test lsbridged["converged"] == true
    @test lsbridged["coefficients"] ≈ coef(lsnative)
    @test lsbridged["loglik"] ≈ loglik(lsnative)
    # Arc 2: the coupled block now fits REML (joint Laplace over the phylo effects
    # and both fixed-effect axes, as native drmTMB); it used to throw here.
    lsreml_native = drm(
        bf(@formula(y ~ x + phylo(1 | species)),
           @formula(sigma ~ phylo(1 | species))),
        Gaussian();
        data = lsdata,
        tree = phyls,
        phylo_coupled = true,
        method = :REML,
    )
    lsreml = drm_bridge(;
        formula = lsformula,
        family = "gaussian",
        data = lsdata,
        tree = phyls,
        options = Dict(:phylo_coupled => true, :method => "REML"),
    )
    @test estimation_method(lsreml_native) == :REML
    @test any(startswith("recov_"), lsreml["coef_names"])
    @test lsreml["estim_method"] == "REML"
    @test lsreml["coefficients"] ≈ coef(lsreml_native)
    @test lsreml["loglik"] ≈ loglik(lsreml_native)

    pprofile = drm_bridge_inference(;
        formula = Dict(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ 1"),
        family = "gaussian",
        data = pdata,
        tree = phy,
        method = "profile",
        level = 0.80,
    )
    @test pprofile["method"] == "profile"
    @test pprofile["param"] == "resd"
    @test pprofile["status"] == "profile"
    @test pprofile["used"] == 1
    @test pprofile["failed"] == 0
    @test isfinite(pprofile["lower"])
    @test isfinite(pprofile["upper"])

    pbootstrap = drm_bridge_inference(;
        formula = Dict(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ 1"),
        family = "gaussian",
        data = pdata,
        tree = phy,
        method = "bootstrap",
        level = 0.80,
        B = 3,
        seed = 20260609,
    )
    @test pbootstrap["method"] == "bootstrap"
    @test pbootstrap["param"] == "resd"
    @test pbootstrap["attempted"] == 3
    @test pbootstrap["used"] >= 2
    @test pbootstrap["failed"] <= 1
    @test isfinite(pbootstrap["lower"])
    @test isfinite(pbootstrap["upper"])

    # ---------------------------------------------------------------------
    # #475: `parm` kwarg on the public `drm_bridge_inference` surface. Before
    # this, drmTMB PR #1080 could only reach an ordinary `fixef:<dpar>:<coef>`
    # target by calling DRModels.jl's underscore-prefixed marshalling internals by
    # qualified name (`DRModels._bridge_data`, `_bridge_formula`, `_bridge_family`,
    # `_bridge_fit`, `_bridge_inference_flatten`, ...) -- a rename here would
    # silently break the R bridge. `parm = "fixef:mu:x"` closes that gap.
    # ---------------------------------------------------------------------

    fx_profile = drm_bridge_inference(;
        formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
        method = "profile", level = 0.90, parm = "fixef:mu:x",
    )
    @test fx_profile["method"] == "profile"
    @test fx_profile["param"] == "mu"
    @test fx_profile["coef"] == "x"
    @test isfinite(fx_profile["lower"])
    @test isfinite(fx_profile["upper"])
    @test fx_profile["lower"] < fx_profile["estimate"] < fx_profile["upper"]

    fx_boot = drm_bridge_inference(;
        formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
        method = "bootstrap", level = 0.90, B = 25, seed = 20260609,
        parm = "fixef:mu:x",
    )
    @test fx_boot["method"] == "bootstrap"
    @test fx_boot["param"] == "mu"
    @test fx_boot["coef"] == "x"
    @test isfinite(fx_boot["lower"])
    @test isfinite(fx_boot["upper"])

    # A malformed or nonexistent target is refused loudly, not silently
    # mislabelled -- mirrors `_bridge_pick_sd_row`'s discipline.
    @test_throws ArgumentError drm_bridge_inference(;
        formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
        method = "profile", parm = "mu:x",             # missing the "fixef:" tag
    )
    @test_throws ArgumentError drm_bridge_inference(;
        formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
        method = "profile", parm = "fixef:mu:not_a_coef",
    )

    # Equivalence check: the SAME `fixef:mu:x` target reached the OLD way --
    # by calling the underscore-prefixed marshalling internals directly by
    # qualified name, exactly as drmTMB PR #1080's Julia glue does today --
    # must produce IDENTICAL numbers to the new public `parm` route. No
    # underscore-prefixed function is called on the `fx_profile`/`fx_boot`
    # side above; this block is only the OLD-route reference computation.
    dat_i = DRModels._bridge_data(data)
    bundle_i, dat_i = DRModels._bridge_formula("y ~ x; sigma ~ x", "gaussian", dat_i)
    fam_i = DRModels._bridge_family("gaussian")
    opts_i = DRModels._bridge_options(Dict{String,Any}())
    fit_i = DRModels._bridge_fit(bundle_i, fam_i, dat_i; tree = nothing, K = nothing,
                            A = nothing, coords = nothing, options = opts_i)

    result_i = DRModels.profile_result(fit_i; level = 0.90, threads = false, parm = :mu)
    row_i = only(filter(r -> r.param === :mu && r.coef == "x", result_i.ci))
    internal_profile = DRModels._bridge_inference_flatten(
        row_i; method = "profile", status = "profile",
        attempted = result_i.attempted, used = result_i.used, failed = result_i.failed,
        elapsed = result_i.elapsed, threaded = result_i.threaded,
        worker_threads = result_i.worker_threads, julia_threads = result_i.julia_threads,
        blas_threads = result_i.blas_threads, message = "profile_result completed")
    @test fx_profile["estimate"] == internal_profile["estimate"]
    @test fx_profile["lower"] == internal_profile["lower"]
    @test fx_profile["upper"] == internal_profile["upper"]

    rng_i = Random.MersenneTwister(20260609)
    result_ib = DRModels.bootstrap_result(fit_i; data = dat_i, B = 25, level = 0.90, rng = rng_i,
                                     tree = nothing, threads = false, failures = :skip,
                                     check_converged = true, algorithm = :auto, g_tol = 1e-8)
    row_ib = only(filter(r -> r.param === :mu && r.coef == "x", result_ib.summary))
    internal_boot = DRModels._bridge_inference_flatten(
        row_ib; method = "bootstrap",
        status = result_ib.used >= 2 ? "bootstrap" : "bootstrap_unavailable",
        attempted = result_ib.attempted, used = result_ib.used, failed = result_ib.failed,
        elapsed = result_ib.elapsed, threaded = result_ib.threaded,
        worker_threads = result_ib.worker_threads, julia_threads = result_ib.julia_threads,
        blas_threads = result_ib.blas_threads,
        message = "$(result_ib.used)/$(result_ib.attempted) successful refits")
    @test fx_boot["estimate"] == internal_boot["estimate"]
    @test fx_boot["lower"] == internal_boot["lower"]
    @test fx_boot["upper"] == internal_boot["upper"]

    # Pin: the pre-existing SD-target path (no `parm`) is BYTE-IDENTICAL to
    # before this change. Reconstruct the same phylogenetic-SD profile/
    # bootstrap result independently through the primitives `drm_bridge_
    # inference` itself calls, and compare to the public no-`parm` call above
    # (`pprofile` / `pbootstrap`, on the same `pdata`/`phy` fixture).
    dat_sd = DRModels._bridge_data(pdata)
    bundle_sd, dat_sd = DRModels._bridge_formula(
        Dict(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ 1"),
        "gaussian", dat_sd)
    fam_sd = DRModels._bridge_family("gaussian")
    opts_sd = DRModels._bridge_options(Dict{String,Any}())
    opts_sd[:profile_ci] = true   # `drm_bridge_inference` sets this for the SD path
    tree_sd = DRModels._bridge_tree(phy)
    fit_sd = DRModels._bridge_fit(bundle_sd, fam_sd, dat_sd; tree = tree_sd, K = nothing,
                             A = nothing, coords = nothing, options = opts_sd)

    result_sd = DRModels.profile_result(fit_sd; level = 0.80, threads = false,
                                   parm = [:resd_sigma, :resd, :resd_mu])
    row_sd = only(filter(r -> r.param in (:resd_sigma, :resd, :resd_mu), result_sd.ci))
    internal_sd_profile = DRModels._bridge_inference_flatten(
        row_sd; method = "profile", status = "profile",
        attempted = result_sd.attempted, used = result_sd.used, failed = result_sd.failed,
        elapsed = result_sd.elapsed, threaded = result_sd.threaded,
        worker_threads = result_sd.worker_threads, julia_threads = result_sd.julia_threads,
        blas_threads = result_sd.blas_threads, message = "profile_result completed")
    @test pprofile["estimate"] == internal_sd_profile["estimate"]
    @test pprofile["lower"] == internal_sd_profile["lower"]
    @test pprofile["upper"] == internal_sd_profile["upper"]
    @test pprofile["param"] == internal_sd_profile["param"]

    rng_sd = Random.MersenneTwister(20260609)
    result_sdb = DRModels.bootstrap_result(fit_sd; data = dat_sd, B = 3, level = 0.80,
                                      rng = rng_sd, tree = tree_sd, threads = false,
                                      failures = :skip, check_converged = true,
                                      algorithm = :auto, g_tol = 1e-8)
    row_sdb = only(filter(r -> r.param in (:resd_sigma, :resd, :resd_mu), result_sdb.summary))
    internal_sd_boot = DRModels._bridge_inference_flatten(
        row_sdb; method = "bootstrap",
        status = result_sdb.used >= 2 ? "bootstrap" : "bootstrap_unavailable",
        attempted = result_sdb.attempted, used = result_sdb.used, failed = result_sdb.failed,
        elapsed = result_sdb.elapsed, threaded = result_sdb.threaded,
        worker_threads = result_sdb.worker_threads, julia_threads = result_sdb.julia_threads,
        blas_threads = result_sdb.blas_threads,
        message = "$(result_sdb.used)/$(result_sdb.attempted) successful refits")
    @test pbootstrap["estimate"] == internal_sd_boot["estimate"]
    @test pbootstrap["lower"] == internal_sd_boot["lower"]
    @test pbootstrap["upper"] == internal_sd_boot["upper"]
    @test pbootstrap["param"] == internal_sd_boot["param"]
end

@testset "keyed univariate `nu` is not a bivariate discriminator (#1090)" begin
    # drmTMB's Workflow G robust-student cell marshals bf(y ~ x, sigma ~ 1,
    # nu ~ 1) as KEYED parts. `:nu` sat in _BRIDGE_BIVARIATE_KEYS (it is
    # threaded through for biv_student), so a plain univariate Student tripped
    # the bivariate branch and died demanding mu1/mu2 — while univariate
    # Student legitimately owns a `nu` formula. Only mu1/mu2/sigma1/sigma2/
    # rho12 discriminate bivariate; `nu` must not.
    Random.seed!(20260828)
    n = 160
    x = randn(n)
    y = 0.4 .+ 0.7 .* x .+ 0.5 .* randn(n)
    d = (; y, x)

    r = DRModels.drm_bridge(; formula = "mu = y ~ x; sigma = sigma ~ 1; nu = nu ~ 1",
                       family = "student", data = d)
    @test r["family"] == "student"
    @test isfinite(Float64(r["loglik"]))
    # mu intercept + mu x + sigma + nu — the keyed nu reached the UNIVARIATE
    # Student bundle rather than tripping the bivariate branch.
    @test length(r["coef_names"]) >= 4

    # Bivariate Student still routes bivariate (mu1/mu2 discriminate, nu rides).
    y2 = 0.1 .+ 0.4 .* x .+ 0.6 .* randn(n)
    d2 = (; y1 = y, y2 = y2, x)
    r2 = DRModels.drm_bridge(; formula = "mu1 = y1 ~ x; mu2 = y2 ~ x; sigma1 = sigma1 ~ 1; sigma2 = sigma2 ~ 1; nu = nu ~ 1; rho12 = rho12 ~ 1",
                        family = "biv_student", data = d2)
    @test isfinite(Float64(r2["loglik"]))
end

@testset "bridge admits skew_normal (drmTMB skew_normal(): mu, sigma, nu)" begin
    # Before this case `_bridge_family("skew_normal")` threw
    # `drm_bridge: unsupported family \`skew_normal\`` even though
    # `SkewNormal()` is a native family (src/skewnormal.jl) — the R engine gate
    # could not admit the family because the Julia side had no tag for it.
    @test DRModels._bridge_family("skew_normal") isa SkewNormal
    @test DRModels._bridge_family("skewnormal") isa SkewNormal

    # Same DGP shape as drmTMB's test-skew-normal-location-scale.R (moment
    # parameterisation: mu = mean, sigma = SD, nu = slant alpha), so the
    # bridged fit exercises every dpar the R side sends.
    Random.seed!(20260608)
    n = 300
    x = randn(n); z = randn(n)
    mu = 0.20 .+ 0.45 .* x
    sigma = exp.(-0.35 .+ 0.18 .* z)
    alpha = 1.6
    delta = alpha / sqrt(1 + alpha^2)
    ms = delta * sqrt(2 / pi)
    omega = sigma ./ sqrt(1 - ms^2)
    xi = mu .- omega .* ms
    y = xi .+ omega .* (delta .* abs.(randn(n)) .+ sqrt(1 - delta^2) .* randn(n))
    d = (; y, x, z)

    r = DRModels.drm_bridge(; formula = "mu = y ~ x; sigma = sigma ~ z; nu = nu ~ 1",
                       family = "skew_normal", data = d)
    @test r["family"] == "skew_normal"
    @test r["estim_method"] == "ML"
    @test isfinite(Float64(r["loglik"]))
    @test r["coef_names"] == ["mu_(Intercept)", "mu_x", "sigma_(Intercept)", "sigma_z", "nu_(Intercept)"]
    # The bridged fit IS the native fit: same formula, same family, same numbers.
    native = drm(bf(@formula(y ~ x), @formula(sigma ~ z), @formula(nu ~ 1)), SkewNormal(); data = d)
    @test Float64(r["loglik"]) ≈ loglik(native) atol = 1e-10
    @test vec(Float64.(r["coefficients"])) ≈ vcat(coef(native, :mu), coef(native, :sigma), coef(native, :nu)) atol = 1e-10

    # coef_labels echo (design 258 §7.1): R's labels for all three dpars are
    # echoed verbatim, so the drmTMB side can validate the round trip.
    labelled = DRModels.drm_bridge(; formula = "mu = y ~ x; sigma = sigma ~ z; nu = nu ~ 1",
                              family = "skew_normal", data = d,
                              options = Dict{String,Any}("coef_labels" => Dict(
                                  "mu" => ["(Intercept)", "x"],
                                  "sigma" => ["(Intercept)", "z"],
                                  "nu" => ["(Intercept)"])))
    @test labelled["coef_names"] == r["coef_names"]
    @test labelled["raw_coef_names"] == r["coef_names"]
end
