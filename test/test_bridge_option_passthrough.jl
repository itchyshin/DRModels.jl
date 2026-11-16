using DRModels
using Test, Random, LinearAlgebra

@testset "drm_bridge option passthrough + gradient exposure" begin
    Random.seed!(20260904)

    # --- (1) unknown option key errors loudly, by name ------------------------
    n = 60
    x = collect(range(-1, 1; length = n))
    y = 0.2 .+ 0.5 .* x .+ 0.3 .* randn(n)
    data = (; y = y, x = x)
    err = try
        drm_bridge(; formula = "y ~ x", family = "gaussian", data = data,
                   options = Dict(:not_a_real_option => 1))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not_a_real_option", sprint(showerror, err))

    # A known key still reaches the fitter and does not throw.
    @test_nowarn drm_bridge(; formula = "y ~ x", family = "gaussian", data = data,
                            options = Dict(:g_tol => 1e-6))

    # A genuinely loose vs. tight `g_tol` demonstrably changes the reported
    # iteration count — proof the option reaches the optimiser, not just that
    # it is accepted without error.
    loose = drm_bridge(; formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
                       options = Dict(:g_tol => 1.0))
    tight = drm_bridge(; formula = "y ~ x; sigma ~ x", family = "gaussian", data = data,
                       options = Dict(:g_tol => 1e-12))
    @test loose["iterations"] < tight["iterations"]

    # --- (2) a control option demonstrably changes the fit ---------------------
    # `sparse = true` forces the sparse phylogenetic LSS engine (vs. dense
    # default); both must recover the same fit (same likelihood) but the
    # native `sparse` kwarg must actually reach `drm(...)` through the bridge.
    G = 40
    m = 3
    phy = random_balanced_tree(G; branch_length = 0.25)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    K = C ./ (sqrt.(diag(C)) * sqrt.(diag(C))')
    species = repeat(1:G, inner = m)
    xls = randn(G * m)
    u = 0.3 .* (cholesky(Symmetric(K)).L * randn(G))
    yls = 0.1 .+ 0.4 .* xls .+ u[species] .+ 0.3 .* randn(G * m)
    lsdata = (; y = yls, x = xls, species = species)
    lsformula = Dict(:mu => "y ~ x + phylo(1 | species)", :sigma => "sigma ~ 1")

    native_dense = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                        Gaussian(); data = lsdata, tree = phy, sparse = false)
    native_sparse = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                         Gaussian(); data = lsdata, tree = phy, sparse = true)

    bridged_dense = drm_bridge(; formula = lsformula, family = "gaussian",
                               data = lsdata, tree = phy,
                               options = Dict(:sparse => false))
    bridged_sparse = drm_bridge(; formula = lsformula, family = "gaussian",
                                data = lsdata, tree = phy,
                                options = Dict(:sparse => true))

    @test bridged_dense["coefficients"] ≈ coef(native_dense)
    @test bridged_sparse["coefficients"] ≈ coef(native_sparse)
    @test bridged_dense["loglik"] ≈ bridged_sparse["loglik"] atol=1e-4

    # The `sparse` value type is validated, not silently coerced: a non-boolean
    # value errors (proves `options[:sparse]` genuinely reaches `Bool(...)` in
    # `_bridge_fit` rather than being dropped on the floor).
    @test_throws MethodError drm_bridge(; formula = lsformula, family = "gaussian",
                                        data = lsdata, tree = phy,
                                        options = Dict(:sparse => "yes"))

    # --- (3) the gradient key: present, finite, named, matches native nllgrad --
    g_data = (;
        y = 0.4 .+ 0.3 .* randn(200) .+ repeat(0.2 .* randn(20), inner = 10) .+
            repeat(0.15 .* randn(10), outer = 20),
        x = randn(200),
        g1 = repeat(1:20, inner = 10),
        g2 = repeat(1:10, outer = 20),
    )
    native_multi = drm(bf(@formula(y ~ x + (1 | g1) + (1 | g2)), @formula(sigma ~ 1)),
                        Gaussian(); data = g_data)
    bridged_multi = drm_bridge(;
        formula = Dict(:mu => "y ~ x + (1 | g1) + (1 | g2)", :sigma => "sigma ~ 1"),
        family = "gaussian", data = g_data,
    )
    @test haskey(bridged_multi, "gradient")
    @test haskey(bridged_multi, "gradient_names")
    g = bridged_multi["gradient"]
    gn = bridged_multi["gradient_names"]
    @test all(isfinite, g)
    @test length(g) == length(gn) == length(bridged_multi["coef_names"])
    @test gn == bridged_multi["coef_names"]

    native_g = zeros(length(coef(native_multi)))
    native_multi.nllgrad(native_g, coef(native_multi))
    @test g ≈ native_g atol=1e-10

    # --- (4) gradient key absent on a route with no stored gradient ------------
    plain_bridged = drm_bridge(; formula = "y ~ x; sigma ~ x", family = "gaussian",
                               data = data)
    @test !haskey(plain_bridged, "gradient")
    @test !haskey(plain_bridged, "gradient_names")
end

@testset "drm_bridge names an option the target fitter does not accept" begin
    Random.seed!(20261031)
    n = 60
    x = randn(n)
    data = (; y = rand(1:5, n), x = x)
    err = try
        drm_bridge(; formula = "y ~ x", family = "poisson", data = data,
                   options = Dict(:sparse => true))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("`sparse`", sprint(showerror, err))
    # `wald` is not served by drm_bridge_inference; the text says so and lists accepted methods.
    err2 = try
        drm_bridge_inference(; formula = "y ~ x", family = "gaussian",
                             data = (; y = randn(n), x = x), method = "wald")
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("\"profile\"", sprint(showerror, err2))
    @test occursin("\"bootstrap\"", sprint(showerror, err2))
end

@testset "drm_bridge forwards q4_f_reltol to the q=4 L-BFGS stopping rule" begin
    Random.seed!(20261108)
    p, m = 12, 3
    phy = random_balanced_tree(p; branch_length = 0.3)
    LC = cholesky(Symmetric(sigma_phy_dense(phy; σ²_phy = 1.0))).L
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    U = LC * randn(p, 4) .* 0.4
    dat = (; y1 = 0.3 .* x .+ U[species, 1] .+ 0.5 .* randn(n),
           y2 = 0.2 .* x .+ U[species, 2] .+ 0.5 .* randn(n), x, species)
    fml = Dict(:mu1 => "y1 ~ x + phylo(1 | species)", :mu2 => "y2 ~ x + phylo(1 | species)",
               :sigma1 => "sigma1 ~ 1 + phylo(1 | species)", :sigma2 => "sigma2 ~ 1 + phylo(1 | species)",
               :rho12 => "rho12 ~ 1")
    fitq(extra) = DRModels.drm_bridge(; formula = fml, family = "biv_gaussian", data = dat, tree = phy,
                                      options = merge(Dict(:q4_iterations => 300, :q4_vcov => false), extra))
    default = fitq(Dict())
    tight = fitq(Dict(:q4_f_reltol => 1e-12))
    loose = fitq(Dict(:q4_f_reltol => 1e-1))
    # A tight tolerance reproduces the default optimum; a very loose one stops the optimiser
    # measurably earlier, which proves the key reaches `fit_q4_sparse_tmb`.
    @test tight["loglik"] ≈ default["loglik"] atol = 1e-6
    @test loose["loglik"] < default["loglik"] - 1.0
end

@testset "drm_bridge refuses q4_f_reltol on routes that never read it" begin
    Random.seed!(20261108)
    p, m = 8, 2
    phy = random_balanced_tree(p; branch_length = 0.3)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    dat = (; y1 = randn(n), y2 = randn(n), x, species)
    q4 = Dict(:mu1 => "y1 ~ x + phylo(1 | species)", :mu2 => "y2 ~ x + phylo(1 | species)",
              :sigma1 => "sigma1 ~ 1 + phylo(1 | species)", :sigma2 => "sigma2 ~ 1 + phylo(1 | species)",
              :rho12 => "rho12 ~ 1")
    q2 = Dict(:mu1 => "y1 ~ x + phylo(1 | species)", :mu2 => "y2 ~ x + phylo(1 | species)",
              :sigma1 => "sigma1 ~ 1", :sigma2 => "sigma2 ~ 1", :rho12 => "rho12 ~ 1")
    resid = Dict(:mu1 => "y1 ~ x", :mu2 => "y2 ~ x", :sigma1 => "sigma1 ~ 1",
                 :sigma2 => "sigma2 ~ 1", :rho12 => "rho12 ~ 1")
    refused(fml, opts) = try
        DRModels.drm_bridge(; formula = fml, family = "biv_gaussian", data = dat, tree = phy, options = opts)
        nothing
    catch e
        e
    end
    e1 = refused(q4, Dict(:q4_f_reltol => 1e-3, :method => "REML"))
    @test e1 isa ArgumentError && occursin("q4_f_reltol", sprint(showerror, e1)) &&
          occursin("q=4 REML route", sprint(showerror, e1))
    e2 = refused(q2, Dict(:q4_f_reltol => 1e-3))
    @test e2 isa ArgumentError && occursin("q=2 structured route", sprint(showerror, e2))
    e3 = refused(resid, Dict(:q4_f_reltol => 1e-3))
    @test e3 isa ArgumentError && occursin("residual-only route", sprint(showerror, e3))
    # the default value is never refused
    @test DRModels._bridge_check_q4_f_reltol_route(nothing, nothing, 1e-7, :ML) === nothing
end
