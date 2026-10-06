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
