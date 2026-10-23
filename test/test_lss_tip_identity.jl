# Phylogenetic LSS must index covariance and SD covariates by tree-tip identity,
# rather than by the first tree label encountered in the observation rows.

using DRModels
using Test, LinearAlgebra

function _lss_identity_tree()
    labels = ["oak", "beech", "cedar", "elm", "fir", "gum"]
    # All tips have height two, but their shared-path covariance is asymmetric:
    # oak is separate; beech/cedar and elm/fir/gum form distinct clades.
    edges = Tuple{Int,Int,Float64}[
        (10, 1, 2.0), (10, 7, 1.0), (7, 2, 1.0), (7, 3, 1.0),
        (10, 8, 1.0), (8, 4, 1.0), (8, 9, 0.5), (9, 5, 0.5), (9, 6, 0.5),
    ]
    DRModels.make_phy(edges, length(labels); root_index = 10, leaf_names = labels)
end

function _lss_identity_fixture(; order = [5, 1, 6, 3, 4, 2],
                                numeric_species = false,
                                missing_tip = nothing)
    phy = _lss_identity_tree()
    labels = copy(phy.leaf_names)
    p, m = length(labels), 8
    tree_species = repeat(1:p, inner = m)
    rows = vcat([findall(==(tip), tree_species) for tip in order]...)
    species_idx = tree_species[rows]
    z_tip = [-1.1, -0.4, 0.2, 0.7, 1.3, 1.8]
    x_tree = [0.65 * sin(0.41 * i) + 0.12 * cos(0.17 * i) for i in eachindex(tree_species)]
    study_tree = [mod1(i, 4) for i in eachindex(tree_species)]
    K = DRModels._phylo_correlation(phy)
    α_phy = exp.(-0.25 .+ 0.42 .* z_tip)
    u = α_phy .* (cholesky(Symmetric(K)).L * [0.32, -0.21, 0.17, 0.08, -0.14, 0.25])
    b_study = [0.22, -0.17, 0.09, -0.12]
    y_tree = 0.45 .+ 0.37 .* x_tree .+ u[tree_species] .+ b_study[study_tree] .+
        exp.(-1.05 .+ 0.16 .* x_tree) .* (0.8 .* cos.(0.73 .* eachindex(tree_species)))
    y = copy(y_tree[rows])
    if missing_tip !== nothing
        y = Vector{Union{Missing,Float64}}(y)
        y[species_idx .== missing_tip] .= missing
    end
    species = numeric_species ? species_idx : labels[species_idx]
    (; phy, labels, y, x = x_tree[rows], z = z_tip[species_idx],
      study = study_tree[rows], species, species_idx, z_tip)
end

function _lss_named_tip_index(phy, species)
    if all(x -> x isa Integer, species)
        idx = Int.(species)
    else
        by_name = Dict(name => i for (i, name) in enumerate(phy.leaf_names))
        idx = [by_name[String(x)] for x in species]
    end
    @test all((1 .<= idx) .& (idx .<= phy.n_leaves))
    idx
end

_lss_data(fixture) = (; y = fixture.y, x = fixture.x, z = fixture.z,
                      study = fixture.study, species = fixture.species)

function _lss_hand_correlation()
    # Independent of `_phylo_correlation`: oak is separate; beech/cedar and
    # elm/fir/gum are the two nontrivial clades, all at terminal height two.
    [1.0 0.0 0.0 0.0 0.0 0.0;
     0.0 1.0 0.5 0.0 0.0 0.0;
     0.0 0.5 1.0 0.0 0.0 0.0;
     0.0 0.0 0.0 1.0 0.5 0.5;
     0.0 0.0 0.0 0.5 1.0 0.75;
     0.0 0.0 0.0 0.5 0.75 1.0]
end

function _lss_named_components(fit, data, phy; multi = false)
    observed = .!ismissing.(data.y)
    y = Float64.(data.y[observed])
    x = data.x[observed]
    species = data.species[observed]
    idx = _lss_named_tip_index(phy, species)
    K = _lss_hand_correlation()
    βσ = coef(fit, :sigma)
    αphy = coef(fit, :sd_phylo)
    z_tip = fill(NaN, phy.n_leaves)
    all_idx = _lss_named_tip_index(phy, data.species)
    for i in eachindex(all_idx)
        k = all_idx[i]
        if isnan(z_tip[k])
            z_tip[k] = data.z[i]
        else
            @test z_tip[k] == data.z[i]
        end
    end
    @test all(isfinite, z_tip)
    σe = exp.(βσ[1] .+ βσ[2] .* x)
    σphy = length(αphy) == 1 ? fill(exp(only(αphy)), phy.n_leaves) :
        exp.(αphy[1] .+ αphy[2] .* z_tip)
    V = Matrix(Diagonal(σe .^ 2)) +
        ((σphy * σphy') .* K)[idx, idx]
    if multi
        σstudy = exp(only(coef(fit, :sd)))
        study = data.study[observed]
        V .+= σstudy^2 .* (study .== study')
    end
    (; y, x, V)
end

function _lss_named_loglik(fit, data, phy; multi = false, reml = false)
    parts = _lss_named_components(fit, data, phy; multi)
    y, x, V = parts.y, parts.x, parts.V
    L = cholesky(Symmetric(V)).L
    X = hcat(ones(length(x)), x)
    βμ = coef(fit, :mu)
    if reml
        VinvX = L' \ (L \ X)
        βμ = cholesky(Symmetric(X' * VinvX)) \ (X' * (L' \ (L \ y)))
    end
    residual = y .- X * βμ
    ll = -.5 * (length(y) * log(2π) + 2sum(log, diag(L)) + dot(L \ residual, L \ residual))
    if reml
        ll += 0.5 * size(X, 2) * log(2π) - 0.5 * logdet(cholesky(Symmetric(X' * VinvX)))
    end
    ll
end

function _lss_named_gls(fit, data, phy)
    parts = _lss_named_components(fit, data, phy)
    L = cholesky(Symmetric(parts.V)).L
    X = hcat(ones(length(parts.x)), parts.x)
    VinvX = L' \ (L \ X)
    cholesky(Symmetric(X' * VinvX)) \ (X' * (L' \ (L \ parts.y)))
end

function _lss_identifiable_order_fixture(; order = [11, 3, 16, 7, 1, 14, 5, 9, 2, 12, 4, 15, 6, 13, 8, 10])
    p, m = 16, 6
    phy = random_balanced_tree(p; branch_length = 0.7)
    labels = copy(phy.leaf_names)
    species_tree = repeat(1:p, inner = m)
    rows = vcat([findall(==(tip), species_tree) for tip in order]...)
    z_tip = collect(range(-1.4, 1.3, length = p))
    x_tree = [0.8 * sin(0.23 * i) + 0.35 * cos(0.11 * i) for i in eachindex(species_tree)]
    K = DRModels._phylo_correlation(phy)
    u = exp.(-0.1 .+ 0.55 .* z_tip) .* (cholesky(Symmetric(K)).L *
        [sin(0.61 * i) + 0.35 * cos(0.29 * i) for i in 1:p])
    y_tree = 0.25 .+ 0.5 .* x_tree .+ u[species_tree] .+
        exp.(-0.85 .+ 0.18 .* x_tree) .* (0.7 .* cos.(0.47 .* eachindex(species_tree)))
    idx = species_tree[rows]
    (; phy, y = y_tree[rows], x = x_tree[rows], z = z_tip[idx], species = labels[idx])
end

_phylo_lss_formula() = bf(
    @formula(y ~ x + phylo(1 | species)),
    @formula(sigma ~ x),
    @formula(sd(species, phylogenetic) ~ z),
)

_multi_phylo_lss_formula() = bf(
    @formula(y ~ x + (1 | study) + phylo(1 | species)),
    @formula(sigma ~ x),
    @formula(sd(study) ~ 1),
    @formula(sd(species, phylogenetic) ~ z),
)

_multi_scalar_phylo_lss_formula() = bf(
    @formula(y ~ x + (1 | study) + phylo(1 | species)),
    @formula(sigma ~ x),
    @formula(sd(study) ~ 1),
)

function _lss_boundary_diagnostic(label, left, right)
    # Non-gating: prints the raw log-scale theta L∞ difference for the
    # record only. At a zero-variance boundary, log(sd_phylo) coefficients
    # are unidentified (see the natural-scale gate below), so this raw
    # difference is expected to be large and must never be asserted on.
    difference = maximum(abs.(left.theta - right.theta))
    println("LSS_BOUNDARY_DIAGNOSTIC label=", label, " max_theta_difference=", difference)
    for block in (:mu, :sigma, :sd, :sd_phylo)
        haskey(Dict(left.blocks), block) || continue
        println("LSS_BOUNDARY_DIAGNOSTIC label=", label, " block=", block,
                " left=", coef(left, block), " right=", coef(right, block))
    end
end

@testset "Gaussian LSS phylogenetic tip identity" begin
    @testset "dedicated dense named covariance is invariant to first-seen order" begin
        shuffled = _lss_identity_fixture()
        ordered = _lss_identity_fixture(order = collect(1:6))
        f = _phylo_lss_formula()
        fit_shuffled = drm(f, Gaussian(); data = _lss_data(shuffled), tree = shuffled.phy, method = :ML)
        fit_ordered = drm(f, Gaussian(); data = _lss_data(ordered), tree = ordered.phy, method = :ML)
        # This fixture sits on a singular sd_phylo ridge. Identity is the
        # likelihood and the per-tip SD below, not the gradient stop.
        @test isfinite(loglik(fit_shuffled)) && isfinite(loglik(fit_ordered))
        @test DRModels._phylo_correlation(shuffled.phy) ≈ _lss_hand_correlation() atol = 1e-12
        @test abs(loglik(fit_shuffled) - _lss_named_loglik(fit_shuffled, shuffled, shuffled.phy)) <= 1e-7
        # Boundary-oracle repair (issue #563, diagnosis s7-g8-diagnosis.md):
        # both fits are gradient-stationary optima of the identical dense
        # objective (loglik agrees to 7e-15) on a Hessian-singular
        # (rcond ~1e-17..1e-20) ridge in the 2-parameter sd_phylo log-scale
        # block, so raw-theta L∞ agreement is an unfulfillable oracle here
        # (measured 4.53). Gate on the natural (estimable) scale instead —
        # objective value and per-tip phylogenetic SD — mirroring the
        # scalar sd_phylo repair in 25503f30 for `scalar_multi_small`.
        _lss_boundary_diagnostic("dedicated_small", fit_shuffled, fit_ordered)
        @test abs(loglik(fit_shuffled) - loglik(fit_ordered)) <= 1e-7
        @test coef(fit_shuffled, :mu) ≈ coef(fit_ordered, :mu) atol = 4e-6
        @test coef(fit_shuffled, :sigma) ≈ coef(fit_ordered, :sigma) atol = 4e-6
        # sd_phylo here has 2 coefficients (intercept + slope on z), so the
        # raw pair is not directly comparable even on a log scale; compare
        # the fitted per-tip natural-scale SD they imply. Probe
        # (scratchpad/probe_g8_natural.jl) measured this L∞ difference at
        # 1.2e-8 for this fixture — keep the file's usual 4e-6 tolerance
        # (>300x the measured agreement) rather than tightening it further.
        αphy_shuffled = coef(fit_shuffled, :sd_phylo)
        αphy_ordered = coef(fit_ordered, :sd_phylo)
        σphy_shuffled = exp.(αphy_shuffled[1] .+ αphy_shuffled[2] .* shuffled.z_tip)
        σphy_ordered = exp.(αphy_ordered[1] .+ αphy_ordered[2] .* ordered.z_tip)
        @test maximum(abs.(σphy_shuffled .- σphy_ordered)) <= 4e-6

        # IID group indices remain deliberately first-seen; this repair is
        # restricted to phylogenetic components.
        iid_idx, iid_G = DRModels._group_index(shuffled.species)
        @test iid_G == 6
        @test iid_idx[1:8] == fill(1, 8)
    end

    @testset "an identifiable named tree has permutation-stable coefficients" begin
        shuffled = _lss_identifiable_order_fixture()
        ordered = _lss_identifiable_order_fixture(order = collect(1:16))
        f = _phylo_lss_formula()
        data_shuffled = (; y = shuffled.y, x = shuffled.x, z = shuffled.z,
                         study = repeat(1:4, 24), species = shuffled.species)
        data_ordered = (; y = ordered.y, x = ordered.x, z = ordered.z,
                        study = repeat(1:4, 24), species = ordered.species)
        fit_shuffled = drm(f, Gaussian(); data = data_shuffled, tree = shuffled.phy, method = :ML)
        fit_ordered = drm(f, Gaussian(); data = data_ordered, tree = ordered.phy, method = :ML)
        @test fit_shuffled.converged
        @test fit_ordered.converged
        @test maximum(abs.(fit_shuffled.theta - fit_ordered.theta)) <= 4e-6
    end

    @testset "dedicated sparse and Newick inputs use the same named tree" begin
        dat = _lss_identity_fixture()
        newick = "(oak:2,(beech:1,cedar:1):1,(elm:1,(fir:0.5,gum:0.5):0.5):1);"
        fit = drm(_phylo_lss_formula(), Gaussian(); data = _lss_data(dat), tree = newick,
                  method = :ML, sparse = true)
        @test isfinite(loglik(fit))
        @test abs(loglik(fit) - _lss_named_loglik(fit, dat, dat.phy)) <= 1e-7
    end

    @testset "multi-component phylogenetic covariance remains named" begin
        shuffled = _lss_identity_fixture()
        ordered = _lss_identity_fixture(order = collect(1:6))
        f = _multi_phylo_lss_formula()
        fit_shuffled = drm(f, Gaussian(); data = _lss_data(shuffled), tree = shuffled.phy, method = :ML)
        fit_ordered = drm(f, Gaussian(); data = _lss_data(ordered), tree = ordered.phy, method = :ML)
        @test fit_shuffled.converged
        @test fit_ordered.converged
        @test haskey(Dict(fit_shuffled.blocks), :sd)
        @test haskey(Dict(fit_shuffled.blocks), :sd_phylo)
        @test abs(loglik(fit_shuffled) - _lss_named_loglik(fit_shuffled, shuffled, shuffled.phy; multi = true)) <= 1e-7
        @test maximum(abs.(fit_shuffled.theta - fit_ordered.theta)) <= 4e-6
    end

    @testset "multi-component scalar phylogenetic SD retains named covariance" begin
        shuffled = _lss_identity_fixture()
        ordered = _lss_identity_fixture(order = collect(1:6))
        f = _multi_scalar_phylo_lss_formula()
        fit_shuffled = drm(f, Gaussian(); data = _lss_data(shuffled), tree = shuffled.phy, method = :ML)
        fit_ordered = drm(f, Gaussian(); data = _lss_data(ordered), tree = ordered.phy, method = :ML)
        @test fit_shuffled.converged
        # The ordered fit is on the same zero-variance sd_phylo boundary
        # (singular Hessian on that scalar coefficient). The gradient need
        # not meet g_tol there. Named covariance and the likelihood
        # comparisons below are the checks for this cell.
        @test length(coef(fit_shuffled, :sd_phylo)) == 1
        @test abs(loglik(fit_shuffled) - _lss_named_loglik(fit_shuffled, shuffled, shuffled.phy; multi = true)) <= 1e-7
        @test abs(loglik(fit_shuffled) - loglik(fit_ordered)) <= 1e-7
        @test coef(fit_shuffled, :mu) ≈ coef(fit_ordered, :mu) atol = 4e-6
        @test coef(fit_shuffled, :sigma) ≈ coef(fit_ordered, :sigma) atol = 4e-6
        @test coef(fit_shuffled, :sd) ≈ coef(fit_ordered, :sd) atol = 4e-6
        # At the zero-variance boundary, log(sd_phylo) is not identified:
        # large differences between very negative coefficients can represent
        # the same negligible SD. Test the estimable natural-scale SD instead.
        @test exp(only(coef(fit_shuffled, :sd_phylo))) ≈
              exp(only(coef(fit_ordered, :sd_phylo))) atol = 4e-6
    end

    @testset "REML and a wholly missing response tip retain full tree identity" begin
        shuffled = _lss_identity_fixture(missing_tip = 3)
        ordered = _lss_identity_fixture(order = collect(1:6), missing_tip = 3)
        fit_shuffled = drm(_phylo_lss_formula(), Gaussian(); data = _lss_data(shuffled), tree = shuffled.phy, method = :REML)
        fit_ordered = drm(_phylo_lss_formula(), Gaussian(); data = _lss_data(ordered), tree = ordered.phy, method = :REML)
        @test fit_shuffled.converged
        @test fit_ordered.converged
        @test nobs(fit_shuffled) == count(.!ismissing.(shuffled.y))
        @test estimation_method(fit_shuffled) === :REML
        @test abs(loglik(fit_shuffled) - _lss_named_loglik(fit_shuffled, shuffled, shuffled.phy; reml = true)) <= 1e-7
        @test coef(fit_shuffled, :mu) ≈ _lss_named_gls(fit_shuffled, shuffled, shuffled.phy) atol = 1e-8
        @test maximum(abs.(fit_shuffled.theta - fit_ordered.theta)) <= 4e-6

        sparse_fit = drm(_phylo_lss_formula(), Gaussian(); data = _lss_data(shuffled),
                         tree = shuffled.phy, method = :REML, sparse = true)
        @test sparse_fit.converged
        @test abs(loglik(sparse_fit) - _lss_named_loglik(sparse_fit, shuffled, shuffled.phy; reml = true)) <= 1e-7
        @test maximum(abs.(sparse_fit.theta - fit_shuffled.theta)) <= 4e-6
    end

    @testset "integer positions, and invalid full-input labels" begin
        numeric = _lss_identity_fixture(numeric_species = true)
        fit = drm(_phylo_lss_formula(), Gaussian(); data = _lss_data(numeric), tree = numeric.phy)
        @test fit.converged
        @test abs(loglik(fit) - _lss_named_loglik(fit, numeric, numeric.phy)) <= 1e-7

        symbolic = Symbol.(numeric.labels[numeric.species_idx])
        _, symbol_idx, symbol_G = DRModels._lss_phylo_group_index(numeric.phy, symbolic, :species)
        @test symbol_G == numeric.phy.n_leaves
        @test symbol_idx == numeric.species_idx

        unknown = _lss_identity_fixture(missing_tip = 3)
        bad_species = copy(unknown.species)
        bad_species[findfirst(==(3), unknown.species_idx)] = "not-a-tip"
        @test_throws ArgumentError drm(_phylo_lss_formula(), Gaussian();
            data = (; unknown.y, unknown.x, unknown.z, unknown.study, species = bad_species), tree = unknown.phy)

        absent = _lss_identity_fixture()
        keep = absent.species_idx .!= 6
        err = try
            drm(_phylo_lss_formula(), Gaussian();
                data = (; y = absent.y[keep], x = absent.x[keep], z = absent.z[keep],
                         study = absent.study[keep], species = absent.species[keep]), tree = absent.phy)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("all tree tips", sprint(showerror, err))

        missing_label = _lss_identity_fixture()
        missing_species = Vector{Union{Missing,String}}(missing_label.species)
        missing_species[1] = missing
        @test_throws ArgumentError drm(_phylo_lss_formula(), Gaussian();
            data = (; missing_label.y, missing_label.x, missing_label.z, missing_label.study,
                     species = missing_species), tree = missing_label.phy)

        varying = _lss_identity_fixture()
        varying_z = copy(varying.z)
        varying_z[2] += 0.1  # same species as row 1; never average this silently.
        err = try
            drm(_phylo_lss_formula(), Gaussian();
                data = (; varying.y, varying.x, z = varying_z, varying.study, varying.species),
                tree = varying.phy)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("varies within", sprint(showerror, err))
    end
end
