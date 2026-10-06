#!/usr/bin/env julia

# Independent diagnostic for the retained six-tip LSS coefficient discrepancies.
# It independently rebuilds the named covariance.  Its default mode evaluates
# only the fixture-definition prefix of test/test_lss_tip_identity.jl and
# byte-checks a reconstructed copy against that original fixture.

using DRModels
using ForwardDiff
using LinearAlgebra
using SHA

const _K_HAND = [
    1.0 0.0 0.0 0.0 0.0 0.0;
    0.0 1.0 0.5 0.0 0.0 0.0;
    0.0 0.5 1.0 0.0 0.0 0.0;
    0.0 0.0 0.0 1.0 0.5 0.5;
    0.0 0.0 0.0 0.5 1.0 0.75;
    0.0 0.0 0.0 0.5 0.75 1.0;
]

function _tree()
    labels = ["oak", "beech", "cedar", "elm", "fir", "gum"]
    edges = Tuple{Int,Int,Float64}[
        (10, 1, 2.0), (10, 7, 1.0), (7, 2, 1.0), (7, 3, 1.0),
        (10, 8, 1.0), (8, 4, 1.0), (8, 9, 0.5), (9, 5, 0.5), (9, 6, 0.5),
    ]
    return DRModels.make_phy(edges, length(labels); root_index = 10, leaf_names = labels)
end

function _reconstructed_fixture(; order = [5, 1, 6, 3, 4, 2], covariance = _K_HAND)
    phy = _tree()
    p, m = phy.n_leaves, 8
    labels = copy(phy.leaf_names)
    tree_species = repeat(1:p, inner = m)
    rows = vcat([findall(==(tip), tree_species) for tip in order]...)
    species_idx = tree_species[rows]
    z_tip = [-1.1, -0.4, 0.2, 0.7, 1.3, 1.8]
    x_tree = [0.65 * sin(0.41 * i) + 0.12 * cos(0.17 * i) for i in eachindex(tree_species)]
    study_tree = [mod1(i, 4) for i in eachindex(tree_species)]

    alpha_phy = exp.(-0.25 .+ 0.42 .* z_tip)
    u = alpha_phy .* (cholesky(Symmetric(covariance)).L *
        [0.32, -0.21, 0.17, 0.08, -0.14, 0.25])
    b_study = [0.22, -0.17, 0.09, -0.12]
    y_tree = 0.45 .+ 0.37 .* x_tree .+ u[tree_species] .+ b_study[study_tree] .+
        exp.(-1.05 .+ 0.16 .* x_tree) .* (0.8 .* cos.(0.73 .* eachindex(tree_species)))
    labels = phy.leaf_names
    return (; phy, labels, rows, tree_species, species_idx, z_tip, x_tree, study_tree, y_tree,
            y = y_tree[rows], x = x_tree[rows], z = z_tip[species_idx],
            study = study_tree[rows], species = labels[species_idx])
end

function _original_test_fixture(order)
    test_path = normpath(joinpath(@__DIR__, "..", "test", "test_lss_tip_identity.jl"))
    text = read(test_path, String)
    marker = "\n@testset \"Gaussian LSS phylogenetic tip identity\" begin"
    marker_range = findfirst(marker, text)
    marker_range === nothing && error("cannot find the first testset marker in $test_path")
    # Evaluate only fixture/helper definitions, never a @testset body.
    fixture_module = Module(gensym(:LssOriginalFixture))
    Base.include_string(fixture_module, text[firstindex(text):prevind(text, first(marker_range))], test_path)
    # The fixture method is defined by `include_string` after this runner was
    # compiled, so invoke it across Julia's world-age boundary deliberately.
    return Base.invokelatest(getfield(fixture_module, :_lss_identity_fixture); order = order)
end

function _assert_fixture_bytes!(reconstructed, original, label)
    for field in (:labels, :y, :x, :z, :study, :species, :species_idx, :z_tip)
        isequal(getproperty(reconstructed, field), getproperty(original, field)) ||
            error("$label reconstructed field `$field` is not byte-identical to the original test fixture")
    end
    println("LSS_BOUNDARY_FIXTURE label=$label original_test_bytes=true")
end

function _fixture_pair(mode::Symbol, order)
    if mode === :original_test_bytes
        # The source-generator call mirrors the test exactly.  `_K_HAND` remains
        # confined to the independent covariance/nll oracle below.
        reconstructed = _reconstructed_fixture(order = order,
            covariance = DRModels._phylo_correlation(_tree()))
        original = _original_test_fixture(order)
        _assert_fixture_bytes!(reconstructed, original, string(mode, "_", join(order, "_")))
        return original, reconstructed
    elseif mode === :hand_reconstruction
        reconstructed = _reconstructed_fixture(order = order, covariance = _K_HAND)
        return reconstructed, reconstructed
    end
    error("unknown fixture mode `$mode` (use `original_test_bytes` or `hand_reconstruction`)")
end

_data(d) = (; y = d.y, x = d.x, z = d.z, study = d.study, species = d.species)

const _DEDICATED_FORMULA = bf(
    @formula(y ~ x + phylo(1 | species)),
    @formula(sigma ~ x),
    @formula(sd(species, phylogenetic) ~ z),
)

const _MULTI_SCALAR_FORMULA = bf(
    @formula(y ~ x + (1 | study) + phylo(1 | species)),
    @formula(sigma ~ x),
    @formula(sd(study) ~ 1),
)

function _block_range(fit, key::Symbol)
    blocks = Dict(fit.blocks)
    haskey(blocks, key) || error("diagnostic expected `$key` coefficient block")
    return blocks[key]
end

function _finite_vector(label, x)
    all(isfinite, x) || error("$label contains non-finite values")
    return x
end

function _covariance_tree_order(fit, d; multi::Bool)
    βσ = coef(fit, :sigma)
    _finite_vector("sigma coefficients", βσ)
    σe = exp.(βσ[1] .+ βσ[2] .* d.x_tree)
    _finite_vector("residual SDs", σe)

    αphy = coef(fit, :sd_phylo)
    _finite_vector("phylogenetic SD coefficients", αphy)
    σphy = length(αphy) == 1 ? fill(exp(only(αphy)), length(d.z_tip)) :
        exp.(αphy[1] .+ αphy[2] .* d.z_tip)
    _finite_vector("phylogenetic SDs", σphy)
    Pphy_tip = (σphy * σphy') .* _K_HAND
    Pphy = Pphy_tip[d.tree_species, d.tree_species]
    V = Matrix(Diagonal(σe .^ 2)) + Pphy
    Piid = zeros(size(V))
    if multi
        αiid = coef(fit, :sd)
        length(αiid) == 1 || error("scalar multi diagnostic expected one IID log-SD")
        σiid = exp(only(αiid))
        isfinite(σiid) || error("IID SD is non-finite")
        Piid .= σiid^2 .* (d.study_tree .== d.study_tree')
        V .+= Piid
    end
    eig = sort(eigvals(Symmetric(V)))
    minimum(eig) > 0 || error("independently rebuilt named covariance is not positive definite")
    return (; V, Pphy, Piid, eig)
end

function _named_nll(fit, d, named)
    βμ = coef(fit, :mu)
    r = d.y_tree .- (βμ[1] .+ βμ[2] .* d.x_tree)
    fac = cholesky(Symmetric(named.V); check = false)
    issuccess(fac) || error("independent named covariance factorization failed")
    return 0.5 * (logdet(fac) + dot(r, fac \ r)) + 0.5 * length(r) * log(2π)
end

function _fit_report(label, fit, d; multi::Bool)
    fit.converged || error("$label did not converge")
    θ = copy(fit.theta)
    nll = fit.nll
    objective = nll(θ)
    isfinite(objective) || error("$label objective is non-finite")
    abs(-objective - loglik(fit)) ≤ 1e-7 ||
        error("$label nll/loglik mismatch: nll=$objective loglik=$(loglik(fit))")
    score = ForwardDiff.gradient(nll, θ)
    H = ForwardDiff.hessian(nll, θ)
    _finite_vector("$label score", score)
    _finite_vector("$label Hessian", H)
    h_eig = sort(eigvals(Symmetric(H)))
    C = Matrix(vcov(fit))
    _finite_vector("$label reported covariance", C)
    c_eig = sort(eigvals(Symmetric(C)))
    named = _covariance_tree_order(fit, d; multi)
    hand_nll = _named_nll(fit, d, named)
    abs(hand_nll - objective) ≤ 1e-7 ||
        error("$label independent named nll mismatch: hand=$hand_nll closure=$objective")

    println("LSS_BOUNDARY_FIT label=$label objective=$objective loglik=$(loglik(fit)) ",
            "score_inf=$(norm(score, Inf)) hessian_min=$(minimum(h_eig)) ",
            "hessian_max=$(maximum(h_eig)) vcov_min=$(minimum(c_eig)) vcov_max=$(maximum(c_eig)) ",
            "hand_nll=$hand_nll")
    for key in (:mu, :sigma, :sd, :sd_phylo)
        haskey(Dict(fit.blocks), key) || continue
        r = _block_range(fit, key)
        println("LSS_BOUNDARY_BLOCK label=$label block=$key indices=$(first(r)):$(last(r)) ",
                "theta=$(repr(θ[r])) hessian_diag=$(repr(diag(H)[r])) ",
                "vcov_diag=$(repr(diag(C)[r]))")
    end
    println("LSS_BOUNDARY_COV label=$label eig_min=$(minimum(named.eig)) eig_max=$(maximum(named.eig)) ",
            "phy_fro=$(norm(named.Pphy)) iid_fro=$(norm(named.Piid)) total_fro=$(norm(named.V)) ",
            "phy_fraction=$(norm(named.Pphy) / norm(named.V))")
    return (; fit, θ, nll, score, H, C, named)
end

function _compare_pair(label, left, right)
    nll_lr = left.nll(right.θ)
    nll_rl = right.nll(left.θ)
    isfinite(nll_lr) && isfinite(nll_rl) || error("$label cross-order objective is non-finite")
    println("LSS_BOUNDARY_CROSS label=$label own_objective_difference=$(abs(left.nll(left.θ) - right.nll(right.θ))) ",
            "left_at_right=$nll_lr right_at_left=$nll_rl")

    for key in (:mu, :sigma, :sd, :sd_phylo)
        haskey(Dict(left.fit.blocks), key) || continue
        r = _block_range(left.fit, key)
        δ = left.θ[r] - right.θ[r]
        println("LSS_BOUNDARY_DELTA label=$label block=$key maxabs=$(maximum(abs.(δ))) l2=$(norm(δ))")
    end

    rphy = _block_range(left.fit, :sd_phylo)
    δphy = zeros(length(left.θ))
    δphy[rphy] .= right.θ[rphy] - left.θ[rphy]
    δnorm2 = dot(δphy, δphy)
    directional = δnorm2 == 0 ? 0.0 : dot(δphy, left.H * δphy) / δnorm2
    line = Float64[]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        θ = copy(left.θ)
        θ[rphy] .= (1 - t) .* left.θ[rphy] .+ t .* right.θ[rphy]
        value = left.nll(θ)
        isfinite(value) || error("$label sd_phylo line objective is non-finite at t=$t")
        push!(line, value)
    end
    base = first(line)
    println("LSS_BOUNDARY_CURVATURE label=$label sd_phylo_directional_hessian=$directional ",
            "line_nll=$(repr(line)) line_delta=$(repr(line .- base))")

    ΔV = left.named.V - right.named.V
    ΔP = left.named.Pphy - right.named.Pphy
    total_scale = max(norm(left.named.V), norm(right.named.V))
    println("LSS_BOUNDARY_NAMED_COVARIANCE label=$label total_difference_fro=$(norm(ΔV)) ",
            "total_relative_fro=$(norm(ΔV) / total_scale) phy_difference_fro=$(norm(ΔP)) ",
            "phy_relative_fro=$(norm(ΔP) / total_scale) left_eig=$(repr(extrema(left.named.eig))) ",
            "right_eig=$(repr(extrema(right.named.eig)))")
end

function _sha256(path)
    return bytes2hex(sha256(read(path)))
end

function main()
    mode = if isempty(ARGS) || ARGS == ["--fixture=original-test-bytes"]
        :original_test_bytes
    elseif ARGS == ["--fixture=hand-reconstruction"]
        :hand_reconstruction
    else
        error("usage: lss_boundary_diagnostic.jl [--fixture=original-test-bytes|--fixture=hand-reconstruction]")
    end
    root = normpath(joinpath(@__DIR__, ".."))
    source = joinpath(root, "src", "gaussian_lss.jl")
    test_source = joinpath(root, "test", "test_lss_tip_identity.jl")
    println("LSS_BOUNDARY_META julia=$(VERSION) threads=$(Threads.nthreads()) ",
            "openblas_env=$(get(ENV, "OPENBLAS_NUM_THREADS", "unset")) ",
            "openblas_actual=$(BLAS.get_num_threads()) fixture_mode=$mode ",
            "source_sha256=$(_sha256(source)) test_sha256=$(_sha256(test_source)) ",
            "runner_sha256=$(_sha256(@__FILE__))")
    if mode === :original_test_bytes
        println("LSS_BOUNDARY_META oracle=hard_correlation_byte_verified_fixture=true")
    else
        println("LSS_BOUNDARY_META oracle=hard_correlation_hand_reconstruction=true")
    end

    shuffled, shuffled_oracle = _fixture_pair(mode, [5, 1, 6, 3, 4, 2])
    ordered, ordered_oracle = _fixture_pair(mode, collect(1:6))
    dedicated_left = _fit_report("dedicated_shuffled", drm(_DEDICATED_FORMULA, Gaussian();
        data = _data(shuffled), tree = shuffled.phy, method = :ML), shuffled_oracle; multi = false)
    dedicated_right = _fit_report("dedicated_ordered", drm(_DEDICATED_FORMULA, Gaussian();
        data = _data(ordered), tree = ordered.phy, method = :ML), ordered_oracle; multi = false)
    _compare_pair("dedicated_small", dedicated_left, dedicated_right)

    scalar_left = _fit_report("scalar_multi_shuffled", drm(_MULTI_SCALAR_FORMULA, Gaussian();
        data = _data(shuffled), tree = shuffled.phy, method = :ML), shuffled_oracle; multi = true)
    scalar_right = _fit_report("scalar_multi_ordered", drm(_MULTI_SCALAR_FORMULA, Gaussian();
        data = _data(ordered), tree = ordered.phy, method = :ML), ordered_oracle; multi = true)
    _compare_pair("scalar_multi_small", scalar_left, scalar_right)
    println("LSS_BOUNDARY_DIAGNOSTIC_COMPLETE")
end

main()
