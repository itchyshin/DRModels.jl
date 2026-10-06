# test_bridge_lss_routes.jl — bridge-vs-direct parity across every LSS route
# (#563 S6): plain sd() ML/REML, sd_phylo() ML/REML, the forced sparse route,
# and missing-response ("include") semantics. One `@testset` per cell so the
# ledger can count them; mirrors `test_bridge_lss_labels.jl`'s payload shape
# and comparison style.
using DRModels, Test, Random, LinearAlgebra

# ---- shared fixture builders ------------------------------------------------

function _s6_iid_fixture(seed; G = 14, m = 6)
    rng = MersenneTwister(seed)
    n = G * m
    g = repeat(1:G, inner = m)
    gx = collect(range(0.2, 1.4; length = G))
    z = gx[g]
    x = randn(rng, n)
    log_sig_b = 0.1 .+ 0.5 .* gx
    b = exp.(log_sig_b) .* randn(rng, G)
    y = 0.4 .+ 0.3 .* x .+ b[g] .+ exp(-0.4) .* randn(rng, n)
    return (; y, x, z, g)
end

function _s6_phylo_fixture(seed; p = 16, m = 5, branch_length = 0.25)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(p; branch_length = branch_length)
    leaf = phy.leaf_names
    sidx = repeat(1:p, inner = m)
    species = leaf[sidx]
    zg = collect(range(-0.6, 0.6; length = p))
    z = zg[sidx]
    n = p * m
    x = randn(rng, n)
    K = DRModels._phylo_correlation(phy)
    LK = cholesky(Symmetric(K)).L
    sigma_a = exp.(0.2 .+ 0.4 .* zg)
    u = sigma_a .* (LK * randn(rng, p))
    y = 0.5 .+ 0.4 .* x .+ u[sidx] .+ exp(-0.5) .* randn(rng, n)
    return phy, (; y, x, z, species)
end

# Pull the `sd`/`sd_phylo` block coefficients out of a bridge `out` payload, in
# fitted order, using the exact raw block prefix (`"sd_"` for iid, excluding
# the disjoint `"sd_phylo_"` prefix; `"sd_phylo_"` for the phylogenetic block).
function _s6_bridge_sd_block(out, prefix::AbstractString; exclude::AbstractString = "")
    names = out["coef_names"]
    vals = out["coef"]
    return [vals[nm] for nm in names if startswith(nm, prefix) &&
                                        (isempty(exclude) || !startswith(nm, exclude))]
end

@testset "sd ML bridge vs direct (#563 S6)" begin
    data = _s6_iid_fixture(563601)
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z))
    direct = drm(f, Gaussian(); data = data, method = :ML)
    out = drm_bridge(formula = "y ~ x + (1 | g); sigma ~ 1; sd(g) ~ z",
                     family = "gaussian", data = data, options = Dict("method" => "ML"))

    @test out["converged"]
    @test is_converged(direct)
    @test out["coefficients"] ≈ coef(direct) atol = 1e-8
    @test out["loglik"] ≈ loglik(direct) atol = 1e-8
    sdblock = _s6_bridge_sd_block(out, "sd_"; exclude = "sd_phylo_")
    @test sdblock ≈ coef(direct, :sd) atol = 1e-8
end

@testset "sd REML bridge vs direct (#563 S6)" begin
    data = _s6_iid_fixture(563602)
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z))
    direct = drm(f, Gaussian(); data = data, method = :REML)
    out = drm_bridge(formula = "y ~ x + (1 | g); sigma ~ 1; sd(g) ~ z",
                     family = "gaussian", data = data, options = Dict("method" => "REML"))

    @test out["converged"]
    @test is_converged(direct)
    @test estimation_method(direct) === :REML
    # The bridge does not surface an explicit method/estimation_method key
    # (grepped `_bridge_flatten`: no such field). REML-ness is instead pinned
    # via the reported loglik, which for a REML fit equals `reml_loglik(fit)`
    # on the Julia side (test_lss_reml.jl's own defining check).
    @test out["loglik"] ≈ reml_loglik(direct) atol = 1e-8
    @test out["coefficients"] ≈ coef(direct) atol = 1e-8
    sdblock = _s6_bridge_sd_block(out, "sd_"; exclude = "sd_phylo_")
    @test sdblock ≈ coef(direct, :sd) atol = 1e-8
end

@testset "sd_phylo ML bridge vs direct (#563 S6)" begin
    phy, data = _s6_phylo_fixture(563603)
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
           @formula(sd(species, phylogenetic) ~ z))
    direct = drm(f, Gaussian(); data = data, tree = phy, method = :ML)
    out = drm_bridge(formula = "y ~ x + phylo(1 | species); sigma ~ 1; sd_phylo(species) ~ z",
                     family = "gaussian", data = data, tree = phy,
                     options = Dict("method" => "ML"))

    @test out["converged"]
    @test is_converged(direct)
    @test out["coefficients"] ≈ coef(direct) atol = 1e-8
    @test out["loglik"] ≈ loglik(direct) atol = 1e-8
    sdblock = _s6_bridge_sd_block(out, "sd_phylo_")
    @test sdblock ≈ coef(direct, :sd_phylo) atol = 1e-8
end

@testset "sd_phylo REML bridge vs direct (#563 S6)" begin
    phy, data = _s6_phylo_fixture(563604)
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
           @formula(sd(species, phylogenetic) ~ z))
    direct = drm(f, Gaussian(); data = data, tree = phy, method = :REML)
    out = drm_bridge(formula = "y ~ x + phylo(1 | species); sigma ~ 1; sd_phylo(species) ~ z",
                     family = "gaussian", data = data, tree = phy,
                     options = Dict("method" => "REML"))

    @test out["converged"]
    @test is_converged(direct)
    @test estimation_method(direct) === :REML
    @test out["loglik"] ≈ reml_loglik(direct) atol = 1e-8
    @test out["coefficients"] ≈ coef(direct) atol = 1e-8
    sdblock = _s6_bridge_sd_block(out, "sd_phylo_")
    @test sdblock ≈ coef(direct, :sd_phylo) atol = 1e-8
end

@testset "sparse bridge vs direct forced route (#563 S6)" begin
    # `_bridge_fit` has no dedicated `sparse` option key (checked against
    # `_bridge_options`/`_bridge_fit`); it forwards `options["algorithm"]`
    # straight to `drm(...; algorithm = ...)`, and `algorithm in (:sparse,
    # :sparse_lbfgs)` forces the sparse route regardless of tree size (the
    # G > 500 auto-dispatch in gaussian_lss.jl is only the *default*). Force
    # the SAME route on both sides on a small tree so the fixture stays fast.
    phy, data = _s6_phylo_fixture(563605; p = 24, m = 4)
    f = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1),
           @formula(sd(species, phylogenetic) ~ z))

    direct_ml = drm(f, Gaussian(); data = data, tree = phy, algorithm = :sparse, method = :ML, g_tol = 1e-8)
    out_ml = drm_bridge(formula = "y ~ x + phylo(1 | species); sigma ~ 1; sd_phylo(species) ~ z",
                        family = "gaussian", data = data, tree = phy,
                        options = Dict("algorithm" => "sparse", "method" => "ML", "g_tol" => 1e-8))

    @test out_ml["converged"]
    @test is_converged(direct_ml)
    @test out_ml["coefficients"] ≈ coef(direct_ml) atol = 1e-8
    @test out_ml["loglik"] ≈ loglik(direct_ml) atol = 1e-8
    sdblock_ml = _s6_bridge_sd_block(out_ml, "sd_phylo_")
    @test sdblock_ml ≈ coef(direct_ml, :sd_phylo) atol = 1e-8

    direct_reml = drm(f, Gaussian(); data = data, tree = phy, algorithm = :sparse, method = :REML, g_tol = 1e-8)
    out_reml = drm_bridge(formula = "y ~ x + phylo(1 | species); sigma ~ 1; sd_phylo(species) ~ z",
                          family = "gaussian", data = data, tree = phy,
                          options = Dict("algorithm" => "sparse", "method" => "REML", "g_tol" => 1e-8))

    @test out_reml["converged"]
    @test is_converged(direct_reml)
    @test estimation_method(direct_reml) === :REML
    @test out_reml["loglik"] ≈ reml_loglik(direct_reml) atol = 1e-8
    @test out_reml["coefficients"] ≈ coef(direct_reml) atol = 1e-8
end

@testset "missing response bridge vs direct (#563 S6)" begin
    # Observed-rows ("include") pattern: the bridge takes exactly the same
    # `data` (with `missing` responses) into the SAME `drm(...)` call it
    # always makes — there is no separate `response =` option on the bridge
    # (grepped `_bridge_options`/`_bridge_fit`: no such key) to request a
    # different mask semantics, so parity here pins that the bridge's single
    # default behaviour matches the direct fit's observed-rows route exactly.
    data0 = _s6_iid_fixture(563606)
    y = Vector{Union{Missing,Float64}}(copy(data0.y))
    y[3] = missing
    y[10] = missing
    y[27] = missing
    y[52] = missing
    data = merge(data0, (; y))
    f = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(sd(g) ~ z))

    direct_ml = drm(f, Gaussian(); data = data, method = :ML)
    out_ml = drm_bridge(formula = "y ~ x + (1 | g); sigma ~ 1; sd(g) ~ z",
                        family = "gaussian", data = data, options = Dict("method" => "ML"))
    @test out_ml["converged"]
    @test is_converged(direct_ml)
    @test out_ml["nobs"] == nobs(direct_ml)
    @test out_ml["coefficients"] ≈ coef(direct_ml) atol = 1e-8
    @test out_ml["loglik"] ≈ loglik(direct_ml) atol = 1e-8

    direct_reml = drm(f, Gaussian(); data = data, method = :REML)
    out_reml = drm_bridge(formula = "y ~ x + (1 | g); sigma ~ 1; sd(g) ~ z",
                          family = "gaussian", data = data, options = Dict("method" => "REML"))
    @test out_reml["converged"]
    @test is_converged(direct_reml)
    @test estimation_method(direct_reml) === :REML
    @test out_reml["nobs"] == nobs(direct_reml)
    @test out_reml["loglik"] ≈ reml_loglik(direct_reml) atol = 1e-8
    @test out_reml["coefficients"] ≈ coef(direct_reml) atol = 1e-8
end

@testset "#546 unknown univariate formula part key errors loudly (#563 S6)" begin
    # A plain semicolon STRING `"y ~ x; foo ~ 1"` does NOT reach this guard:
    # every part of a bare string is positional (key === nothing in
    # `_bridge_parse_formula_part`), so it is routed into `bf(positional...)`
    # instead, which throws `bf`'s OWN (also non-silent) guard —
    # `ArgumentError("bf: unknown distributional parameter ...")` — confirmed
    # interactively. The bridge's `"unknown univariate formula part"` message
    # lives in the KEYED branch of `_bridge_formula` (src/bridge.jl ~l.630),
    # reached only when `formula` is a Dict/NamedTuple. Both are exercised
    # here so neither hazard silently drops a formula part.
    n = 40
    data = (; y = randn(n), x = randn(n))

    err_string = nothing
    try
        drm_bridge(formula = "y ~ x; foo ~ 1", family = "gaussian", data = data)
    catch e
        err_string = e
    end
    @test err_string isa ArgumentError
    @test occursin("bf: unknown distributional parameter", sprint(showerror, err_string))

    err_keyed = nothing
    try
        drm_bridge(formula = Dict(:mu => "y ~ x", :foo => "foo ~ 1"),
                   family = "gaussian", data = data)
    catch e
        err_keyed = e
    end
    @test err_keyed isa ArgumentError
    @test occursin("unknown univariate formula part", sprint(showerror, err_keyed))
end
