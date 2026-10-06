# test_parity_gaussian_phylo_mean.jl — standalone same-target fixture for
# gaussian_phylo_mean (ML, univariate, sigma ~ 1).
#
# Wired into test/runtests.jl by Option A (#445) after #423+#428 landed.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_parity_gaussian_phylo_mean.jl")'
#
# Claim fence: this PR adds a same-target fixture for gaussian_phylo_mean
# within the row's declared tolerance. claim_status stays partial. Not
# parity complete. Not last fixture-gap. Not sigma-phylo. Not interval
# coverage. Not R-via-Julia bridge admission.
# Workflow G fixtures stay 0.6.0 / ML / no tree; this cell is 0.7.0 ML + tree
# outside test/parity/fixtures/.
#
# #483: the PREVIOUS cell (seed 111) simulated its "phylo" effect as IID
# noise indexed by species (rnorm() per species, no tree structure at all)
# with 1 obs/tip, so the phylogenetic SD was boundary-adjacent and
# UNIDENTIFIABLE — the profiled negative log-likelihood moved by <1e-4 as
# log_sd_phylo ranged over [-30,-10]. A comparison there would have passed
# silently on a parameter neither engine can estimate (the difference was
# 5.045e-06, inside the old atol_re_sd = 1e-4).
#
# This fixture instead uses seed 404 (n_tip=18, n_each=4), whose DGP is a
# GENUINELY phylogenetically-correlated random effect (chol(vcv(tree, corr =
# TRUE)) %*% rnorm(), sd_phylo_true = 1.5, sd_resid = 0.3). The seed was
# chosen BY MEASURING: test/parity/gen_gaussian_phylo_mean.R runs a profiled
# nll sweep before accepting a seed (recorded in nll_profile.csv), and this
# cell's estimate moves the profile by ~5.6 nll units within 1 log-unit and
# is not boundary-adjacent (log_sd_phylo_se = 0.312, see
# expected.meta.toml [identifiability]) — real curvature, not a flat
# plateau.
#
# THE SCALE TRAP (kept from the retired cell, unchanged): DRModels.jl's
# `re_sd(fit)[:species]` is on the RAW branch-length scale (tip variance =
# tree height h); drmTMB's is on the CORRELATION scale (`ape::vcv(tree, corr
# = TRUE)`, tip variance 1 regardless of h). The two agree only when h == 1.
# This fixture's primary tree has h ≈ 2.474, so the raw comparison would
# silently be wrong by sqrt(h) ≈ 1.573 without the conversion applied below.
#
# MULTI-HEIGHT ROUND TRIP (#483: kept, now on an identifiable cell — this is
# the only thing that catches the sqrt(height) scale error, and it is
# invisible on a height-1 tree). tree_h0.5.newick / tree_h1.0.newick /
# tree_h3.0.newick are the SAME data refit against the SAME topology
# rescaled to three absolute heights; heights.toml records drmTMB's native
# refit at each. drmTMB's sd_phylo_corr is height-invariant there (the
# correlation-scale objective is bit-identical regardless of raw height); the
# test below checks that DRModels.jl's raw re_sd, after x sqrt(measured height),
# lands on that same invariant value at all three heights.

# #476: wrapped in a module so this file's FIXTURE/_load_data/_coef_named/
# _within cannot collide with the same names in another parity test file
# included later by runtests.jl (they used to live bare in Main).
module TestParityGaussianPhyloMean

using DRModels
using Test
using TOML
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "phylo-mean", "gaussian-phylo-mean")

function _load_data(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y, :x))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(col)
        end
    end
    return NamedTuple(pairs)
end

function _coef_named(fit)
    θ = coef(fit)
    namemap = Dict(p => ns for (p, ns) in fit.coefnames)
    out = Dict{String,Float64}()
    for (param, r) in fit.blocks
        param === :phylocov && continue
        param === :resd && continue
        haskey(namemap, param) || continue
        names = namemap[param]
        slice = θ[r]
        length(names) == length(slice) || error("name/coef mismatch for $param")
        for (nm, est) in zip(names, slice)
            out["$(param)_$(nm)"] = est
        end
    end
    return out
end

_within(a, b, rtol, atol) = abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

## drmTMB's sd_phylo (correlation scale) from a fitted DRModels.jl model + its own
## tree-height reading: species_corr_scale == re_sd(fit)[:species] * sqrt(h).
function _sd_phylo_corr_scale(fit, tree_newick::AbstractString)
    phy = augmented_phy(tree_newick)
    h = phylo_tree_height(phy)
    sds = re_sd(fit)
    @test haskey(sds, :species)
    sd_raw = Float64(sds[:species])
    return sd_raw * sqrt(h), h
end

@testset "gaussian_phylo_mean same-target fixture" begin
    @test isfile(joinpath(FIXTURE, "data.csv"))
    @test isfile(joinpath(FIXTURE, "tree.newick"))
    @test isfile(joinpath(FIXTURE, "expected.toml"))
    @test isfile(joinpath(FIXTURE, "expected.meta.toml"))

    expected = TOML.parsefile(joinpath(FIXTURE, "expected.toml"))
    meta = TOML.parsefile(joinpath(FIXTURE, "expected.meta.toml"))
    @test meta["drmtmb_version"] == "0.7.0"
    @test haskey(meta, "r_call")
    @test haskey(meta, "seed")
    @test meta["seed"] == 404
    @test meta["n_tip"] == 18
    @test expected["fit"]["method"] == "ML"
    @test expected["fit"]["engine"] == "tmb"
    @test expected["fit"]["family"] == "gaussian"
    @test haskey(expected, "status")
    @test expected["status"]["converged"] == true
    @test haskey(expected["status"], "pdHess")
    @test haskey(expected["status"], "interval_status")
    @test expected["status"]["interval_status"] != "coverage_claimed"

    # #483: this cell's phylo SD must NOT be boundary-adjacent — that is the
    # whole point of the reseed. If a future regeneration ever produces a
    # boundary-adjacent estimate again, this fixture has silently regressed
    # to the failure mode #483 fixed.
    @test haskey(meta, "identifiability")
    @test meta["identifiability"]["boundary_adjacent"] == false

    dat = _load_data(FIXTURE)
    tree = read(joinpath(FIXTURE, "tree.newick"), String)
    form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = tree)

    @test estimation_method(fit) === :ML
    @test is_converged(fit)
    @test isfinite(loglik(fit))

    atol_ll = Float64(get(expected["tol"], "atol_loglik", 1e-6))
    atol_c = Float64(get(expected["tol"], "atol_coef", 1e-5))
    rtol_c = Float64(get(expected["tol"], "rtol_coef", 1e-5))
    ref_ll = Float64(expected["fit"]["loglik"])
    @test _within(loglik(fit), ref_ll, 0.0, atol_ll)

    got = _coef_named(fit)
    for (name, ref) in expected["coef"]
        @test haskey(got, name)
        @test _within(got[name], Float64(ref), rtol_c, atol_c)
    end

    # The row's DEFINING quantity: the fitted phylogenetic SD (see the SCALE
    # TRAP note above the testset). Compare on drmTMB's CORRELATION scale by
    # multiplying DRModels.jl's RAW-scale re_sd by sqrt(tree_height).
    @test haskey(expected, "re_sd")
    @test haskey(expected["re_sd"], "species_corr_scale")
    @test haskey(meta, "tree_height")

    ref_h = Float64(meta["tree_height"])
    sd_phylo_corr, h = _sd_phylo_corr_scale(fit, tree)
    @test _within(h, ref_h, 1e-6, 1e-6)   # Julia's and R's tree-height readings agree

    ref_sd_phylo = Float64(expected["re_sd"]["species_corr_scale"])
    atol_re_sd = Float64(get(expected["tol"], "atol_re_sd", 1e-4))
    @test _within(sd_phylo_corr, ref_sd_phylo, 0.0, atol_re_sd)

    @testset "multi-height round trip (#483, kept — invisible on a height-1 tree)" begin
        @test isfile(joinpath(FIXTURE, "heights.toml"))
        heights = TOML.parsefile(joinpath(FIXTURE, "heights.toml"))
        @test haskey(heights, "height")
        rows = heights["height"]
        @test length(rows) == 3

        # Match each heights.toml row to its tree_h<height>.newick file by
        # closest numeric height (R's format() and Julia's string() do not
        # promise identical text for the same float) rather than a literal
        # filename guess.
        available = filter(f -> startswith(f, "tree_h") && endswith(f, ".newick"),
                            readdir(FIXTURE))

        for row in rows
            th = Float64(row["target_height"])
            dists = [abs(parse(Float64, replace(replace(f, "tree_h" => ""), ".newick" => "")) - th)
                     for f in available]
            treefile = joinpath(FIXTURE, available[argmin(dists)])
            @test isfile(treefile)
            tree_h = read(treefile, String)
            fit_h = drm(form, Gaussian(); data = dat, tree = tree_h)
            @test is_converged(fit_h)

            sd_phylo_corr_h, h_measured = _sd_phylo_corr_scale(fit_h, tree_h)
            @test _within(h_measured, Float64(row["measured_height"]), 1e-6, 1e-6)

            ref_h_val = Float64(row["sd_phylo_corr"])
            @test _within(sd_phylo_corr_h, ref_h_val, 0.0, atol_re_sd)
        end
    end
end

end # module TestParityGaussianPhyloMean
