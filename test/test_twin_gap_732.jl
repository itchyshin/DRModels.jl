# test_twin_gap_732.jl — twin drmTMB#1272: `re_sd(fit)` on a `phylo(1 | g)`
# grouping is reported on the RAW branch-length scale (tip variance = the
# tree's height `h`) by every route whose `fit.phylo_scale === :covariance`
# (every non-Gaussian Laplace/GLMM phylo route, and the sparse Gaussian-mean
# route on a NON-ultrametric tree; the sparse Gaussian-mean route on an
# ultrametric tree is on the correlation scale since #979). drmTMB instead
# reports that SD on the tip-CORRELATION scale
# (`ape::vcv(tree, corr = TRUE)`, tip variance 1). FE and logLik already agree
# between the two engines (#732 issue title) — this is a reporting-convention
# gap, not a numerical defect, and the exact conversion is
#
#     sd_drmTMB == re_sd(fit)[:g] * sqrt(phylo_tree_height(augmented_phy(tree)))
#
# already documented in `src/sparse_phy.jl`'s `phylo_tree_height` docstring and
# already exercised (without a public accessor option) by
# `test_parity_gaussian_phylo_mean.jl` (Gaussian, 3 tree heights) and
# `test_cumlogit_phylo.jl` (CumulativeLogit, non-Gaussian Laplace). This file
# adds the public `re_sd(fit; scale = :drmtmb, tree = ...)` option that
# performs that conversion, and pins it against the SAME two R-oracle fixtures
# instead of re-deriving new ones.
#
# NOT claiming: a route where a `:covariance`-scale fit mixes a phylo grouping
# with a non-phylo grouping in the same `re_sd` dict (every such combination
# found in this repo — Gaussian's `structured_with_ordinary_bar`, the two- and
# multi-structured Gaussian routes, the dense phylo fallback — sets
# `phylo_scale = :correlation` instead, so `scale = :drmtmb` is a no-op there;
# every route defaulting to `:covariance` is phylo-only). If a future route
# violates that, `scale = :drmtmb` would over-multiply an unrelated group —
# see the caveat in the `re_sd` docstring.
module TestTwinGap732

using DRModels
using Test
using TOML
using DelimitedFiles: readdlm
using Random, LinearAlgebra, Logging

_within(a, b, rtol, atol) = abs(a - b) <= max(atol, rtol * max(abs(a), abs(b)))

@testset "twin #732/drmTMB#1272 — re_sd scale = :drmtmb" begin

    @testset "Gaussian phylo(1 | species) mean (same fixture as test_parity_gaussian_phylo_mean.jl)" begin
        fixture = joinpath(@__DIR__, "parity", "phylo-mean", "gaussian-phylo-mean")
        raw, header = readdlm(joinpath(fixture, "data.csv"), ','; header = true)
        cols = Symbol.(strip.(string.(vec(header))))
        numeric = Set((:y, :x))
        dat = NamedTuple(map(enumerate(cols)) do (j, name)
            col = raw[:, j]
            name in numeric ? name => Float64[parse(Float64, string(v)) for v in col] :
                               name => string.(col)
        end)
        tree = read(joinpath(fixture, "tree.newick"), String)
        expected = TOML.parsefile(joinpath(fixture, "expected.toml"))
        atol_re_sd = Float64(get(expected["tol"], "atol_re_sd", 1e-4))
        ref_sd_phylo = Float64(expected["re_sd"]["species_corr_scale"])

        form = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))
        fit = drm(form, Gaussian(); data = dat, tree = tree)
        @test is_converged(fit)
        # #979: the sparse sigma ~ 1 route fits an ultrametric tree at unit height,
        # so its `re_sd` is already drmTMB's correlation-scale number and
        # `scale = :drmtmb` is a no-op (no `tree` needed), as on the other
        # `:correlation` routes below.
        @test fit.phylo_scale === :correlation

        native = re_sd(fit)
        @test haskey(native, :species)
        phy = augmented_phy(tree)
        @test _within(Float64(native[:species]), ref_sd_phylo, 0.0, atol_re_sd)

        # The accessor option is the identity, with or without a tree.
        @test re_sd(fit; scale = :drmtmb) == native
        @test re_sd(fit; scale = :drmtmb, tree = tree) == native
        @test re_sd(fit; scale = :drmtmb, tree = phy) == native
        @test re_sd(fit; scale = :native) == native
    end

    @testset "CumulativeLogit phylo(1 | species) mean, non-Gaussian Laplace (same fixture/oracle as test_cumlogit_phylo.jl)" begin
        fixture = joinpath(@__DIR__, "parity", "fixtures", "cumlogit-mu-phylo")
        raw, header = readdlm(joinpath(fixture, "data.csv"), ','; header = true)
        cols = Symbol.(strip.(string.(vec(header))))
        j = Dict(c => i for (i, c) in enumerate(cols))
        species = string.(raw[:, j[:species]])
        x = Float64[parse(Float64, string(v)) for v in raw[:, j[:x]]]
        y = Float64[parse(Float64, string(v)) for v in raw[:, j[:y_int]]]
        data = (; species, x, y)
        tree = read(joinpath(fixture, "tree.newick"), String)

        # drmTMB 0.7.0 oracle, pinned in test_cumlogit_phylo.jl's header.
        sdphylo_corr_R = 1.471822

        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), CumulativeLogit();
                  data = data, tree = tree, se = false)
        @test fit.converged
        @test fit.phylo_scale === :covariance

        native = re_sd(fit)
        @test haskey(native, :species)

        got = re_sd(fit; scale = :drmtmb, tree = tree)
        @test _within(Float64(got[:species]), sdphylo_corr_R, 1e-3, 0.0)
        @test re_sd(fit) == native   # default unchanged
    end

    @testset "phylo_scale === :correlation route: scale = :drmtmb is a no-op" begin
        # Gaussian's `structured_with_ordinary_bar` route (phylo(1|sp) + (1|h))
        # fits the phylo marker against the tip CORRELATION matrix directly
        # (`_withphyloscale(..., :correlation)`, src/gaussian_core.jl ~944), so
        # its raw `re_sd` already IS drmTMB's number — no conversion needed,
        # and no `tree` keyword should be required to get it.
        Random.seed!(20260927)
        ntip, nh = 14, 5
        phy = random_balanced_tree(ntip; branch_length = 0.3)
        C = DRModels._phylo_correlation(phy)
        species = repeat(phy.leaf_names, outer = nh)
        h = repeat(["h$k" for k in 1:nh], inner = ntip)
        n = length(species)
        x = randn(n)
        a = 0.8 .* (cholesky(Symmetric(C)).L * randn(ntip))
        b = 0.7 .* randn(nh)
        leaf = Dict(nm => i for (i, nm) in enumerate(phy.leaf_names))
        hix = Dict("h$k" => k for k in 1:nh)
        y = [1.0 + 0.5 * x[i] + a[leaf[species[i]]] + b[hix[h[i]]] + 0.5 * randn() for i in 1:n]
        data = (; y, x, species, h)

        f = bf(@formula(y ~ x + phylo(1 | species) + (1 | h)), @formula(sigma ~ 1))
        fit = drm(f, Gaussian(); data = data, tree = phy)
        @test is_converged(fit)
        @test fit.phylo_scale === :correlation

        native = re_sd(fit)
        @test haskey(native, :species)
        @test haskey(native, :h)

        # No `tree` needed on a :correlation-scale fit — the whole point of
        # gating on `phylo_scale`.
        got = re_sd(fit; scale = :drmtmb)
        @test got == native
        # Passing a tree anyway must not change the (already-correct) answer.
        got_with_tree = re_sd(fit; scale = :drmtmb, tree = phy)
        @test got_with_tree == native
    end

    @testset "error paths" begin
        fixture = joinpath(@__DIR__, "parity", "phylo-mean", "gaussian-phylo-mean")
        raw, header = readdlm(joinpath(fixture, "data.csv"), ','; header = true)
        cols = Symbol.(strip.(string.(vec(header))))
        numeric = Set((:y, :x))
        dat = NamedTuple(map(enumerate(cols)) do (j, name)
            col = raw[:, j]
            name in numeric ? name => Float64[parse(Float64, string(v)) for v in col] :
                               name => string.(col)
        end)
        tree = read(joinpath(fixture, "tree.newick"), String)
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                  Gaussian(); data = dat, tree = tree)

        @test_throws ArgumentError re_sd(fit; scale = :bogus)

        # A `:covariance`-scale fit needs the tree to convert. The Gaussian sparse
        # route is on the correlation scale since #979 for an ultrametric tree, so
        # use a NON-ultrametric one (first tip branch doubled), which stays raw.
        tree_nu = replace(tree, r":([0-9.eE+-]+)" => m -> ":" * repr(2 * parse(Float64, m[2:end])); count = 1)
        @test !DRModels._phylo_tree_is_ultrametric(augmented_phy(tree_nu))
        fit_nu = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)),
                Gaussian(); data = dat, tree = tree_nu)
        end
        @test fit_nu.phylo_scale === :covariance
        @test_throws ArgumentError re_sd(fit_nu; scale = :drmtmb)   # no tree given
    end
end

end # module TestTwinGap732
