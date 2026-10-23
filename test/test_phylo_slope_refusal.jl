# test_phylo_slope_refusal.jl — silent-data-loss fix (tonight's issue).
#
# `_split_ranef` (src/gaussian_ranef.jl:19) used to keep only the grouping
# symbol of a `phylo(...)` marker, so `phylo(1 + x | group)` (and
# `phylo(0 + x | group)`, `phylo(x | group)`) silently fit the SAME
# intercept-only phylogenetic random effect as `phylo(1 | group)` on every
# univariate family, Gaussian included — no error, byte-for-byte identical
# theta/loglik/coefnames. drmTMB fits a genuine two-SD phylogenetic random
# slope on Gaussian for this construct; DRModels.jl has no such route yet.
#
# Fix: fail closed. `phylo(<not literal 1> | group)` on a univariate route
# now throws `ArgumentError` naming the marker and saying it is not
# implemented, instead of quietly dropping the slope.
#
# Untouched by design (regression-asserted below, not exercised by the fix):
#   - bivariate q4/q2 `phylo(1 | species)` / `phylo(1 | tag | species)`
#     (gaussian_bivariate.jl `_split_bivariate_q4_rhs` / `_q4_marker_group`
#     already enforce intercept-only independently of `_split_ranef`).
#   - the `(1 | tag | phylo(group))` mean↔scale coupling
#     (locscale_frontend.jl `_ls_parse_coupled`; `phylo` there wraps only the
#     grouping symbol, never reaches `_split_ranef`'s phylo branch at all).
#
# #620: the same choke point (`_split_ranef`'s relmat/animal/spatial branches)
# had the identical bug for `relmat(...)`/`animal(...)`/`spatial(...)` markers.
# They now refuse too, with a shorter message (no drmTMB-Gaussian claim, since
# that has not been verified for these markers): "is not implemented on the
# univariate routes; only the intercept form is".
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "structured markers refuse non-intercept lhs (#620)" begin
    Random.seed!(20260902)
    p = 8
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = 3)
    n = length(species)
    x = randn(n)

    @testset "Poisson" begin
        y = Float64.([rand(Distributions.Poisson(exp(0.2 + 0.25xi))) for xi in x])
        data = (; y, x, species)

        err1 = try
            drm(bf(@formula(y ~ x + phylo(1 + x | species))), Poisson();
                data = data, tree = phy, se = false)
            nothing
        catch e
            e
        end
        @test err1 isa ArgumentError
        @test occursin("phylo(1 + x | species)", err1.msg)
        @test occursin("not implemented", err1.msg)

        err2 = try
            drm(bf(@formula(y ~ x + phylo(0 + x | species))), Poisson();
                data = data, tree = phy, se = false)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("phylo(0 + x | species)", err2.msg)
        @test occursin("not implemented", err2.msg)

        # Positive control: the still-supported intercept-only marker still fits.
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson();
                   data = data, tree = phy, se = false)
        @test all(isfinite, fit.theta)
    end

    @testset "Gaussian" begin
        y = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
        data = (; y, x, species)

        # #620 follow-up landed: the Gaussian MEAN now fits `phylo(1 + x | g)` as
        # drmTMB's two-SD (intercept + slope, independent fields) model — see
        # test_phylo_slope_two_sd.jl. It must produce a TWO-row :resd block, not
        # the intercept-only fit this file was written to catch.
        fit_slope = drm(bf(@formula(y ~ x + phylo(1 + x | species))), Gaussian();
                        data = data, tree = phy)
        @test all(isfinite, fit_slope.theta)
        @test first(cn[2] for cn in fit_slope.coefnames if cn[1] === :resd) == ["species", "species:x"]

        err2 = try
            drm(bf(@formula(y ~ x + phylo(0 + x | species))), Gaussian();
                data = data, tree = phy)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("phylo(0 + x | species)", err2.msg)
        @test occursin("not implemented", err2.msg)

        # Positive control: the still-supported intercept-only marker still fits.
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Gaussian();
                   data = data, tree = phy)
        @test all(isfinite, fit.theta)
    end

    @testset "relmat" begin
        Random.seed!(20260905)
        G = 8
        Mm = randn(G, G); Kraw = Mm * Mm' / G + I
        d = sqrt.(diag(Kraw)); K = Kraw ./ (d * d')
        m = 3; n = G * m
        id = repeat(1:G, inner = m)
        x = randn(n)
        y = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
        data = (; y, x, id)

        err1 = try
            drm(bf(@formula(y ~ x + relmat(1 + x | id))), Gaussian(); data = data, K = K)
            nothing
        catch e
            e
        end
        @test err1 isa ArgumentError
        @test occursin("relmat(1 + x | id)", err1.msg)
        @test occursin("not implemented", err1.msg)

        err2 = try
            drm(bf(@formula(y ~ x + relmat(0 + x | id))), Gaussian(); data = data, K = K)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("relmat(0 + x | id)", err2.msg)
        @test occursin("not implemented", err2.msg)

        # Positive control (fixture shape copied from test_gaussian_structured.jl):
        # the still-supported intercept-only marker still fits.
        fit = drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1)), Gaussian();
                   data = data, K = K)
        @test isfinite(loglik(fit))
    end

    @testset "animal" begin
        Random.seed!(20260906)
        G = 8
        Mm = randn(G, G); A0 = Mm * Mm' / G + I
        d = sqrt.(diag(A0)); A = A0 ./ (d * d')
        m = 3; n = G * m
        id = repeat(1:G, inner = m)
        x = randn(n)
        y = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
        data = (; y, x, id)

        err1 = try
            drm(bf(@formula(y ~ x + animal(1 + x | id))), Gaussian(); data = data, A = A)
            nothing
        catch e
            e
        end
        @test err1 isa ArgumentError
        @test occursin("animal(1 + x | id)", err1.msg)
        @test occursin("not implemented", err1.msg)

        err2 = try
            drm(bf(@formula(y ~ x + animal(0 + x | id))), Gaussian(); data = data, A = A)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("animal(0 + x | id)", err2.msg)
        @test occursin("not implemented", err2.msg)

        # Positive control (fixture shape copied from test_gaussian_structured.jl):
        # the still-supported intercept-only marker still fits.
        fit = drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)), Gaussian();
                   data = data, A = A)
        @test isfinite(loglik(fit))
    end

    @testset "spatial" begin
        Random.seed!(20260907)
        G = 8
        coords = rand(G, 2) .* 10.0
        m = 3; n = G * m
        site = repeat(1:G, inner = m)
        x = randn(n)
        y = 0.2 .+ 0.4 .* x .+ 0.3 .* randn(n)
        data = (; y, x, site)

        err1 = try
            drm(bf(@formula(y ~ x + spatial(1 + x | site))), Gaussian(); data = data, coords = coords)
            nothing
        catch e
            e
        end
        @test err1 isa ArgumentError
        @test occursin("spatial(1 + x | site)", err1.msg)
        @test occursin("not implemented", err1.msg)

        # Bare `x` (no explicit `0 +`) — the coordinator's own example shape.
        err2 = try
            drm(bf(@formula(y ~ x + spatial(x | site))), Gaussian(); data = data, coords = coords)
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError
        @test occursin("spatial(x | site)", err2.msg)
        @test occursin("not implemented", err2.msg)

        # Positive control (fixture shape copied from test_gaussian_spatial.jl):
        # the still-supported intercept-only marker still fits.
        fit = drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1)), Gaussian();
                   data = data, coords = coords)
        @test isfinite(loglik(fit))
    end
end

@testset "regression: bivariate q4 phylo (different syntax) unaffected" begin
    # Small self-contained copy of the `_q4_frontend_data`/`_q4_formula` DGP and
    # call in test_gaussian_bivariate_phylo.jl: `phylo(1 | species)` on mu1, mu2,
    # sigma1, sigma2 — the labelled 3-part-bar front end never routes through
    # `_split_ranef` for these axes, so it must still construct and fit after
    # the univariate-route fix above.
    Random.seed!(20260904)
    p = 8
    nrep = 3
    phy = random_balanced_tree(p; branch_length = 0.2)
    species_idx = repeat(1:p, inner = nrep)
    species = [phy.leaf_names[k] for k in species_idx]
    n = length(species_idx)
    x = randn(n)
    y1 = 1.0 .+ 0.5 .* x .+ 0.2 .* randn(n)
    y2 = -0.3 .+ 0.4 .* x .+ 0.2 .* randn(n)
    data = (; y1, y2, x, species)

    form = bf(
        mu1 = @formula(y1 ~ x + phylo(1 | species)),
        mu2 = @formula(y2 ~ x + phylo(1 | species)),
        sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
        sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
        rho12 = @formula(rho12 ~ 1),
    )
    fit = drm(form, Gaussian();
              data = data, tree = phy,
              q4_iterations = 30, q4_n_newton = 10, q4_vcov = false)
    @test isfinite(loglik(fit))
end

@testset "regression: (1 | tag | phylo(group)) location-scale coupling unaffected" begin
    # Different syntax again: `phylo` here wraps only the grouping symbol inside
    # a coupled `(1 | tag | phylo(group))` term, parsed by locscale_frontend.jl,
    # never by `_split_ranef`'s phylo branch.
    Random.seed!(20260903)
    p = 8
    m = 3
    phy = random_balanced_tree(p; branch_length = 0.25)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    y = Float64.([rand(Distributions.Gamma(1.0, exp(0.2 + 0.3xi))) for xi in x])
    data = (; y, x, species)

    fit = drm(bf(@formula(y ~ x + (1 | p | phylo(species))),
                 @formula(sigma ~ 1 + (1 | p | phylo(species)))),
              DRModels.Gamma(); data = data, tree = phy, se = false)
    @test isfinite(loglik(fit))
end

@testset "regression: bivariate lognormal spatial(1|group) delegation unaffected" begin
    # bivariate_lognormal.jl delegates the WHOLE fit to `drm(f, Gaussian(); ...)`
    # (the BivariateDrmFormula route in gaussian_bivariate.jl), which parses
    # structured markers via `_split_bivariate_q4_rhs`/`_q4_marker_group` — never
    # `_split_ranef` — so the #620 checks above must not touch it.
    Random.seed!(20260908)
    G = 8
    coords = rand(G, 2) .* 10.0
    grp = repeat(1:G, inner = 3)
    n = length(grp)
    x = randn(n)
    y1 = exp.(0.2 .+ 0.3 .* x .+ 0.2 .* randn(n))
    y2 = exp.(-0.1 .+ 0.2 .* x .+ 0.2 .* randn(n))
    dat = (; y1, y2, x, g = grp)
    form = bf(mu1 = @formula(y1 ~ x + spatial(1 | g)),
              mu2 = @formula(y2 ~ x + spatial(1 | g)),
              sigma1 = @formula(sigma1 ~ 1 + spatial(1 | g)),
              sigma2 = @formula(sigma2 ~ 1 + spatial(1 | g)),
              rho12 = @formula(rho12 ~ 1))
    fit = drm(form, LogNormal(); data = dat, coords = coords, q4_vcov = false)
    @test isfinite(loglik(fit))
end
