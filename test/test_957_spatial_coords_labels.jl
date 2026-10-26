# test_957_spatial_coords_labels.jl — #957 (spatial half): row k of `coords` used to be
# taken as the k-th site in first-seen order of the grouping column, with only
# the row count checked. A coordinate matrix in sorted-id order (the natural
# thing to pass) then gave a different likelihood and a collapsed range with
# `converged = true`. `coords` is now matched to the data by site label, as `A`
# is: `ids = [label of each row]`, or a Dict keyed by site label. An unlabelled
# matrix keeps the first-seen convention and warns once per session.
using DRModels
using Test, Random, LinearAlgebra

@testset "#957: spatial coords matched to sites by label" begin
    rng = MersenneTwister(957)
    G = 12; m = 5
    sites_sorted = ["s$(i)" for i in 1:G]                       # coords rows, sorted by id
    xy = rand(rng, G, 2) .* 10.0
    D = [sqrt(sum(abs2, xy[k, :] .- xy[l, :])) for k in 1:G, l in 1:G]
    u = 0.8 .* (cholesky(Symmetric(exp.(-D ./ 2.0) + 1e-8I)).L * randn(rng, G))
    visit = [5, 2, 12, 1, 7, 3, 9, 4, 11, 6, 10, 8]              # data visit sites in this order
    site_col = repeat(sites_sorted[visit], inner = m); n = length(site_col)
    x = randn(rng, n)
    y = 1 .+ 0.5x .+ u[indexin(site_col, sites_sorted)] .+ 0.5 .* randn(rng, n)
    d = (y = y, x = x, site = site_col)
    f = bf(@formula(y ~ x + spatial(1 | site)), @formula(sigma ~ 1))

    ll_ref = loglik(drm(f, Gaussian(); data = d, coords = xy[visit, :]))   # first-seen order
    # the old, silently wrong route: sorted coords with no labels is NOT the same fit
    ll_wrong = loglik(drm(f, Gaussian(); data = d, coords = xy))
    @test !isapprox(ll_wrong, ll_ref; atol = 1e-3)

    @test loglik(drm(f, Gaussian(); data = d, coords = xy, ids = sites_sorted)) ≈ ll_ref atol = 1e-8
    dict = Dict(sites_sorted[i] => xy[i, :] for i in 1:G)
    @test loglik(drm(f, Gaussian(); data = d, coords = dict)) ≈ ll_ref atol = 1e-8
    # extra labelled sites are dropped; a site with no coordinates is refused
    xyplus = vcat(xy, [100.0 100.0])
    @test loglik(drm(f, Gaussian(); data = d, coords = xyplus, ids = vcat(sites_sorted, "z9"))) ≈
          ll_ref atol = 1e-8
    @test_throws ArgumentError drm(f, Gaussian(); data = d, coords = xy[1:G-1, :],
                                   ids = sites_sorted[1:G-1])
    @test_throws ArgumentError drm(f, Gaussian(); data = d, coords = xy, ids = sites_sorted[1:G-1])
    # `ids` with neither `A` nor `coords` is still refused
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                                   data = d, ids = sites_sorted)

    # an unlabelled matrix keeps the first-seen convention and warns once per session
    DRModels._COORDS_WARNED[] = false
    @test_logs (:warn, r"coords") drm(f, Gaussian(); data = d, coords = xy[visit, :])
    @test_logs drm(f, Gaussian(); data = d, coords = xy[visit, :])      # second call is silent
end

@testset "#957: Poisson spatial coords matched by label" begin
    rng = MersenneTwister(9571)
    G = 25; m = 8
    sites_sorted = ["p$(i)" for i in 1:G]
    xy = rand(rng, G, 2) .* 10.0
    D = [sqrt(sum(abs2, xy[k, :] .- xy[l, :])) for k in 1:G, l in 1:G]
    u = 0.9 .* (cholesky(Symmetric(exp.(-D ./ 3.0) + 1e-8I)).L * randn(rng, G))
    visit = randperm(rng, G)
    site_col = repeat(sites_sorted[visit], inner = m); n = length(site_col)
    x = randn(rng, n)
    λ = exp.(0.3 .+ 0.4x .+ u[indexin(site_col, sites_sorted)])
    yc = [let L = exp(-l); k = 0; p = 1.0; while true; p *= rand(rng); p <= L && break; k += 1; end; k end for l in λ]
    d = (y = yc, x = x, site = site_col)
    f = bf(@formula(y ~ x + spatial(1 | site)))
    ll_ref = loglik(drm(f, Poisson(); data = d, coords = xy[visit, :]))
    @test !isapprox(loglik(drm(f, Poisson(); data = d, coords = xy)), ll_ref; atol = 1e-3)
    @test loglik(drm(f, Poisson(); data = d, coords = xy, ids = sites_sorted)) ≈ ll_ref atol = 1e-6
end
