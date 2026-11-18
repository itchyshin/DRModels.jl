# Animal `A` label errors (message content) and `_marginal_compare_guard`
# early return when both fits are `:VA`.
using DRModels
using Test, Random, LinearAlgebra
import Distributions

@testset "animal(1 | id): A labels vs id levels — clear errors" begin
    rng = MersenneTwister(4242)
    G = 6; m = 4
    ids_sorted = ["a$(i)" for i in 1:G]
    B = randn(rng, G, G); A = Matrix(Symmetric(B * B' / G + I))
    col = repeat(ids_sorted[[3, 1, 6, 2, 5, 4]], inner = m); n = length(col)
    x = randn(rng, n); y = 1 .+ 0.5x .+ randn(rng, n)
    d = (y = y, x = x, animal = col)
    f = bf(@formula(y ~ x + animal(1 | animal)), @formula(sigma ~ 1))

    # A level of the data with no row in A: names the offender
    err = try drm(f, Gaussian(); data = d, A = A[1:G-1, 1:G-1], ids = ids_sorted[1:G-1]); nothing
          catch e; e end
    @test err isa ArgumentError
    @test occursin("no row in `A`", err.msg) && occursin("\"a6\"", err.msg)

    # Duplicate labels
    err = try drm(f, Gaussian(); data = d, A = A, ids = vcat(ids_sorted[1:G-1], "a1")); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("more than once", err.msg)

    # Label count vs matrix size
    err = try drm(f, Gaussian(); data = d, A = A, ids = ids_sorted[1:G-1]); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("label(s)", err.msg)

    # Sorting by label reproduces the first-seen-order fit
    first_seen = unique(col)
    perm = indexin(first_seen, ids_sorted)
    ll_ref = loglik(drm(f, Gaussian(); data = d, A = A[perm, perm]))
    @test loglik(drm(f, Gaussian(); data = d, A = A, ids = ids_sorted)) ≈ ll_ref atol = 1e-8
end

@testset "_marginal_compare_guard: both :VA early return" begin
    rng = MersenneTwister(136)
    G = 40; per = 10; n = G * per
    g = repeat(1:G, inner = per); x = randn(rng, n)
    λ = exp.(0.4 .- 0.3 .* x .+ 0.5 .* randn(rng, G)[g])
    y = Float64.([rand(rng, Distributions.Poisson(l)) for l in λ])
    data = (; y, x, g)
    va_full = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data, marginal = :VA)
    va_red  = drm(bf(@formula(y ~ 1 + (1 | g))), Poisson(); data = data, marginal = :VA)
    la_full = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = data)
    @test va_full.marginal === :VA && va_red.marginal === :VA

    # Mixed VA / non-VA throws the ELBO message
    err = try DRModels._marginal_compare_guard(va_full, la_full, "lrtest"); nothing
          catch e; e end
    @test err isa ArgumentError && occursin("ELBO", err.msg)

    # Same tag => early return `nothing` (current behaviour)
    @test DRModels._marginal_compare_guard(va_red, va_full, "lrtest") === nothing
    # Two ELBOs are not log-likelihoods either; if this should refuse, flip to @test_throws.
    @test_broken (try DRModels._marginal_compare_guard(va_red, va_full, "lrtest"); false
                  catch e; e isa ArgumentError end)
end
