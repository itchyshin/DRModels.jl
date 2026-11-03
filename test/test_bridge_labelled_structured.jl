# test_bridge_labelled_structured.jl — drmTMB#1512: a labelled K / A in another
# row order must give the same fit as the first-appearance order, and an
# unalignable one must be refused, not fitted against the wrong rows.
using DRModels
using Test
using Random
using LinearAlgebra

@testset "drm_bridge labelled relmat K / animal A (drmTMB#1512)" begin
    rng = MersenneTwister(1512)
    G, nrep = 12, 4
    ids = ["id$(lpad(i, 2, '0'))" for i in 1:G]
    # first-appearance order is deliberately NOT sorted
    appear = ids[randperm(rng, G)]
    B = randn(rng, G, G)
    Kfull = Symmetric(B * B' / G + I)           # SPD, rows/cols ordered as `ids`
    ord = [findfirst(==(l), ids) for l in appear]
    Kappear = Matrix(Kfull)[ord, ord]           # first-appearance order
    u = cholesky(Symmetric(Kappear)).L * randn(rng, G)
    grp = repeat(appear, inner = nrep)
    x = randn(rng, G * nrep)
    y = 0.5 .+ 0.3 .* x .+ repeat(u, inner = nrep) .+ 0.7 .* randn(rng, G * nrep)
    dat = (; y, x, id = grp)
    ksorted = Matrix(Kfull)                     # sorted-id order (labels = ids)

    fit(; kw...) = drm_bridge(; formula = "y ~ x + relmat(1 | id); sigma ~ 1",
                              family = "gaussian", data = dat, kw...)
    ref = fit(K = Kappear)
    # The bug: sorted order without labels pairs the wrong rows.
    @test abs(fit(K = ksorted)["loglik"] - ref["loglik"]) > 1e-3
    # The fix: same matrix, labelled -> identical fit to first-appearance order.
    al = fit(K = ksorted, K_labels = ids, structured_group = "id")
    @test al["loglik"] ≈ ref["loglik"] atol = 1e-8
    # Shuffled labelled input agrees too.
    p = randperm(rng, G)
    sh = fit(K = ksorted[p, p], K_labels = ids[p], structured_group = "id")
    @test sh["loglik"] ≈ ref["loglik"] atol = 1e-8

    # animal(): same contract through `A`.
    fa = drm_bridge(; formula = "y ~ x + animal(1 | id); sigma ~ 1",
                    family = "gaussian", data = dat, A = ksorted,
                    A_labels = ids, structured_group = "id")
    fa_ref = drm_bridge(; formula = "y ~ x + animal(1 | id); sigma ~ 1",
                        family = "gaussian", data = dat, A = Kappear)
    @test fa["loglik"] ≈ fa_ref["loglik"] atol = 1e-8

    # Refusals.
    @test_throws ArgumentError fit(K = ksorted[1:G-1, 1:G-1], K_labels = ids[1:G-1],
                                   structured_group = "id")      # missing level
    @test_throws ArgumentError fit(K = ksorted, K_labels = vcat(ids[1:G-1], ids[1]),
                                   structured_group = "id")      # duplicate label
    @test_throws ArgumentError fit(K = ksorted, K_labels = ids[1:G-1],
                                   structured_group = "id")      # length mismatch
    @test_throws ArgumentError fit(K = ksorted, K_labels = ids)  # no group
    @test_throws ArgumentError fit(K_labels = ids, structured_group = "id")
    @test_throws ArgumentError fit(K = ksorted, K_labels = ids, structured_group = "nope")
end

@testset "drm_bridge labelled K with numeric ids (drmTMB#1512)" begin
    # R sends numeric ids as Float64 while its row names are strings: id 1 is
    # `1.0` here and "1" in R, and 100000 is `100000.0` here but "1e+05" under
    # R's as.character(). Both must match the plain-integer string.
    rng = MersenneTwister(15121)
    ids = [100000, 2, 7, 31, 4]
    G, nrep = length(ids), 3
    appear = ids[randperm(rng, G)]
    B = randn(rng, G, G)
    Kfull = Matrix(Symmetric(B * B' / G + I))      # indexed by `ids`
    ord = [findfirst(==(i), ids) for i in appear]
    Kappear = Kfull[ord, ord]
    u = cholesky(Symmetric(Kappear)).L * randn(rng, G)
    gid = repeat(appear, inner = nrep)
    x = randn(rng, G * nrep)
    y = 0.4 .+ 0.3 .* x .+ repeat(u, inner = nrep) .+ 0.7 .* randn(rng, G * nrep)
    labels = string.(ids)                          # what R sends: "100000", "2", ...

    for idcol in (Float64.(gid), gid)              # Float64 (from R) and Int
        dat = (; y, x, id = idcol)
        run(; kw...) = drm_bridge(; formula = "y ~ x + relmat(1 | id); sigma ~ 1",
                                  family = "gaussian", data = dat, kw...)
        ref = run(K = Kappear)
        al = run(K = Kfull, K_labels = labels, structured_group = "id")
        @test al["loglik"] ≈ ref["loglik"] atol = 1e-8
    end

    @test DRModels._group_label(1.0) == "1"
    @test DRModels._group_label(100000.0) == "100000"
    @test DRModels._group_label(2.5) == "2.5"
    @test DRModels._group_label("a") == "a"
    # R-boxed labels: Vector{Any} of strings, and a bare String for one level.
    @test DRModels._align_labelled_matrix(ones(1, 1), "only", ["only", "only"], "K") == ones(1, 1)
    @test DRModels._align_labelled_matrix(Kfull, Any[labels...], Float64.(gid), "K") ≈ Kappear
end

@testset "animal(1|id) A order: first appearance vs sorted + A_labels (#957)" begin
    # The animal-models tutorial documents that `A` must follow the first
    # appearance of `id` in the data. A sorted-id A fitted unlabelled pairs the
    # wrong rows (silently different logLik); with A_labels it matches.
    rng = MersenneTwister(957)
    G, nrep = 15, 4
    ids = ["a$(lpad(i, 2, '0'))" for i in 1:G]          # sorted order
    appear = ids[randperm(rng, G)]                      # first-appearance order
    B = randn(rng, G, G)
    Asorted = Matrix(Symmetric(B * B' / G + I))         # rows/cols follow `ids`
    ord = [findfirst(==(l), ids) for l in appear]
    Aappear = Asorted[ord, ord]
    u = 0.8 .* (cholesky(Symmetric(Aappear)).L * randn(rng, G))
    id = repeat(appear, inner = nrep)
    x = randn(rng, G * nrep)
    y = 0.3 .+ 0.5 .* x .+ repeat(u, inner = nrep) .+ 0.4 .* randn(rng, G * nrep)
    dat = (; y, x, id)
    f = "y ~ x + animal(1 | id); sigma ~ 1"
    fit(; kw...) = drm_bridge(; formula = f, family = "gaussian", data = dat, kw...)

    ref = fit(A = Aappear)                                          # documented order
    lab = fit(A = Asorted, A_labels = ids, structured_group = "id") # sorted + labels
    @test lab["loglik"] ≈ ref["loglik"] atol = 1e-8
    # Direct drm(): `ids =` labels the sorted A (the route the tutorial names).
    fd = bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1))
    ll_direct(; kw...) = loglik(drm(fd, Gaussian(); data = dat, kw...))
    @test ll_direct(A = Asorted, ids = ids) ≈ ll_direct(A = Aappear) atol = 1e-8
    @test ll_direct(A = Asorted, ids = ids) ≈ ref["loglik"] atol = 1e-6
    # The documented warning is justified: sorted A without labels differs.
    @test abs(fit(A = Asorted)["loglik"] - ref["loglik"]) > 1e-3

    # The one-line way to build the first-appearance order, as shown in the tutorial.
    @test unique(id) == appear
    p = indexin(unique(id), ids)
    @test Asorted[p, p] == Aappear
    @test ll_direct(A = Asorted[p, p]) ≈ ll_direct(A = Aappear) atol = 1e-8
end
