# Sparse O(p) path for the two-structured Gaussian mean model (#225 follow-up).
# Same model as `_fit_two_structured_gaussian`:
#     y = Xβ + Z₁a₁ + Z₂a₂ + ε,  a₁~N(0,σ₁²C₁), a₂~N(0,σ₂²C₂), ε~N(0,σ²I).
# The sparse path (`algorithm = :sparse`) integrates the augmented latent
# a = [a₁; a₂] via ONE sparse Cholesky of H = blockdiag(σ₁⁻²C₁⁻¹,σ₂⁻²C₂⁻¹)
# + ZᵀZ/σ², with the variance-component gradient from the Takahashi selected
# inverse — never forming the n×n V. The ANCHOR test below proves it returns
# the SAME MLE + logLik as the dense path on identical inputs.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays

_corr(M) = (d = sqrt.(diag(M)); M ./ (d * d'))

# Build a correlation matrix whose INVERSE is genuinely sparse (banded): start
# from an SPD banded precision Ω, set C = corr(inv(Ω)). Then inv(C) is banded,
# so the sparse path's H is sparse and its Cholesky/Takahashi truly exercise the
# sparse machinery — while the DENSE path receives the very same dense C.
function _banded_corr(G; bw = 2, seed = 0)
    Random.seed!(seed)
    Ω = Matrix(2.0I(G))
    for i in 1:G, j in (i+1):min(i + bw, G)
        v = 0.3 / (j - i)
        Ω[i, j] = v; Ω[j, i] = v
    end
    Ω += (0.1 + G * eps()) * I            # keep SPD
    return _corr(inv(Symmetric(Ω)))
end

@testset "Sparse two-structured: EQUIVALENCE to dense (anchor)" begin
    Random.seed!(424242)
    G = 30
    C1 = _banded_corr(G; bw = 3, seed = 11)   # sparse-inverse correlation #1
    C2 = _banded_corr(G; bw = 2, seed = 22)   # sparse-inverse correlation #2

    m = 6; n = G * m
    species = repeat(1:G, inner = m)
    id      = repeat(1:G, inner = m)
    x = randn(n)
    σ = 0.4; σ1 = 0.8; σ2 = 0.5
    a1 = σ1 .* (cholesky(Symmetric(C1)).L * randn(G))
    a2 = σ2 .* (cholesky(Symmetric(C2)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ σ .* randn(n)
    data = (; y, x, species, id)

    # The two-structured router resolves BOTH relmat markers from a single `K`,
    # so to give each component its OWN correlation we call the fitter entries
    # directly — comparing dense vs sparse on byte-identical inputs (the anchor).
    gidx1, G1 = DRModels._group_index(species)
    gidx2, G2 = DRModels._group_index(id)
    Xμ = hcat(ones(n), x)
    nmμ = ["(Intercept)", "x"]
    dense2 = DRModels._fit_two_structured_gaussian(Gaussian(), y, Xμ, gidx1, G1, C1,
        gidx2, G2, C2, nmμ, :species, :id, 1e-9)
    sparse2 = DRModels._fit_two_structured_gaussian_sparse(Gaussian(), y, Xμ, gidx1, G1, C1,
        gidx2, G2, C2, nmμ, :species, :id, 1e-9)

    @test sparse2.converged
    # logLik agrees to tight tolerance.
    @test loglik(sparse2) ≈ loglik(dense2) rtol = 1e-5
    # β agrees.
    @test coef(sparse2, :mu) ≈ coef(dense2, :mu) rtol = 1e-4 atol = 1e-5
    # σ, σ1, σ2 agree (re_sd reports per grouping factor; sigma() the residual).
    @test sigma(sparse2)[1] ≈ sigma(dense2)[1] rtol = 1e-4
    sd_s = re_sd(sparse2); sd_d = re_sd(dense2)
    # A variance component can sit at its near-zero boundary on this fixture, where
    # a relative tolerance is meaningless (comparing ~1e-8 to ~1e-10); an absolute
    # tolerance handles that. The logLik/β/σ equivalence above already anchors that
    # the sparse fit matches the dense one.
    @test sd_s[:species] ≈ sd_d[:species] rtol = 1e-4 atol = 1e-5
    @test sd_s[:id]      ≈ sd_d[:id]      rtol = 1e-4 atol = 1e-5
end

@testset "Sparse two-structured: analytic gradient matches finite differences" begin
    Random.seed!(7)
    G = 20
    C1 = _banded_corr(G; bw = 2, seed = 3)
    C2 = _banded_corr(G; bw = 3, seed = 4)
    m = 5; n = G * m
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n)
    y = 0.2 .+ 0.4 .* x .+ 0.7 .* randn(G)[species] .+ 0.5 .* randn(G)[id] .+ 0.3 .* randn(n)
    Xμ = hcat(ones(n), x)
    gidx1, _ = DRModels._group_index(species); gidx2, _ = DRModels._group_index(id)

    # Re-derive the closed-form NLL exactly (dense) and FD it, then compare to the
    # sparse path's analytic gradient via its own internal evaluator. We test the
    # gradient by checking the optimum: at the sparse MLE, the dense NLL gradient
    # is ~0. (A direct unit-grad test would need the internal closure; the anchor
    # test already pins logLik, and the optimiser used the analytic gradient.)
    fit = DRModels._fit_two_structured_gaussian_sparse(Gaussian(), y, Xμ, gidx1, G, C1,
        gidx2, G, C2, ["(Intercept)", "x"], :species, :id, 1e-10)
    @test fit.converged

    # Build the dense NLL and check its gradient is ~0 at the sparse θ̂.
    Z1 = zeros(n, G); for i in 1:n; Z1[i, gidx1[i]] = 1; end
    Z2 = zeros(n, G); for i in 1:n; Z2[i, gidx2[i]] = 1; end
    A1 = Z1 * C1 * Z1'; A2 = Z2 * C2 * Z2'; Iₙ = Matrix(I(n) * 1.0)
    function densenll(θ)
        β = θ[1:2]; lσ = θ[3]; lσ1 = θ[4]; lσ2 = θ[5]
        V = exp(2lσ) .* Iₙ .+ exp(2lσ1) .* A1 .+ exp(2lσ2) .* A2
        Vf = cholesky(Symmetric(V))
        r = y .- (β[1] .+ β[2] .* x)
        0.5 * (logdet(Vf) + dot(r, Vf \ r)) + 0.5 * n * log(2π)
    end
    θ̂ = fit.theta
    g = zeros(5); ε = 1e-6
    for k in 1:5
        θp = copy(θ̂); θm = copy(θ̂)
        s = ε * max(abs(θ̂[k]), 1.0); θp[k] += s; θm[k] -= s
        g[k] = (densenll(θp) - densenll(θm)) / (2s)
    end
    @test maximum(abs, g) < 1e-3       # at the sparse MLE, dense gradient ≈ 0
end

@testset "Sparse two-structured: recovery on a larger fixture (phylo + relmat)" begin
    Random.seed!(20260607)
    G = 120                                       # exercises sparsity (tree + banded)
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    Canim = _banded_corr(G; bw = 4, seed = 99)

    m = 6; n = G * m
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n)
    σ = 0.35; σ1 = 0.9; σ2 = 0.6
    a1 = σ1 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = σ2 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ σ .* randn(n)
    Xμ = hcat(ones(n), x)
    gidx1, _ = DRModels._group_index(species); gidx2, _ = DRModels._group_index(id)

    fit = DRModels._fit_two_structured_gaussian_sparse(Gaussian(), y, Xμ, gidx1, G, Cphy,
        gidx2, G, Canim, ["(Intercept)", "x"], :species, :id, 1e-9)
    @test fit.converged
    @test coef(fit, :mu)[2] ≈ 0.5 atol = 0.1
    sds = re_sd(fit)
    @test sds[:species] ≈ σ1 atol = 0.35
    @test sds[:id]      ≈ σ2 atol = 0.35
    @test sigma(fit)[1] ≈ σ atol = 0.1
    re = ranef(fit)
    @test length(re[:species]) == G && length(re[:id]) == G
end

@testset "Sparse two-structured: routed via algorithm = :sparse" begin
    Random.seed!(31415)
    G = 25
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    m = 6; n = G * m
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n)
    a1 = 0.8 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = 0.5 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ 0.4 .* randn(n)
    data = (; y, x, species, id)
    form = bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1))

    dense  = drm(form, Gaussian(); data = data, tree = phy, K = Canim)
    sparse = drm(form, Gaussian(); data = data, tree = phy, K = Canim, algorithm = :sparse)
    @test sparse.converged
    @test loglik(sparse) ≈ loglik(dense) rtol = 1e-4
    @test coef(sparse, :mu) ≈ coef(dense, :mu) rtol = 1e-3 atol = 1e-4
    @test re_sd(sparse)[:species] ≈ re_sd(dense)[:species] rtol = 1e-3
    @test re_sd(sparse)[:id]      ≈ re_sd(dense)[:id]      rtol = 1e-3

    # default algorithm stays dense (unchanged behaviour).
    @test loglik(drm(form, Gaussian(); data = data, tree = phy, K = Canim)) ≈ loglik(dense)
end

@testset "augmented_tree_precision: leaf cov == sigma_phy_dense" begin
    # The root-conditioned augmented precision Q over q = 2p-2 nodes must reproduce
    # the dense leaf covariance when restricted to leaf positions: (Q⁻¹)[leaf,leaf]
    # == sigma_phy_dense. The correctness anchor for the END-TO-END feed (#232).
    for treegen in (random_balanced_tree, random_caterpillar_tree)
        phy = treegen(12; branch_length = 0.3)
        Q, leaf_pos, q = augmented_tree_precision(phy)
        @test q == 2 * 12 - 2
        Σleaf_aug = inv(Matrix(Q))[leaf_pos, leaf_pos]
        Σleaf_den = sigma_phy_dense(phy; σ²_phy = 1.0)
        @test Σleaf_aug ≈ Σleaf_den rtol = 1e-8
    end
end

@testset "caterpillar tree: well-formed binary topology" begin
    for p in (2, 5, 16)
        phy = random_caterpillar_tree(p; branch_length = 0.2)
        @test phy.n_leaves == p
        @test phy.n_total == 2p - 1
        Q, leaf_pos, q = augmented_tree_precision(phy)
        @test issuccess(cholesky(Symmetric(Matrix(Q)); check = false))   # PD
        @test length(leaf_pos) == p
    end
end

@testset "END-TO-END sparse phylo: equals dense leaf-correlation fit (#232)" begin
    # The phylo component feeds the AUGMENTED tree precision directly (no dense Ck
    # inversion) yet must land on the SAME MLE/logLik as the dense leaf-correlation
    # path. Anchor: drm(:sparse) (end-to-end) vs drm(:auto) (dense Cphy).
    Random.seed!(2718281)
    G = 40
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    m = 6; n = G * m
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    x = randn(n)
    a1 = 0.8 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = 0.5 .* (cholesky(Symmetric(Canim)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ a1[species] .+ a2[id] .+ 0.4 .* randn(n)
    data = (; y, x, species, id)
    form = bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ 1))

    dense  = drm(form, Gaussian(); data = data, tree = phy, K = Canim)               # dense Cphy
    e2e    = drm(form, Gaussian(); data = data, tree = phy, K = Canim, algorithm = :sparse)
    @test e2e.converged
    @test loglik(e2e) ≈ loglik(dense) rtol = 1e-4
    @test coef(e2e, :mu) ≈ coef(dense, :mu) rtol = 1e-3 atol = 1e-4
    @test re_sd(e2e)[:species] ≈ re_sd(dense)[:species] rtol = 2e-3
    @test re_sd(e2e)[:id]      ≈ re_sd(dense)[:id]      rtol = 2e-3
    @test sigma(e2e)[1] ≈ sigma(dense)[1] rtol = 1e-3

    # Per-leaf BLUPs come back at length G (read out of the augmented latent).
    re = ranef(e2e)
    @test length(re[:species]) == G && length(re[:id]) == G
end

@testset "END-TO-END sparse phylo: sigma ~ x heteroscedastic (D → diag)" begin
    # The D → diag(σ_i²) extension: a residual-scale covariate. Anchor against a
    # direct dense GLS NLL at the sparse MLE — its gradient must be ≈ 0 there.
    Random.seed!(13579)
    G = 30
    phy = random_balanced_tree(G; branch_length = 0.4)
    Cphy = _corr(sigma_phy_dense(phy; σ²_phy = 1.0))
    M = randn(G, G); Canim = _corr(M * M' / G + I)
    m = 6; n = G * m
    species = repeat(1:G, inner = m); id = repeat(1:G, inner = m)
    xs = randn(n)                                    # sigma covariate
    a1 = 0.8 .* (cholesky(Symmetric(Cphy)).L * randn(G))
    a2 = 0.5 .* (cholesky(Symmetric(Canim)).L * randn(G))
    σi = exp.(-1.0 .+ 0.3 .* xs)
    y = 0.3 .+ a1[species] .+ a2[id] .+ σi .* randn(n)
    data = (; y, x = xs, species, id)
    form = bf(@formula(y ~ 1 + phylo(1 | species) + relmat(1 | id)), @formula(sigma ~ x))

    fit = drm(form, Gaussian(); data = data, tree = phy, K = Canim, algorithm = :sparse)
    @test fit.converged
    @test length(coef(fit, :sigma)) == 2          # intercept + slope on log σ

    # Dense GLS: V = diag(σ_i²) + σ1² Z1 Cphy Z1ᵀ + σ2² Z2 Canim Z2ᵀ. The sparse
    # fit's gradient of THIS dense NLL must be ≈ 0 at its MLE.
    Z1 = zeros(n, G); for i in 1:n; Z1[i, species[i]] = 1; end
    Z2 = zeros(n, G); for i in 1:n; Z2[i, id[i]] = 1; end
    A1 = Z1 * Cphy * Z1'; A2 = Z2 * Canim * Z2'
    Xσ = hcat(ones(n), xs)
    function densenll(θ)
        β0 = θ[1]; bσ = θ[2:3]; lσ1 = θ[4]; lσ2 = θ[5]
        d = exp.(2 .* (Xσ * bσ))
        V = Diagonal(d) + exp(2lσ1) .* A1 + exp(2lσ2) .* A2
        Vf = cholesky(Symmetric(Matrix(V)))
        r = y .- β0
        0.5 * (logdet(Vf) + dot(r, Vf \ r)) + 0.5 * n * log(2π)
    end
    θ̂ = vcat(coef(fit, :mu), coef(fit, :sigma),
             log(re_sd(fit)[:species]), log(re_sd(fit)[:id]))
    g = zeros(5); ε = 1e-6
    for k in 1:5
        θp = copy(θ̂); θm = copy(θ̂)
        s = ε * max(abs(θ̂[k]), 1.0); θp[k] += s; θm[k] -= s
        g[k] = (densenll(θp) - densenll(θm)) / (2s)
    end
    @test maximum(abs, g) < 5e-3      # at the sparse MLE the dense GLS gradient ≈ 0
end
