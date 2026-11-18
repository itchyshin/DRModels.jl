# Gaussian structured sigma intercept with an SD formula, phylo and spatial
# providers (twin of the phylo/spatial cases of drmTMB
# test-gaussian-sigma-structured-sd.R; both fit natively in drmTMB 0.8.1):
#   y ~ x,  sigma ~ 1 + phylo(1 | id)    + sd(id, phylogenetic, sigma) ~ z
#   y ~ x,  sigma ~ 1 + spatial(1 | id)  + sd(id, spatial, sigma) ~ z
# phylo K = the tree's correlation matrix; spatial K = the native FIXED kernel
# exp(-d / median positive distance) + 1e-6 I. Oracle: an independent dense
# Laplace over the raw sigma effects u ~ N(0, diag(s) K diag(s)), with K built
# here from the generated tree's own MRCA depths (not from DRModels' tree code)
# or from the coordinates, ForwardDiff derivatives and its own Newton search.
# Tolerances fixed before the first run: |logLik - oracle| < 1e-5; oracle mode
# |grad|_inf < 1e-6, Hessian PD; constant-SD fit equals the oracle with a
# constant SD to 1e-5.
using DRModels, Random, Test, LinearAlgebra, ForwardDiff
import Distributions
using Distributions: Normal, NegativeBinomial, logpdf
using Statistics: median

# Random ultrametric tree on `ntip` labelled tips: Newick string plus the
# shared-path (MRCA depth) covariance C, from the recursion itself.
function _gps_tree(rng, labels; height = 1.0)
    function build(labs, depth)
        if length(labs) == 1
            return labs[1] * ":" * string(height - depth), Dict((labs[1], labs[1]) => height)
        end
        local k, dn, nl, Cl, nr, Cr, C        # do not capture the outer names
        k = rand(rng, 1:(length(labs) - 1))
        dn = depth + (height - depth) * rand(rng) * 0.6           # child node depth
        (nl, Cl) = build(labs[1:k], dn); (nr, Cr) = build(labs[(k+1):end], dn)
        C = merge(Cl, Cr)
        for a in labs[1:k], b in labs[(k+1):end]
            C[(a, b)] = dn; C[(b, a)] = dn
        end
        return "(" * nl * "," * nr * "):" * string(dn - depth), C
    end
    k = rand(rng, 1:(length(labels) - 1))                      # root at depth 0
    (nl, Cl) = build(labels[1:k], 0.0); (nr, Cr) = build(labels[(k+1):end], 0.0)
    C = merge(Cl, Cr)
    for a in labels[1:k], b in labels[(k+1):end]
        C[(a, b)] = 0.0; C[(b, a)] = 0.0
    end
    newick = "(" * nl * "," * nr * ");"
    G = length(labels)
    Cm = [C[(labels[i], labels[j])] for i in 1:G, j in 1:G]
    return newick, Cm
end

function _gps_fixture(provider::Symbol; seed, n_g = 48, n_each = 20)
    rng = MersenneTwister(seed)
    ids = [string("s", lpad(string(j), 2, '0')) for j in 1:n_g]
    tree = nothing; coords = nothing
    K = if provider === :phylo
        tree, C = _gps_tree(rng, shuffle(rng, ids))
        Cl = similar(C)
        # C is in the shuffled tip order; reorder to ids order through labels
        lab_order = [m.match for m in eachmatch(r"s\d\d", tree)]
        pos = Dict(l => i for (i, l) in enumerate(lab_order))
        P = [pos[l] for l in ids]
        C = C[P, P]
        C ./ sqrt.(diag(C) * diag(C)')
    else
        coords = hcat(repeat(1:8; inner = 8)[1:n_g], repeat(1:8; outer = 8)[1:n_g]) .* 1.0
        D = [norm(coords[i, :] .- coords[j, :]) for i in 1:n_g, j in 1:n_g]
        rng_ = median([D[i, j] for i in 1:n_g for j in (i+1):n_g if D[i, j] > 0])
        exp.(-D ./ rng_) + 1e-6 * I
    end
    z = collect(range(-1, 1; length = n_g))
    u = cholesky(Symmetric(K)).L * randn(rng, n_g)
    alpha = [log(0.45), 0.5]
    gi = repeat(1:n_g; inner = n_each)
    x = randn(rng, length(gi))
    sc = exp.(alpha[1] .+ alpha[2] .* z)
    y = 0.2 .+ 0.5 .* x .+ exp.(log(0.5) .+ sc[gi] .* u[gi]) .* randn(rng, length(gi))
    data = (; y, x, id = ids[gi], z = z[gi])
    return (; data, K, z, gi, tree, coords, ids)
end

function _gps_oracle(y, mu, s0, gi, K, sd_b)
    G = length(sd_b)
    Sigma = Symmetric(sd_b .* K .* sd_b')
    Cs = cholesky(Sigma)
    negjoint(u) = begin
        s = zero(eltype(u))
        for i in eachindex(y)
            s -= logpdf(Normal(mu[i], exp(s0 + u[gi[i]])), y[i])
        end
        q = Cs.L \ u
        s + 0.5 * dot(q, q) + 0.5 * logdet(Cs) + 0.5 * G * log(2π)
    end
    u = zeros(G)
    for _ in 1:200
        g = ForwardDiff.gradient(negjoint, u)
        maximum(abs, g) < 1e-11 && break
        H = ForwardDiff.hessian(negjoint, u)
        step = H \ g; t = 1.0; f0 = negjoint(u)
        while t > 1e-12 && negjoint(u .- t .* step) > f0; t /= 2; end
        u .-= t .* step
    end
    H = ForwardDiff.hessian(negjoint, u)
    return (; loglik = -negjoint(u) + 0.5 * G * log(2π) - 0.5 * logdet(cholesky(Symmetric(H))),
            max_grad = maximum(abs, ForwardDiff.gradient(negjoint, u)),
            min_eig = minimum(eigvals(Symmetric(H))))
end

_gps_formula(p, sdrhs) = p === :phylo ?
    bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 | id)), sdrhs === :z ?
       @formula(sd(id, phylogenetic, sigma) ~ z) : @formula(sd(id, phylogenetic, sigma) ~ 1)) :
    bf(@formula(y ~ x), @formula(sigma ~ 1 + spatial(1 | id)), sdrhs === :z ?
       @formula(sd(id, spatial, sigma) ~ z) : @formula(sd(id, spatial, sigma) ~ 1))

_gps_fit(fx, p, sdrhs = :z) = p === :phylo ?
    drm(_gps_formula(p, sdrhs), Gaussian(); data = fx.data, tree = fx.tree) :
    drm(_gps_formula(p, sdrhs), Gaussian(); data = fx.data, coords = fx.coords)

for (p, seed) in ((:phylo, 2026110711), (:spatial, 2026110712))
    @testset "Gaussian $p sigma intercept with SD formula matches a dense Laplace oracle" begin
        fx = _gps_fixture(p; seed)
        fit = _gps_fit(fx, p)
        @test fit.converged
        @test all(isfinite, stderror(fit))
        key = only(first(b) for b in fit.blocks if first(b) ∉ (:mu, :sigma))
        al = coef(fit, key)
        X = hcat(ones(length(fx.data.y)), fx.data.x)
        o = _gps_oracle(fx.data.y, X * coef(fit, :mu), coef(fit, :sigma)[1], fx.gi, fx.K,
                        exp.(al[1] .+ al[2] .* fx.z))
        @test o.max_grad < 1e-6
        @test o.min_eig > 0
        @test abs(loglik(fit) - o.loglik) < 1e-5
        # constant-SD limit
        fit1 = _gps_fit(fx, p, :one)
        a1 = coef(fit1, only(first(b) for b in fit1.blocks if first(b) ∉ (:mu, :sigma)))
        o1 = _gps_oracle(fx.data.y, X * coef(fit1, :mu), coef(fit1, :sigma)[1], fx.gi, fx.K,
                         fill(exp(a1[1]), length(fx.z)))
        @test abs(loglik(fit1) - o1.loglik) < 1e-5
    end
end

@testset "phylo/spatial sigma SD: refusals" begin
    fx = _gps_fixture(:phylo; seed = 2026110711)
    @test_throws ArgumentError drm(_gps_formula(:phylo, :z), Gaussian(); data = fx.data)   # no tree
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 | id)),
                                      @formula(sd(id, spatial, sigma) ~ z)), Gaussian();
                                   data = fx.data, tree = fx.tree)                       # level mismatch
    fs = _gps_fixture(:spatial; seed = 2026110712)
    @test_throws ArgumentError drm(_gps_formula(:spatial, :z), Gaussian(); data = fs.data)  # no coords
end
