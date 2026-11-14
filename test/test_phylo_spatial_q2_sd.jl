# Phylogenetic and coordinate-spatial independent intercept-plus-slope q2 blocks
# with one SD formula per coefficient (Gaussian, Poisson, NegBinomial2):
#   y ~ x + phylo|spatial(1 + x | g),  sd(g, phylogenetic|spatial, coef(intercept)) ~ z0,
#   sd(g, phylogenetic|spatial, coef(x)) ~ z1.
# Kernels (native drmTMB conventions): phylo = tip correlation of the tree
# (ape::vcv(tree, corr = TRUE)); spatial = exp(-d / median distance) + 1e-6 I.
# Oracles share no fitter code: the phylo correlation is built from the balanced
# tree's shared-path lengths by hand, the spatial kernel straight from the grid;
# Gaussian uses the exact dense MvNormal density and the count families an
# independent dense ForwardDiff Laplace in the standardised latent b ~ N(0, I2 ⊗ C).
# Tolerances (fixed before the first run): |loglik - oracle| < 1e-6 (Gaussian),
# < 1e-5 (Laplace; the inner solve tolerance dominates); fit converged.
using DRModels, Test, Random, LinearAlgebra
import Distributions, ForwardDiff
using SpecialFunctions: loggamma

const _PSQ2_BITS = 5                       # 32 tips, balanced binary tree

function _psq2_tree()
    G = 2^_PSQ2_BITS
    ids = ["t$(lpad(i, 2, '0'))" for i in 1:G]
    nw(ix, root) = length(ix) == 1 ? "$(ids[ix[1]]):1.5" :
        "(" * nw(ix[1:end÷2], false) * "," * nw(ix[end÷2+1:end], false) * ")" * (root ? ";" : ":0.1")
    height = 1.5 + 0.1 * (_PSQ2_BITS - 1)
    bits(i) = reverse(digits(i - 1; base = 2, pad = _PSQ2_BITS))
    C = [i == j ? 1.0 : (findfirst(bits(i) .!= bits(j)) - 1) * 0.1 / height for i in 1:G, j in 1:G]
    return nw(collect(1:G), true), ids, C
end

function _psq2_grid(G)
    side = round(Int, sqrt(G))
    xy = [(Float64(i), Float64(j)) for j in 0:side-1 for i in 0:side-1]
    D = [hypot(xy[k][1] - xy[l][1], xy[k][2] - xy[l][2]) for k in 1:G, l in 1:G]
    rng = Distributions.median([D[k, l] for k in 1:G for l in k+1:G])
    return xy, exp.(-D ./ rng) + 1e-6 * I
end

# Rows are shuffled, so data order differs from both tree and coordinate order.
function _psq2_fixture_generate(C, ids; family, m = 10, seed = 20261107)
    rng = MersenneTwister(seed)
    G = length(ids)
    z0 = repeat([-1.0, 1.0], G ÷ 2)
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    L = cholesky(Symmetric(C)).L
    a0, a1 = family === :gaussian ? ([log(0.6), 0.4], [log(0.35), -0.4]) :
                                    ([log(0.45), 0.3], [log(0.3), -0.3])
    u0 = exp.(a0[1] .+ a0[2] .* z0) .* (L * randn(rng, G))
    u1 = exp.(a1[1] .+ a1[2] .* z1) .* (L * randn(rng, G))
    gi = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.2, 1.2; length = m)), G)
    eta = 0.3 .+ 0.35 .* x .+ u0[gi] .+ x .* u1[gi]
    y = family === :gaussian ? eta .+ 0.5 .* randn(rng, length(eta)) :
        family === :poisson ? Float64.(rand.(rng, Distributions.Poisson.(exp.(eta)))) :
        Float64.(rand.(rng, Distributions.NegativeBinomial.(1 / 0.6^2, (1 / 0.6^2) ./ ((1 / 0.6^2) .+ exp.(eta)))))
    ord = randperm(rng, length(y))
    data = (; y = y[ord], x = x[ord], z0 = z0[gi][ord], z1 = z1[gi][ord], id = ids[gi][ord])
    return (; data, gi = gi[ord], z0, z1)
end

# The committed datasets (test/fixtures/q2_stable/psq2_*.tsv) are what the tests
# fit. They were produced by `_psq2_fixture_generate` on Julia 1.10.12. Julia 1.11
# changed the MersenneTwister seeding, so the same seed gives a DIFFERENT dataset on
# 1.12, and several draws of this 32-tip design sit on a weakly identified optimum
# (non-PD Hessian, rho at the boundary). Committing the data makes the fixture
# identical on every Julia version; the generator is kept for regeneration only.
# The phylo and spatial kernels differ, so the file is chosen by sum(C).
function _psq2_fixture(C, ids; family, kw...)
    kind = Dict(75.789 => "phylo", 428.265 => "spatial")[round(sum(C); digits = 3)]
    lines = readlines(joinpath(@__DIR__, "fixtures", "q2_stable", "psq2_$(family)_$(kind).tsv"))
    rows = [split(l, '\t') for l in lines[3:end]]
    col(j, T) = T === String ? String[r[j] for r in rows] : parse.(T, [r[j] for r in rows])
    data = (; y = col(1, Float64), x = col(2, Float64), z0 = col(3, Float64),
              z1 = col(4, Float64), id = col(5, String))
    gi = col(6, Int)
    z1g = zeros(length(ids)); z1g[gi] .= data.z1
    return (; data, gi, z0 = repeat([-1.0, 1.0], length(ids) ÷ 2), z1 = z1g)
end

function _psq2_formula(kind, fam)
    mu = kind === :phylo ? @formula(y ~ x + phylo(1 + x | id)) : @formula(y ~ x + spatial(1 + x | id))
    f0 = kind === :phylo ? @formula(sd(id, phylogenetic, coef(intercept)) ~ z0) :
                           @formula(sd(id, spatial, coef(intercept)) ~ z0)
    f1 = kind === :phylo ? @formula(sd(id, phylogenetic, coef(x)) ~ z1) :
                           @formula(sd(id, spatial, coef(x)) ~ z1)
    return fam isa Poisson ? bf(mu, f0, f1) : bf(mu, @formula(sigma ~ 1), f0, f1)
end

function _psq2_theta(fit)
    b = fit.blocks
    sdb = [last(p) for p in b if startswith(String(first(p)), "sdtarget_")]
    sig = [last(p) for p in b if first(p) === :sigma]
    beta = fit.theta[last(b[1])]
    logsig = isempty(sig) ? NaN : fit.theta[only(sig)][1]
    return beta, logsig, fit.theta[sdb[1]], fit.theta[sdb[2]]
end

function _psq2_gaussian_oracle(fit, fx, C)
    beta, logsig, a0, a1 = _psq2_theta(fit)
    d = fx.data; gi = fx.gi; n = length(d.y)
    s0 = exp.(a0[1] .+ a0[2] .* fx.z0); s1 = exp.(a1[1] .+ a1[2] .* fx.z1)
    V = [(s0[gi[i]] * s0[gi[j]] + d.x[i] * d.x[j] * s1[gi[i]] * s1[gi[j]]) * C[gi[i], gi[j]] for i in 1:n, j in 1:n]
    V += exp(2logsig) * I
    return Distributions.logpdf(Distributions.MvNormal(beta[1] .+ beta[2] .* d.x, Symmetric(V)), d.y)
end

function _psq2_laplace_oracle(fit, fx, C, fam)
    beta, logsig, a0, a1 = _psq2_theta(fit)
    d = fx.data; gi = fx.gi; G = size(C, 1)
    s0 = exp.(a0[1] .+ a0[2] .* fx.z0); s1 = exp.(a1[1] .+ a1[2] .* fx.z1)
    Ci = inv(Symmetric(C))
    k = fam isa NegBinomial2 ? exp(-2logsig) : NaN
    function joint(b)
        eta = beta[1] .+ beta[2] .* d.x .+ s0[gi] .* b[gi] .+ d.x .* s1[gi] .* b[G .+ gi]
        mu = exp.(eta)
        lp = fam isa Poisson ? sum(d.y .* eta .- mu .- loggamma.(d.y .+ 1)) :
            sum(loggamma.(d.y .+ k) .- loggamma(k) .- loggamma.(d.y .+ 1) .+
                k .* log.(k ./ (k .+ mu)) .+ d.y .* log.(mu ./ (k .+ mu)))
        return lp - 0.5 * (dot(b[1:G], Ci * b[1:G]) + dot(b[G+1:end], Ci * b[G+1:end]))
    end
    b = zeros(2G)
    for _ in 1:100
        g = ForwardDiff.gradient(joint, b); H = ForwardDiff.hessian(joint, b)
        step = -(Symmetric(H) \ g); b .+= step
        maximum(abs, step) < 1e-12 && break
    end
    H = -ForwardDiff.hessian(joint, b)
    return joint(b) - logdet(C) - 0.5 * logdet(Symmetric(H))   # logdet(I2 ⊗ C)/2 = logdet(C)
end

@testset "phylo/spatial (1 + x | g) q2 with coefficient SD formulas" begin
    nwk, ids, Cphy = _psq2_tree()
    xy, Csp = _psq2_grid(length(ids))
    coords = Dict(ids[k] => [xy[k]...] for k in eachindex(ids))
    for (kind, C) in ((:phylo, Cphy), (:spatial, Csp)),
        (fam, fsym) in ((Gaussian(), :gaussian), (Poisson(), :poisson), (NegBinomial2(), :nb2))
        @testset "$kind $fsym" begin
            fx = _psq2_fixture(C, ids; family = fsym)
            kw = kind === :phylo ? (; tree = nwk) : (; coords)
            fit = drm(_psq2_formula(kind, fam), fam; data = fx.data, kw...)
            @test fit.converged
            @test length(fit.theta) == (fam isa Poisson ? 6 : 7)
            @test all(isfinite, stderror(fit))
            oracle = fam isa Gaussian ? _psq2_gaussian_oracle(fit, fx, C) :
                     _psq2_laplace_oracle(fit, fx, C, fam)
            @test loglikelihood(fit) ≈ oracle atol = (fam isa Gaussian ? 1e-6 : 1e-5)
        end
    end

    fx = _psq2_fixture(Cphy, ids; family = :gaussian)
    f = _psq2_formula(:phylo, Gaussian())
    @testset "refusals" begin
        # Non-ultrametric tree: drmTMB's correlation scale requires ultrametricity.
        bad = replace(nwk, "t01:1.5" => "t01:2.5")
        @test_throws r"ultrametric" drm(f, Gaussian(); data = fx.data, tree = bad)
        @test_throws r"ultrametric" drm(f, Poisson(); data = merge(fx.data, (; y = round.(abs.(fx.data.y)))), tree = bad)
        # Wrong provider keywords (the entry guard refuses an unused `K`).
        @test_throws r"`K = …` was supplied" drm(f, Gaussian(); data = fx.data, tree = nwk, K = Cphy)
        fs = _psq2_formula(:spatial, Gaussian())
        @test_throws r"both `K = …` and `coords = …`" drm(fs, Gaussian(); data = fx.data, coords, K = Csp)
        # One SD formula is not enough for the q2 block.
        one = bf(@formula(y ~ x + phylo(1 + x | id)), @formula(sd(id, phylogenetic, coef(x)) ~ z1))
        @test_throws r"one SD formula for the intercept" drm(one, Gaussian(); data = fx.data, tree = nwk)
        @test_throws r"one SD formula for the intercept" drm(one, NegBinomial2();
            data = merge(fx.data, (; y = round.(abs.(fx.data.y)))), tree = nwk)
        # A tip with no rows.
        keep = fx.data.id .!= "t07"
        sub = map(v -> v[keep], fx.data)
        @test_throws r"every tip" drm(f, Gaussian(); data = sub, tree = nwk)
        # Gamma reads a phylo `coef(intercept)` target on `phylo(1 | g)` as the
        # selector-free intercept target: the same model, the same fit.
        ypos = exp.(0.2 .* fx.data.y)
        gdat = merge(fx.data, (; y = ypos))
        g_coef = drm(bf(@formula(y ~ x + phylo(1 | id)), @formula(sigma ~ 1),
            @formula(sd(id, phylogenetic, coef(intercept)) ~ z0)), Gamma(); data = gdat, tree = nwk)
        g_plain = drm(bf(@formula(y ~ x + phylo(1 | id)), @formula(sigma ~ 1),
            @formula(sd(id, phylogenetic) ~ z0)), Gamma(); data = gdat, tree = nwk)
        @test loglikelihood(g_coef) ≈ loglikelihood(g_plain) atol = 1e-8
        # Routes that are not implemented still refuse (no silent fallback). Binomial logit
        # now has this route (test_binomial_phylo_spatial_q2_sd.jl); the probit link,
        # which native drmTMB also refuses here, does not.
        yb = Float64.(fx.data.y .> 0.3)
        @test_throws ArgumentError drm(bf(@formula(y ~ x + phylo(1 + x | id)),
            @formula(sd(id, phylogenetic, coef(intercept)) ~ z0),
            @formula(sd(id, phylogenetic, coef(x)) ~ z1)), Binomial(link = :probit);
            data = merge(fx.data, (; y = yb)), tree = nwk)
    end
end
