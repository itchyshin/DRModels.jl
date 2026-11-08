# REML for the Gaussian structured q2 SD-regression routes
#   y ~ x + KIND(1 + x | id),  sd(id, LEVEL, coef(intercept)) ~ z0,  sd(id, LEVEL, coef(x)) ~ z1,
# KIND in phylo, spatial, animal, relmat. Native drmTMB (REML = TRUE) admits exactly:
#   phylo (labelled or not, fixed-effect sigma of any shape), and spatial / animal /
#   relmat UNLABELLED with sigma ~ 1. Everything else must keep refusing.
#
# Independent oracle (shares no fitter code). At the fitted variance parameters it builds the
# dense marginal covariance V = diag(sigma^2) + sum_k (w_k w_k') o (s_k s_k' o C)[g, g]
# (+ the rho cross term for the labelled block) by hand, computes the GLS mean b-hat, and the
# restricted log-likelihood
#   l_R = log N(y | X b-hat, V) + (p/2) log(2 pi) - (1/2) log|X' V^-1 X|
# (beta_mu integrated out of exp(l_ML); this is the constant native drmTMB reports, because
# TMB's Laplace step over beta_mu is exact for a Gaussian). Checks:
#   reml_loglik(fit) == l_R, ml_loglik(fit) == the dense ML value at the same theta,
#   the fitted mean coefficients are the GLS ones, and l_R at theta-hat is not beaten by
#   perturbations of the variance parameters (stationarity from outside the package).
# Tolerances fixed before the first run: 1e-6 on log-likelihoods, 1e-6 on coefficients.
using DRModels, Test, Random, LinearAlgebra
import Distributions

const _RQ2_BITS = 5   # 32 tips

function _rq2_tree()
    G = 2^_RQ2_BITS
    ids = ["t$(lpad(i, 2, '0'))" for i in 1:G]
    nw(ix, root) = length(ix) == 1 ? "$(ids[ix[1]]):1.5" :
        "(" * nw(ix[1:end÷2], false) * "," * nw(ix[end÷2+1:end], false) * ")" * (root ? ";" : ":0.1")
    height = 1.5 + 0.1 * (_RQ2_BITS - 1)
    bits(i) = reverse(digits(i - 1; base = 2, pad = _RQ2_BITS))
    C = [i == j ? 1.0 : (findfirst(bits(i) .!= bits(j)) - 1) * 0.1 / height for i in 1:G, j in 1:G]
    return nw(collect(1:G), true), ids, C
end

function _rq2_grid(G)
    side = round(Int, sqrt(G))
    xy = [(Float64(i), Float64(j)) for j in 0:side-1 for i in 0:side-1]
    D = [hypot(xy[k][1] - xy[l][1], xy[k][2] - xy[l][2]) for k in 1:G, l in 1:G]
    rng = Distributions.median([D[k, l] for k in 1:G for l in k+1:G])
    return xy, exp.(-D ./ rng) + 1e-6 * I
end

function _rq2_fixture(C, ids; m = 10, seed = 20261108, rho = 0.0, sig_slope = 0.0)
    rng = MersenneTwister(seed)
    G = length(ids)
    z0 = repeat([-1.0, 1.0], G ÷ 2)
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    L = cholesky(Symmetric(C)).L
    n0 = randn(rng, G); n1 = randn(rng, G)
    u0 = exp.(log(0.6) .+ 0.4 .* z0) .* (L * n0)
    u1 = exp.(log(0.35) .- 0.4 .* z1) .* (L * (rho .* n0 .+ sqrt(1 - rho^2) .* n1))
    gi = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.2, 1.2; length = m)), G)
    sig = exp.(log(0.5) .+ sig_slope .* x)
    y = 0.3 .+ 0.35 .* x .+ u0[gi] .+ x .* u1[gi] .+ sig .* randn(rng, length(x))
    ord = randperm(rng, length(y))
    data = (; y = y[ord], x = x[ord], z0 = z0[gi][ord], z1 = z1[gi][ord], id = ids[gi][ord])
    return (; data, gi = gi[ord], z0, z1)
end

# theta pieces from the fit blocks: (beta, sigma coefficients, a0, a1, eta_rho or nothing)
function _rq2_parts(fit)
    b = Dict(first(p) => last(p) for p in fit.blocks)
    sdb = [last(p) for p in fit.blocks if startswith(String(first(p)), "sdtarget_")]
    eta = haskey(b, :recov) ? fit.theta[b[:recov]][1] : nothing
    return fit.theta[b[:mu]], fit.theta[b[:sigma]], fit.theta[sdb[1]], fit.theta[sdb[2]], eta
end

# Dense ML and REML values at the supplied variance parameters; beta from GLS.
function _rq2_oracle(y, X, Xs, x, gi, z0, z1, C, sigc, a0, a1, eta)
    n = length(y)
    s0 = exp.(a0[1] .+ a0[2] .* z0); s1 = exp.(a1[1] .+ a1[2] .* z1)
    V = [(s0[gi[i]] * s0[gi[j]] + x[i] * x[j] * s1[gi[i]] * s1[gi[j]]) * C[gi[i], gi[j]] for i in 1:n, j in 1:n]
    if eta !== nothing
        rho = 0.999999 * tanh(eta)
        V += [rho * (x[j] * s0[gi[i]] * s1[gi[j]] + x[i] * s1[gi[i]] * s0[gi[j]]) * C[gi[i], gi[j]] for i in 1:n, j in 1:n]
    end
    V += Diagonal(exp.(2 .* (Xs * sigc)))
    Vs = Symmetric(V)
    Vi = inv(Vs)
    beta = (X' * Vi * X) \ (X' * Vi * y)
    ml = Distributions.logpdf(Distributions.MvNormal(X * beta, Vs), y)
    p = size(X, 2)
    reml = ml + 0.5 * p * log(2π) - 0.5 * logdet(Symmetric(X' * Vi * X))
    return (; ml, reml, beta)
end

_rq2_formula(kind, correlated, sigx) = begin
    lab = correlated
    mu = kind === :phylo ? (lab ? @formula(y ~ x + phylo(1 + x | p | id)) : @formula(y ~ x + phylo(1 + x | id))) :
         kind === :spatial ? @formula(y ~ x + spatial(1 + x | id)) :
         kind === :animal ? (lab ? @formula(y ~ x + animal(1 + x | p | id)) : @formula(y ~ x + animal(1 + x | id))) :
                            @formula(y ~ x + relmat(1 + x | id))
    lev = Dict(:phylo => :phylogenetic, :spatial => :spatial, :animal => :animal, :relmat => :relmat)[kind]
    f0 = eval(:(@formula($(Symbol("sd"))(id, $lev, coef(intercept)) ~ z0)))
    f1 = eval(:(@formula($(Symbol("sd"))(id, $lev, coef(x)) ~ z1)))
    return bf(mu, sigx ? @formula(sigma ~ x) : @formula(sigma ~ 1), f0, f1)
end

@testset "Gaussian structured q2 SD regression: REML" begin
    nwk, ids, Cphy = _rq2_tree()
    G = length(ids)
    xy, Csp = _rq2_grid(G)
    coords = Dict(ids[k] => [xy[k]...] for k in eachindex(ids))
    Cani = Matrix(0.4 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :)))
    Cmap = Dict(:phylo => Cphy, :spatial => Csp, :animal => Cani, :relmat => Cani)
    # relmat `K` carries no labels, so it is given in first-appearance order of the (shuffled) rows.
    first_order(d) = [findfirst(==(v), ids) for v in unique(d.id)]
    kw(kind, d) = kind === :phylo ? (; tree = nwk) : kind === :spatial ? (; coords) :
                  kind === :animal ? (; A = Cani, ids = ids) :
                  (; K = Cani[first_order(d), first_order(d)])

    # (kind, labelled, heteroscedastic sigma) cells native drmTMB admits under REML
    admitted = [(:phylo, false, false), (:spatial, false, false), (:animal, false, false),
                (:relmat, false, false), (:phylo, true, false), (:phylo, false, true)]
    for (kind, lab, sigx) in admitted
        @testset "$kind labelled=$lab sigma~x=$sigx" begin
            C = Cmap[kind]
            fx = _rq2_fixture(C, ids; rho = lab ? 0.5 : 0.0, sig_slope = sigx ? 0.3 : 0.0)
            d = fx.data
            f = _rq2_formula(kind, lab, sigx)
            fit = drm(f, Gaussian(); data = d, method = :REML, kw(kind, d)...)
            ml = drm(f, Gaussian(); data = d, kw(kind, d)...)
            @test fit.converged
            @test fit.estim_method === :REML
            @test ml.estim_method === :ML
            beta, sigc, a0, a1, eta = _rq2_parts(fit)
            X = hcat(ones(length(d.y)), d.x)
            Xs = sigx ? hcat(ones(length(d.y)), d.x) : ones(length(d.y), 1)
            o = _rq2_oracle(d.y, X, Xs, d.x, fx.gi, fx.z0, fx.z1, C, sigc, a0, a1, eta)
            @test reml_loglik(fit) ≈ o.reml atol = 1e-6
            @test ml_loglik(fit) ≈ o.ml atol = 1e-6
            @test loglikelihood(fit) ≈ o.reml atol = 1e-6
            @test beta ≈ o.beta atol = 1e-6                       # GLS mean at the REML variances
            @test all(isfinite, stderror(fit))
            # Stationarity from outside the package: no perturbation of the variance
            # parameters improves the hand-built restricted likelihood.
            rng = MersenneTwister(99)
            for _ in 1:6
                s = 0.05
                p = (sigc .+ s .* randn(rng, length(sigc)), a0 .+ s .* randn(rng, 2),
                     a1 .+ s .* randn(rng, 2), eta === nothing ? nothing : eta + s * randn(rng))
                op = _rq2_oracle(d.y, X, Xs, d.x, fx.gi, fx.z0, fx.z1, C, p...)
                @test op.reml <= o.reml + 1e-7
            end
            # ML route untouched and different from REML.
            @test ml_loglik(ml) ≈ loglikelihood(ml) atol = 1e-12
            @test abs(loglikelihood(ml) - loglikelihood(fit)) > 1e-3
            # The restriction changes the variance estimates (REML is not ML relabelled).
            @test fit.theta != ml.theta
        end
    end

    @testset "refusals mirror native drmTMB" begin
        fx = _rq2_fixture(Cani, ids)
        d = fx.data
        # labelled non-phylo block: native REML refuses it
        @test_throws r"labelled \(correlated\)" drm(_rq2_formula(:animal, true, false), Gaussian();
            data = d, method = :REML, kw(:animal, d)...)
        # heteroscedastic sigma with a non-phylo provider: native REML refuses it
        for kind in (:animal, :relmat, :spatial)
            C = Cmap[kind]
            fxk = _rq2_fixture(C, ids)
            @test_throws r"constant residual scale" drm(_rq2_formula(kind, false, true), Gaussian();
                data = fxk.data, method = :REML, kw(kind, fxk.data)...)
        end
        # rank-deficient mean design
        dd = merge(d, (; x2 = 2 .* d.x))
        f = bf(@formula(y ~ x + x2 + animal(1 + x | id)), @formula(sigma ~ 1),
               @formula(sd(id, animal, coef(intercept)) ~ z0), @formula(sd(id, animal, coef(x)) ~ z1))
        @test_throws r"full-rank" drm(f, Gaussian(); data = dd, method = :REML, kw(:animal, dd)...)
    end
end
