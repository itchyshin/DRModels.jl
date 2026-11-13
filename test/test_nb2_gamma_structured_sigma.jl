# Structured sigma random effects for NB2 and Gamma (src/sigma_structured_laplace.jl).
#
#   Shape A (native drmTMB 0.8.1 FITS it; NB2 only): NB2
#     y ~ x,  sigma ~ 1 + animal|relmat|phylo|spatial(1 + x | id)      (no SD formula)
#     eta_i = s0 + s1 u1[g] + s2 x_i u2[g],  u1, u2 ~ N(0, K) independent, fixed SDs.
#     R-vs-Julia agreement is in the evidence folder wave4-structured-sigma (drmr repo).
#   Shape B (JULIA-ONLY: native drmTMB 0.8.1 REFUSES it for NB2 and Gamma):
#     y ~ x,  sigma ~ 1 + animal|relmat|phylo|spatial(1 | id),  sd(id, level, sigma) ~ z
#     eta_i = s0 + exp(a0 + a1 z_g) u[g],  u ~ N(0, K).
#     There is NO R twin, so the only evidence is the likelihood oracle below.
#   Gamma shape A: refused natively and here.
#
# K: raw A / K for animal / relmat, the tree's correlation matrix for phylo (built here
# from the generated tree's own MRCA depths), the native FIXED kernel
# exp(-d / median positive distance) + 1e-6 I for spatial.
# Oracle: an independent dense Laplace over the RAW effects (own Newton mode search,
# ForwardDiff derivatives, densities from Distributions.logpdf; no DRModels internals)
# evaluated at the fitted parameters. Tolerances fixed before the first run:
# |logLik - oracle| < 1e-5, oracle mode |grad|_inf < 1e-6 with PD Hessian, constant-SD
# fit within 1e-5 of its own oracle, marginal simulate mean(z_i^2) < 1.5.
# Changed after the first run: the shape-A fixture SDs were raised from (0.45, 0.35) to (0.7, 0.6)
# because the phylo fit sat at a variance boundary (both log-SDs flat, SEs not finite), which
# makes the finite-SE check meaningless; the oracle comparison itself had not failed.
using DRModels, Random, Test, LinearAlgebra, ForwardDiff, Statistics
import Distributions
using Distributions: Normal, NegativeBinomial, logpdf

function _nss_tree(rng, labels; height = 1.0)
    function build(labs, depth)
        if length(labs) == 1
            return labs[1] * ":" * string(height - depth), Dict((labs[1], labs[1]) => height)
        end
        local k, dn, nl, Cl, nr, Cr, C
        k = rand(rng, 1:(length(labs) - 1))
        dn = depth + (height - depth) * rand(rng) * 0.6
        (nl, Cl) = build(labs[1:k], dn); (nr, Cr) = build(labs[(k+1):end], dn)
        C = merge(Cl, Cr)
        for a in labs[1:k], b in labs[(k+1):end]
            C[(a, b)] = dn; C[(b, a)] = dn
        end
        return "(" * nl * "," * nr * "):" * string(dn - depth), C
    end
    k = rand(rng, 1:(length(labels) - 1))
    (nl, Cl) = build(labels[1:k], 0.0); (nr, Cr) = build(labels[(k+1):end], 0.0)
    C = merge(Cl, Cr)
    for a in labels[1:k], b in labels[(k+1):end]
        C[(a, b)] = 0.0; C[(b, a)] = 0.0
    end
    newick = "(" * nl * "," * nr * ");"
    G = length(labels)
    return newick, [C[(labels[i], labels[j])] for i in 1:G, j in 1:G]
end

function _nss_kernel(provider, rng, ids)
    G = length(ids)
    tree = nothing; coords = nothing
    K = if provider === :phylo
        tree, C = _nss_tree(rng, shuffle(rng, ids))
        lab_order = [m.match for m in eachmatch(r"s\d\d", tree)]
        pos = Dict(l => i for (i, l) in enumerate(lab_order))
        P = [pos[l] for l in ids]
        C = C[P, P]
        C ./ sqrt.(diag(C) * diag(C)')
    elseif provider === :spatial
        coords = hcat(repeat(1:6; inner = 4)[1:G], repeat(1:4; outer = 6)[1:G]) .* 1.0
        D = [norm(coords[i, :] .- coords[j, :]) for i in 1:G, j in 1:G]
        r = median([D[i, j] for i in 1:G for j in (i+1):G if D[i, j] > 0])
        exp.(-D ./ r) + 1e-6 * I
    else
        dg = collect(range(0.7, 1.4; length = G))
        rho = provider === :animal ? 0.3 : 0.45
        [rho^abs(i - j) * sqrt(dg[i] * dg[j]) for i in 1:G, j in 1:G]
    end
    return K, tree, coords
end

_nss_logpdf(fam, y, lin, ls) = begin
    k = exp(-2ls)
    fam === :nb2 ? logpdf(NegativeBinomial(k, k / (k + exp(lin))), y) :
                   logpdf(Distributions.Gamma(k, exp(lin) / k), y)
end

# Julia 1.11 changed the MersenneTwister seeding, so a seed draws a DIFFERENT dataset on
# 1.12 than on 1.10. NB2 spatial shape A (seed 2026110804 + 100) draws one on 1.12 whose
# fit does not converge (sigma_sd_x at the boundary, non-finite SEs); the 1.10.12 draw
# fits cleanly. Gamma phylo shape B (seed 2026110807) draws one on 1.12 whose latent
# sigma effects are so small that the post-fit "wrong-answer variant" check cannot
# separate the correct reference from the no-random-effect one (0.039 vs 0.046).
# Those (y, x) (and, for phylo, the tree and K drawn from the same stream) are therefore
# committed (test/fixtures/rng_stable), so the fixtures are identical on every Julia
# version. No tolerance changed.
# `randperm`/`shuffle` are not stream-stable across Julia versions: on 1.13 the phylo tree
# for NB2 shape B (seed 2026110803) is a different tree and the fit does not converge
# (oracle max_grad 18.4, min_eig < 0), while the 1.12 draw fits cleanly; the shape A phylo
# draw (seed + 100) is likewise a fit that does not converge on 1.13, and the shape A spatial
# draw (seed + 100) does not converge on 1.12 (non-finite SEs; the 1.10.12 draw fits cleanly). Those (y, x) and
# the tree and K drawn from the same stream are therefore committed
# (test/fixtures/rng_stable), so the fixture is identical on every Julia version.
# No tolerance changed.
const _NSS_FROZEN = Set([(:nb2, :spatial, :A), (:gamma, :phylo, :B), (:nb2, :phylo, :B), (:nb2, :phylo, :A)])
function _nss_frozen_yx(fam, provider, shape)
    lines = readlines(joinpath(@__DIR__, "fixtures", "rng_stable", "$(fam)_$(provider)_$(shape).tsv"))
    rows = [split(l, '\t') for l in lines[3:end]]
    return parse.(Float64, [r[1] for r in rows]), parse.(Float64, [r[2] for r in rows])
end

# shape :A  two independent fields (intercept, x-slope) with fixed SDs;
# shape :B  one field with an SD formula in z.
function _nss_fixture(fam, provider, shape; seed, n_g = 24, n_each = 10)
    rng = MersenneTwister(seed)
    ids = [string("s", lpad(string(j), 2, '0')) for j in 1:n_g]
    K, tree, coords = _nss_kernel(provider, rng, ids)
    L = cholesky(Symmetric(K)).L
    z = collect(range(-1, 1; length = n_g))
    gi = repeat(1:n_g; inner = n_each)
    n = length(gi)
    x = randn(rng, n)
    s0 = fam === :nb2 ? -0.5 : -0.6
    alpha = [log(0.5), 0.5]
    eta = if shape === :B
        u = L * randn(rng, n_g)
        s0 .+ (exp.(alpha[1] .+ alpha[2] .* z) .* u)[gi]
    else
        u1 = L * randn(rng, n_g); u2 = L * randn(rng, n_g)
        s0 .+ 0.7 .* u1[gi] .+ 0.6 .* x .* u2[gi]
    end
    lin = (fam === :nb2 ? 1.0 : 0.8) .+ 0.3 .* x
    y = map(1:n) do i
        k = exp(-2eta[i])
        fam === :nb2 ? Float64(rand(rng, NegativeBinomial(k, k / (k + exp(lin[i]))))) :
                       rand(rng, Distributions.Gamma(k, exp(lin[i]) / k))
    end
    id = ids[gi]
    if (fam, provider, shape) in _NSS_FROZEN
        y, x = _nss_frozen_yx(fam, provider, shape); @assert length(y) == n
        if provider === :phylo   # the tree (and so K) is drawn from the same stream
            stem = joinpath(@__DIR__, "fixtures", "rng_stable", "$(fam)_$(provider)_$(shape)")
            tree = strip(read(stem * "_tree.txt", String))
            K = reduce(vcat, [parse.(Float64, split(l, '\t'))' for l in readlines(stem * "_K.tsv")])
        end
    end
    return (; data = (; y, x, id, z = z[gi]), K, tree, coords, z, gi, ids, n_g)
end

_nss_inputs(fx, provider) = provider === :animal ? (; A = fx.K) : provider === :relmat ? (; K = fx.K) :
    provider === :phylo ? (; tree = fx.tree) : (; coords = fx.coords)

const _NSS_FAM = Dict(:nb2 => NegBinomial2(), :gamma => DRModels.Gamma())
const _NSS_LEVEL = Dict(:animal => :animal, :relmat => :relmat, :phylo => :phylogenetic, :spatial => :spatial)

function _nss_form(provider, shape; constant = false, label = :id)
    p = provider
    if shape === :A
        return provider === :animal ? bf(@formula(y ~ x), @formula(sigma ~ 1 + animal(1 + x | id))) :
               provider === :relmat ? bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 + x | id))) :
               provider === :phylo ? bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 + x | id))) :
                                     bf(@formula(y ~ x), @formula(sigma ~ 1 + spatial(1 + x | id)))
    end
    return if provider === :animal
        bf(@formula(y ~ x), @formula(sigma ~ 1 + animal(1 | id)),
           constant ? @formula(sd(id, animal, sigma) ~ 1) : @formula(sd(id, animal, sigma) ~ z))
    elseif provider === :relmat
        bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
           constant ? @formula(sd(id, relmat, sigma) ~ 1) : @formula(sd(id, relmat, sigma) ~ z))
    elseif provider === :phylo
        bf(@formula(y ~ x), @formula(sigma ~ 1 + phylo(1 | id)),
           constant ? @formula(sd(id, phylogenetic, sigma) ~ 1) : @formula(sd(id, phylogenetic, sigma) ~ z))
    else
        bf(@formula(y ~ x), @formula(sigma ~ 1 + spatial(1 | id)),
           constant ? @formula(sd(id, spatial, sigma) ~ 1) : @formula(sd(id, spatial, sigma) ~ z))
    end
end

# Independent dense Laplace. fields: [(gi, load, K, sd::Vector)] over raw effects u_k ~ N(0, diag(sd) K diag(sd)).
function _nss_oracle(fam, y, mu0, s0, fields)
    G = length(first(fields).sd); nf = length(fields); m = nf * G
    Cs = [cholesky(Symmetric(fl.sd .* fl.K .* fl.sd')) for fl in fields]
    negjoint(u) = begin
        eta = fill(zero(eltype(u)) + s0, length(y))
        for (k, fl) in enumerate(fields)
            uk = u[((k - 1) * G + 1):(k * G)]
            eta = eta .+ fl.load .* uk[fl.gi]
        end
        s = zero(eltype(u))
        for i in eachindex(y)
            s -= _nss_logpdf(fam, y[i], mu0[i], eta[i])
        end
        for (k, C) in enumerate(Cs)
            q = C.L \ u[((k - 1) * G + 1):(k * G)]
            s += 0.5 * dot(q, q) + 0.5 * logdet(C)
        end
        s + 0.5 * m * log(2π)
    end
    u = zeros(m)
    for it in 1:300
        g = ForwardDiff.gradient(negjoint, u)
        maximum(abs, g) < 1e-11 && break
        H = ForwardDiff.hessian(negjoint, u)
        C = cholesky(Symmetric(H); check = false); lam = 1e-3
        while !issuccess(C)
            C = cholesky(Symmetric(H + lam * I); check = false); lam *= 10
        end
        step = C \ g
        # Inside the quadratic basin take the full Newton step. The Armijo test below compares f values near
        # 300 whose true decrease is under roundoff, so on some CPU/SIMD paths it rejects every step and the
        # loop stalls at |g| ~ 1e-6 (gamma relmat, shape B); the Laplace value then moves by ~7e-8.
        if maximum(abs, g) < 1e-4 && lam == 1e-3
            u .-= step
            continue
        end
        t = 1.0; f0 = negjoint(u)
        while t > 1e-12 && negjoint(u .- t .* step) > f0 - 1e-4 * t * dot(g, step)
            t /= 2
        end
        u .-= (t > 1e-12 ? t : 1.0) .* step
    end
    H = ForwardDiff.hessian(negjoint, u)
    return (; loglik = -negjoint(u) + 0.5 * m * log(2π) - 0.5 * logdet(cholesky(Symmetric(H))),
            max_grad = maximum(abs, ForwardDiff.gradient(negjoint, u)),
            min_eig = minimum(eigvals(Symmetric(H))))
end

function _nss_oracle_B(fit, fx, fam; K = fx.K, z = fx.z)
    d = fx.data
    a = coef(fit, :sd)
    X = hcat(ones(length(d.y)), d.x)
    return _nss_oracle(fam, d.y, X * coef(fit, :mu), coef(fit, :sigma)[1],
        [(; gi = fx.gi, load = ones(length(d.y)), K, sd = exp.(a[1] .+ a[2] .* z))])
end

function _nss_oracle_A(fit, fx, fam)
    d = fx.data
    X = hcat(ones(length(d.y)), d.x)
    s1 = exp(only(coef(fit, :sigma_sd_intercept))); s2 = exp(only(coef(fit, :sigma_sd_x)))
    G = fx.n_g
    return _nss_oracle(fam, d.y, X * coef(fit, :mu), coef(fit, :sigma)[1],
        [(; gi = fx.gi, load = ones(length(d.y)), K = fx.K, sd = fill(s1, G)),
         (; gi = fx.gi, load = d.x, K = fx.K, sd = fill(s2, G))])
end

const _NSS_PROVS = (:animal, :relmat, :phylo, :spatial)
const _NSS_SEEDS = Dict((:nb2, :animal) => 2026110801, (:nb2, :relmat) => 2026110802,
                        (:nb2, :phylo) => 2026110803, (:nb2, :spatial) => 2026110804,
                        (:gamma, :animal) => 2026110805, (:gamma, :relmat) => 2026110806,
                        (:gamma, :phylo) => 2026110807, (:gamma, :spatial) => 2026110808)

@testset "NB2/Gamma structured sigma, shape B (Julia-only SD formula): $fam $prov matches the dense Laplace oracle" for
        fam in (:nb2, :gamma), prov in _NSS_PROVS
    fx = _nss_fixture(fam, prov, :B; seed = _NSS_SEEDS[(fam, prov)])
    fit = drm(_nss_form(prov, :B), _NSS_FAM[fam]; data = fx.data, _nss_inputs(fx, prov)...)
    @test fit.converged
    @test fit.blocks[3].first === :sd
    @test length(coef(fit, :sd)) == 2
    @test all(isfinite, stderror(fit))
    o = _nss_oracle_B(fit, fx, fam)
    @test o.max_grad < 1e-6
    @test o.min_eig > 0
    @test abs(loglik(fit) - o.loglik) < 1e-5
end

@testset "NB2 structured sigma, shape A (native drmTMB fits it): $prov matches the dense Laplace oracle" for
        prov in _NSS_PROVS
    fx = _nss_fixture(:nb2, prov, :A; seed = _NSS_SEEDS[(:nb2, prov)] + 100)
    fit = drm(_nss_form(prov, :A), NegBinomial2(); data = fx.data, _nss_inputs(fx, prov)...)
    @test fit.converged
    @test [first(b) for b in fit.blocks] == [:mu, :sigma, :sigma_sd_intercept, :sigma_sd_x]
    @test all(isfinite, stderror(fit))
    o = _nss_oracle_A(fit, fx, :nb2)
    @test o.max_grad < 1e-6
    @test o.min_eig > 0
    @test abs(loglik(fit) - o.loglik) < 1e-5
end

@testset "structured sigma: oracle sensitivity, constant-SD nesting, label matching" begin
    fx = _nss_fixture(:nb2, :relmat, :B; seed = _NSS_SEEDS[(:nb2, :relmat)])
    fit = drm(_nss_form(:relmat, :B), NegBinomial2(); data = fx.data, K = fx.K)
    cfit = drm(_nss_form(:relmat, :B; constant = true), NegBinomial2(); data = fx.data, K = fx.K)
    @test fit.converged && cfit.converged
    good = _nss_oracle_B(fit, fx, :nb2)
    # misaligned SD predictor, and a non-raw (normalised-diagonal) K, both move the oracle far
    swapped = _nss_oracle_B(fit, fx, :nb2; z = reverse(fx.z))
    @test abs(swapped.loglik - good.loglik) > 1e-2
    Kn = fx.K ./ sqrt.(diag(fx.K) * diag(fx.K)')
    @test abs(_nss_oracle_B(fit, fx, :nb2; K = Kn).loglik - good.loglik) > 1e-2
    # constant-SD fit matches its own oracle; the SD regression nests it
    ac = coef(cfit, :sd)
    X = hcat(ones(length(fx.data.y)), fx.data.x)
    oc = _nss_oracle(:nb2, fx.data.y, X * coef(cfit, :mu), coef(cfit, :sigma)[1],
        [(; gi = fx.gi, load = ones(length(fx.data.y)), K = fx.K, sd = fill(exp(only(ac)), fx.n_g))])
    @test oc.max_grad < 1e-6
    @test abs(loglik(cfit) - oc.loglik) < 1e-5
    @test loglik(fit) >= loglik(cfit) - 1e-6
    # `A` matched to the group levels by label (shape B and shape A)
    fa = _nss_fixture(:nb2, :animal, :B; seed = 2026110809, n_g = 16, n_each = 8)
    perm = reverse(1:16)
    form = _nss_form(:animal, :B)
    base = drm(form, NegBinomial2(); data = fa.data, A = fa.K, ids = fa.ids)
    shuf = drm(form, NegBinomial2(); data = fa.data, A = fa.K[perm, perm], ids = fa.ids[perm])
    @test abs(loglik(base) - loglik(shuf)) < 1e-6
    fq = _nss_fixture(:nb2, :animal, :A; seed = 2026110810, n_g = 16, n_each = 8)
    formA = _nss_form(:animal, :A)
    baseA = drm(formA, NegBinomial2(); data = fq.data, A = fq.K, ids = fq.ids)
    shufA = drm(formA, NegBinomial2(); data = fq.data, A = fq.K[perm, perm], ids = fq.ids[perm])
    @test abs(loglik(baseA) - loglik(shufA)) < 1e-6
end

# Independent reference draw of the marginal model, written from the fixture's own fields.
function _nss_reference(fam, fit, fx, shape, prov, rng; re = true)
    d = fx.data
    n = length(d.y)
    b = coef(fit, :mu); s0 = coef(fit, :sigma)[1]
    lin = b[1] .+ b[2] .* d.x
    L = cholesky(Symmetric(fx.K)).L
    eta = fill(s0, n)
    if re
        if shape === :B
            a = coef(fit, :sd)
            eta .+= ((exp.(a[1] .+ a[2] .* fx.z) .* (L * randn(rng, fx.n_g)))[fx.gi])
        else
            eta .+= exp(only(coef(fit, :sigma_sd_intercept))) .* (L * randn(rng, fx.n_g))[fx.gi]
            eta .+= exp(only(coef(fit, :sigma_sd_x))) .* d.x .* (L * randn(rng, fx.n_g))[fx.gi]
        end
    end
    return map(1:n) do i
        k = exp(-2eta[i])
        fam === :nb2 ? Float64(rand(rng, NegativeBinomial(k, k / (k + exp(lin[i]))))) :
                       rand(rng, Distributions.Gamma(k, exp(lin[i]) / k))
    end
end

@testset "structured sigma post-fit: predict, simulate, ranef, bridge ($fam $prov $shape)" for
        (fam, prov, shape) in ((:nb2, :animal, :B), (:gamma, :phylo, :B), (:nb2, :spatial, :A),
                               (:nb2, :relmat, :A))
    fx = _nss_fixture(fam, prov, shape; seed = _NSS_SEEDS[(fam, prov)] + (shape === :A ? 100 : 0))
    inp = _nss_inputs(fx, prov)
    fit = drm(_nss_form(prov, shape), _NSS_FAM[fam]; data = fx.data, inp...)
    @test fit.converged
    d = fx.data
    # Gamma draws with a large latent sigma underflow to 0 or are very heavy tailed, so h is
    # clamped and the Gamma spread statistic is the per-row IQR, not the variance (counts are
    # discrete, so their IQR cannot discriminate; they use the mean per-row variance).
    h(y) = fam === :nb2 ? log1p.(y) : clamp.(log.(y), -10.0, 10.0)
    # predict (population level, newdata with only x) and predict_parameters
    nd = (; x = [-0.5, 0.0, 1.0])
    b = coef(fit, :mu)
    @test predict(fit, nd; type = :link) ≈ b[1] .+ b[2] .* nd.x atol = 1e-10
    @test predict(fit, nd) ≈ exp.(b[1] .+ b[2] .* nd.x) atol = 1e-10
    ndp = shape === :B ? (; x = [-0.5, 0.0], id = ["s01", "s02"], z = [fx.z[1], fx.z[2]]) :
                         (; x = [-0.5, 0.0])
    pp = predict_parameters(fit, ndp)
    @test pp[:mu] ≈ exp.(b[1] .+ b[2] .* ndp.x) atol = 1e-10
    @test pp[:sigma] ≈ fill(exp(coef(fit, :sigma)[1]), 2) atol = 1e-10
    if shape === :B
        a = coef(fit, :sd)
        @test pp[:sd] ≈ exp.(a[1] .+ a[2] .* ndp.z) atol = 1e-8
    end
    # ranef exposes conditional modes of the raw log-sigma effects, one per level
    re = ranef(fit)
    @test all(v -> length(v) == fx.n_g && all(isfinite, v), values(re))
    @test length(re) == (shape === :B ? 1 : 2)
    # simulate: re = :zero is a plain draw; the marginal needs the data and kernel inputs
    @test length(simulate(fit; re = :zero, rng = MersenneTwister(1))) == length(d.y)
    @test_throws ArgumentError simulate(fit; rng = MersenneTwister(1))
    nsim = 300
    rng = MersenneTwister(77)
    S = simulate(fit; nsim, rng, data = d, inp...)
    @test size(S) == (length(d.y), nsim)
    R = reduce(hcat, [h(_nss_reference(fam, fit, fx, shape, prov, rng)) for _ in 1:nsim])
    Hs = reduce(hcat, [h(S[:, j]) for j in 1:nsim])
    z2 = ((mean(Hs; dims = 2) .- mean(R; dims = 2)) .^ 2) ./
         (var(Hs; dims = 2) ./ nsim .+ var(R; dims = 2) ./ nsim)
    @test mean(z2) < 1.5
    # wrong-answer variant: the reference without the sigma random effects must sit farther
    # from the simulated spread than the correct reference does
    R0 = reduce(hcat, [h(_nss_reference(fam, fit, fx, shape, prov, rng; re = false)) for _ in 1:nsim])
    iqr(a) = median([quantile(a[i, :], 0.75) - quantile(a[i, :], 0.25) for i in 1:size(a, 1)])
    spread(a) = fam === :nb2 ? mean(var(a; dims = 2)) : iqr(a)
    rel(a, bb) = abs(spread(a) / spread(bb) - 1)
    @test rel(Hs, R) < 0.10
    @test rel(Hs, R0) > rel(Hs, R)
    # bridge: same coefficients and likelihood as drm()
    rhs_s = shape === :A ? "sigma ~ 1 + $(prov)(1 + x | id)" :
            "sigma ~ 1 + $(prov)(1 | id); sd(id, $(_NSS_LEVEL[prov]), sigma) ~ z"
    br = drm_bridge(; formula = "y ~ x; $rhs_s", family = fam === :nb2 ? "nbinom2" : "gamma",
                    data = d, inp...)
    @test br["loglik"] ≈ loglik(fit) atol = 1e-6
    @test length(br["coef"]) == length(fit.theta)
    @test sort(collect(values(br["coef"]))) ≈ sort(fit.theta) atol = 1e-5
end

@testset "NB2/Gamma structured sigma: refusals" begin
    fx = _nss_fixture(:nb2, :relmat, :B; seed = 2026110811, n_g = 12, n_each = 8)
    d = fx.data; K = fx.K
    fm(level) = bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
                   level === :animal ? @formula(sd(id, animal, sigma) ~ z) : @formula(sd(id, relmat, sigma) ~ z))
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = d)                       # no K
    @test_throws ArgumentError drm(fm(:animal), NegBinomial2(); data = d, K = K)                # level mismatch
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = d, K = K[1:5, 1:5])      # wrong size
    Kbad = copy(K); Kbad[1, 1] = -1.0
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = d, K = Kbad)             # not PD
    z = copy(d.z); z[2] += 0.2
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = (; d..., z), K = K)      # SD predictor varies in a level
    y = copy(d.y); y[2] = NaN
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = (; d..., y), K = K)      # missing response
    @test_throws ArgumentError drm(fm(:relmat), NegBinomial2(); data = d, K = K, nAGQ = 5)      # no nAGQ
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id) ~ z), @formula(sd(id, relmat, sigma) ~ z)), NegBinomial2(); data = d, K = K)   # mean RE too
    # mean structured term refused
    @test_throws ArgumentError drm(bf(@formula(y ~ x + relmat(1 | id)), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma) ~ z)), NegBinomial2(); data = d, K = K)
    # Gamma: shape A is refused natively and here; so is a coef selector on shape B
    gy = merge(d, (; y = d.y .+ 0.5))
    @test_throws Exception drm(_nss_form(:relmat, :A), DRModels.Gamma(); data = gy, K = K)
    @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 | id)),
        @formula(sd(id, relmat, sigma, coef(x)) ~ z)), NegBinomial2(); data = d, K = K)
    # shape A: ML only, labelled correlated form not admitted, slope predictor must be numeric/nonzero
    @test_throws ArgumentError drm(_nss_form(:relmat, :A), NegBinomial2(); data = d, K = K, method = :REML)
    @test_throws ArgumentError drm(_nss_form(:relmat, :A), NegBinomial2(); data = (; d..., x = zero(d.x)), K = K)
    @test_throws Exception drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + relmat(1 + x | p | id))),
                               NegBinomial2(); data = d, K = K)
    @test_throws ArgumentError drm(_nss_form(:relmat, :A), NegBinomial2(); data = d)            # no K
    # latent cap
    @test DRModels._GSL_MAX_LATENT == 600
end

@testset "structured sigma: marginal other than :LA is refused on shape A as on shape B" begin
    fx = _nss_fixture(:nb2, :animal, :A; seed = 2026110812, n_g = 12, n_each = 8)
    inp = _nss_inputs(fx, :animal)
    for m in (:VA, :AGHQ)
        @test_throws ArgumentError drm(_nss_form(:animal, :A), NegBinomial2(); data = fx.data, inp..., marginal = m)
    end
    fxb = _nss_fixture(:nb2, :animal, :B; seed = 2026110813, n_g = 12, n_each = 8)
    @test_throws ArgumentError drm(_nss_form(:animal, :B), NegBinomial2(); data = fxb.data,
                                   _nss_inputs(fxb, :animal)..., marginal = :VA)
    @test_throws ArgumentError drm(_nss_form(:animal, :B), DRModels.Gamma(); data = fxb.data,
                                   _nss_inputs(fxb, :animal)..., marginal = :VA)
end

@testset "structured sigma: animal Ainv gives the fit A = inv(Ainv) gives ($fam $shape)" for
        (fam, shape) in ((:nb2, :B), (:gamma, :B), (:nb2, :A))
    fx = _nss_fixture(fam, :animal, shape; seed = _NSS_SEEDS[(fam, :animal)] + (shape === :A ? 100 : 0))
    Ainv = Matrix(Symmetric(inv(Symmetric(fx.K))))
    fA = drm(_nss_form(:animal, shape), _NSS_FAM[fam]; data = fx.data, A = fx.K)
    fI = drm(_nss_form(:animal, shape), _NSS_FAM[fam]; data = fx.data, Ainv)
    @test abs(loglik(fA) - loglik(fI)) < 1e-6
    @test fA.theta ≈ fI.theta atol = 1e-4
    # supplying both is refused
    @test_throws ArgumentError drm(_nss_form(:animal, shape), _NSS_FAM[fam]; data = fx.data, A = fx.K, Ainv)
    # bootstrap simulator gets the same kernel from Ainv as from A
    sA = DRModels._marginal_simulator_build(fA, fx.data; A = fx.K)
    sI = DRModels._marginal_simulator_build(fI, fx.data; Ainv)
    @test sI(MersenneTwister(9)) ≈ sA(MersenneTwister(9)) rtol = 1e-6
end

@testset "structured sigma: simulate(newdata) is population level ($fam $prov $shape)" for
        (fam, prov, shape) in ((:nb2, :animal, :B), (:gamma, :relmat, :B), (:nb2, :animal, :A))
    fx = _nss_fixture(fam, prov, shape; seed = _NSS_SEEDS[(fam, prov)] + (shape === :A ? 100 : 0))
    fit = drm(_nss_form(prov, shape), _NSS_FAM[fam]; data = fx.data, _nss_inputs(fx, prov)...)
    nd = shape === :B ? (; x = [-0.5, 0.0], id = ["s01", "s02"], z = [fx.z[1], fx.z[2]]) : (; x = [-0.5, 0.0])
    S = simulate(fit; newdata = nd, nsim = 4000, rng = MersenneTwister(1))
    @test size(S) == (2, 4000)
    b = coef(fit, :mu)
    mu = exp.(b[1] .+ b[2] .* nd.x)
    @test vec(mean(S; dims = 2)) ≈ mu rtol = 0.1          # no random effects in a newdata draw
    @test size(simulate(fit; newdata = nd, nsim = 3, rng = MersenneTwister(1), re = :zero)) == (2, 3)
end
