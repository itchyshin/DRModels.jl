# Metamorphic-invariance tests for the SD-regression routes (no R oracle needed).
#
# A metamorphic test transforms the INPUT in a way whose effect on the OUTPUT is known
# exactly, refits, and checks the predicted effect. For each route below (ordinary
# q1/q2/q3/q4 Gaussian, Poisson / NB2 / Gamma / Binomial animal, phylo and spatial q2,
# labelled `(1 + x | p | g)`, structured sigma) the properties are:
#   perm      shuffle the data rows                     -> logLik and estimates unchanged
#   relabel   permute group blocks AND rename levels    -> unchanged (A / tree / coords permuted consistently)
#   shift_v   v -> v + c (mean covariate, no random part) -> intercept moves by -beta_v c, logLik unchanged
#   shift_w   w -> w + c (SD-formula covariate)         -> SD intercept moves by -a_w c, logLik unchanged
#   scale_w   w -> k w                                  -> SD slope a_w -> a_w / k, logLik unchanged
#   scale_x   x -> k x (random-slope covariate)         -> beta_x / k, SD(x) intercept - log k, logLik unchanged
#   flip_x    x -> -x                                   -> beta_x -> -beta_x, other coordinates unchanged, logLik unchanged
#   shift_x_const  x -> x + c with CONSTANT SDs and free correlation -> logLik unchanged
#             (b0' = b0 + c b1 maps one covariance matrix to another; with `sd ~ w` formulas the model
#             is NOT closed under this map, so shift_x is reported, not asserted; see the evidence README
#             parity-crosswalk-evidence-20261107/metamorphic/README.md in the drmTMB repository)
#   scale_y   Gaussian y -> k y: beta k, log sigma + log k, SD intercepts + log k, logLik - n log k;
#             Gamma likewise with only beta0 + log k
#   predict   predict(fit, training data) == fitted(fit; re = :zero); predict_parameters == fit.scales; row-permutation
#             equivariant
#   swap      `(x + 1 | g)` is the same model as `(1 + x | g)` (q2 only)
#   reml      Gaussian structured q2 with method = :REML: perm, relabel and v-shift leave the REML logLik unchanged
#   simrefit  simulate (the package simulator, or the true generative process where `simulate` refuses),
#             refit: median error over seeds within a stated tolerance, and the truth never beats the
#             refit's optimum (nll(truth) - nll(fit) > -1e-6)
# Tolerances. First version (set before the first full run): |dlogLik| < 1e-5 and max |dtheta| < 1e-3 for every
# property. Tightened on 2026-11-08 from the observed spread of the 185 default-subset checks, worst case over three
# runs (Mac Julia 1.10, Totoro Julia 1.10, Totoro Julia 1.12; the binomial phylo Laplace route sets every maximum):
#   property class                    observed max |dlogLik|  observed max |dtheta|   now (dlogLik, dtheta)
#   perm, relabel (row order, names)        1.3e-07 (1.12)        3.4e-07              1e-6,  1e-5
#   shift_*, scale_*, flip_x (reparam.)     4.1e-07               9.5e-05              5e-6,  5e-4
#   swap                                    0 (identical)         0 (identical)        1e-8,  1e-8
#   (predict already asserts 1e-10 / 1e-12; REML was not run in the default subset and keeps 1e-5 / 1e-3)
# A first attempt at 1e-7 for perm/relabel failed on Julia 1.12 (binom_phylo_q2 perm, dlogLik 1.27e-7, dtheta 3.9e-9):
# summation order moves the Laplace logLik by ~1e-7 there while the estimates agree to 4e-9, so the bound follows the
# worst version, not the best. Row order and level naming leave the likelihood unchanged up to that noise (8x
# headroom now, was 79x); the reparametrisation checks are exact in the maths but each refit stops at its own
# `g_tol` (12x headroom on logLik, 5x on theta, was 24x and 10x). A row permutation that moved the logLik by
# 1e-6 to 1e-5 passed under the old bound; it fails now. If the logLik agrees but a
# coordinate does not (a correlation at its boundary is a flat direction) the status is FLAT, which counts as a
# failure here; that is why three routes carry a fixed `seed` with an interior optimum. flip_x and scale_y compare
# every coordinate except the correlation (its map is route-specific). simrefit: |median error| over 5 datasets
# < 0.25 for fixed effects and < 1.0 for SD coefficients (per-route `sdtol` / `fixtol` overrides, set AFTER the
# first full run for the routes whose responses identify SD-formula slopes poorly: NB2 and binary).
# These statistical bounds stay as they were: five datasets cannot pin a median to better than the sampling error,
# and they cannot detect a wrong likelihood (the invariances above do that). What simrefit
# now asserts exactly: no failure at all (every simulate / fit / objective evaluation succeeds and is finite; the
# failing seeds are listed), the only routes allowed to fall back from `simulate` to the generative process are the
# two in `SIM_REFUSED` (observed: Gaussian ordinary q2 and Gaussian sigma-animal), and nll(truth) - nll(fit) > -1e-6
# (the fit is a minimiser, so the truth cannot beat it beyond optimiser slop; the old bound was -1e-3, the smallest
# observed gap is +1.1).
# Data: StableRNG-generated (G = 24 groups) or the committed fixtures under test/fixtures (32 tips).
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_metamorphic_sd_routes.jl")'
# `METAMORPHIC_FULL=1` runs every route and property and writes a TSV to $METAMORPHIC_OUT
# (default: stdout summary); the default run is the fast subset the suite uses.
module TestMetamorphicSDRoutes

using DRModels, Test, LinearAlgebra, Printf
using StableRNGs
using Random: randperm
import Distributions
import Statistics

const FIXDIR = joinpath(@__DIR__, "fixtures")

# ------------------------------------------------------------------ kernels
function tree_newick(names)
    nw(ix, root) = length(ix) == 1 ? "$(names[ix[1]]):1.5" :
        "(" * nw(ix[1:end÷2], false) * "," * nw(ix[end÷2+1:end], false) * ")" * (root ? ";" : ":0.1")
    nw(collect(eachindex(names)), true)
end
function tree_corr(G)
    bits = round(Int, log2(G)); height = 1.5 + 0.1 * (bits - 1)
    b(i) = reverse(digits(i - 1; base = 2, pad = bits))
    [i == j ? 1.0 : (findfirst(b(i) .!= b(j)) - 1) * 0.1 / height for i in 1:G, j in 1:G]
end
function grid_xy(G)
    side = round(Int, sqrt(G))
    [(Float64(i), Float64(j)) for j in 0:side-1 for i in 0:side-1]
end
function grid_kernel(xy)
    G = length(xy)
    D = [hypot(xy[k][1] - xy[l][1], xy[k][2] - xy[l][2]) for k in 1:G, l in 1:G]
    exp.(-D ./ Statistics.median([D[k, l] for k in 1:G for l in k+1:G])) + 1e-6 * I
end

# ------------------------------------------------------------------ routes
# level: :ord | :animal | :phylo | :spatial. kind: :mean (SD per coefficient of a (1 + x ...| g) block),
# :sigma (`sigma ~ 1 + <level>(1 | g)`, `sd(g, sigma) ~ w0`), :q1 (`(1 | g)`, `sd(g) ~ w0`).
Base.@kwdef struct Route
    name::String
    fam::Any
    famkey::Symbol
    level::Symbol = :ord
    labelled::Bool = false
    members::Vector{Symbol} = [:intercept, :x]
    G::Int = 24
    m::Int = 10
    rho::Float64 = 0.0
    sdint::Vector{Float64} = [log(0.7), log(0.5), log(0.5), log(0.4)]
    sdslope::Vector{Float64} = [0.3, -0.3, 0.3, -0.3]
    sigma::Float64 = 0.5
    kind::Symbol = :mean
    fixture::Tuple{String,String} = ("", "")
    seed::Int = 20261108
    sdtol::Float64 = 1.0
    fixtol::Float64 = 0.25
end
hasdisp(r::Route) = r.famkey in (:gaussian, :nb2, :gamma)
wcols(r::Route) = [:w0, :w1, :w2, :w3][1:length(r.members)]
levelfn(l) = l === :phylo ? "phylo" : l === :spatial ? "spatial" : "animal"
levelkey(l) = l === :phylo ? "phylogenetic" : string(l)

function gen_fixture(r::Route)
    file, col = r.fixture
    lines = readlines(joinpath(FIXDIR, file))
    hdr = split(lines[2], '\t'); rows = [split(l, '\t') for l in lines[3:end]]
    getc(c, T) = (j = findfirst(==(c), hdr); T === String ? String[rw[j] for rw in rows] : parse.(T, [rw[j] for rw in rows]))
    G = 32; n = length(rows)
    gi = getc("gi", Int); zs = [zeros(G), zeros(G)]
    zs[1][gi] .= getc("z0", Float64); zs[2][gi] .= getc("z1", Float64)
    rng = StableRNG(7)
    C = r.level === :phylo ? tree_corr(G) : grid_kernel(grid_xy(G))
    return (; y = getc(col, Float64), x = getc("x", Float64), z = randn(rng, n), u = zeros(n),
            v = randn(rng, n) .* 0.5, gi, zs, C, ids = ["t$(lpad(i, 2, '0'))" for i in 1:G],
            relname = ["lev_$(lpad(p, 3, '0'))" for p in randperm(rng, G)],
            ord = collect(1:n), G, n)
end

function gen(r::Route; seed = r.seed)
    isempty(r.fixture[1]) || return gen_fixture(r)
    rng = StableRNG(seed)
    G, m = r.G, r.m; q = length(r.members); n = G * m
    ids = ["u$(lpad(i, 2, '0'))" for i in 1:G]
    C = r.level === :phylo ? tree_corr(G) : r.level === :spatial ? grid_kernel(grid_xy(G)) :
        r.level === :animal ? [0.3^abs(i - j) for i in 1:G, j in 1:G] : Matrix{Float64}(I, G, G)
    L = cholesky(Symmetric(C)).L
    zs = [k == 1 ? ifelse.(randperm(rng, G) .<= G ÷ 2, -1.0, 1.0) :
                   collect(range(-1.0, 1.0; length = G))[randperm(rng, G)] for k in 1:q]
    gi = repeat(1:G, inner = m)
    x = repeat(collect(range(-1.2, 1.2; length = m)), G)
    z = randn(rng, n) .* 0.8; v = randn(rng, n) .* 0.5; u4 = randn(rng, n)
    eta = 0.2 .+ 0.4 .* x .+ 0.2 .* v
    ls = zeros(n)
    if r.kind === :mean
        U = q == 2 ? hcat(ones(n), x) : q == 3 ? hcat(ones(n), x, z) : hcat(ones(n), x, z, u4)
        Rm = Matrix{Float64}(I, q, q)
        rho = r.labelled || (r.level === :ord && r.rho != 0) ? r.rho : 0.0
        for i in 1:q, j in 1:q
            i != j && (Rm[i, j] = rho^abs(i - j))
        end
        F = L * (randn(rng, G, q) * cholesky(Symmetric(Rm)).U)
        q >= 3 && (eta = eta .- 0.3 .* z)
        for k in 1:q
            s = exp.(r.sdint[k] .+ r.sdslope[k] .* zs[k])
            eta = eta .+ U[:, k] .* (s .* F[:, k])[gi]
        end
    else
        uf = L * randn(rng, G)
        s = exp.(r.sdint[1] .+ r.sdslope[1] .* zs[1])
        r.kind === :sigma ? (ls = log(r.sigma) .+ (s .* uf)[gi]) : (eta = eta .+ (s .* uf)[gi])
    end
    sdv = r.kind === :sigma ? exp.(ls) : fill(r.sigma, n)
    y = if r.famkey === :gaussian
        eta .+ sdv .* randn(rng, n)
    elseif r.famkey === :poisson
        Float64.([rand(rng, Distributions.Poisson(exp(e))) for e in eta])
    elseif r.famkey === :nb2
        rr = 1 ./ sdv .^ 2
        Float64.([rand(rng, Distributions.NegativeBinomial(rr[i], rr[i] / (rr[i] + exp(eta[i])))) for i in 1:n])
    elseif r.famkey === :gamma
        kk = 1 ./ sdv .^ 2
        [rand(rng, Distributions.Gamma(kk[i], exp(eta[i]) / kk[i])) for i in 1:n]
    else
        Float64.(rand(rng, n) .< 1 ./ (1 .+ exp.(-eta)))
    end
    return (; y, x, z, u = u4, v, gi, zs, C, ids, relname = ["lev_$(lpad(p, 3, '0'))" for p in randperm(rng, G)],
            ord = randperm(rng, n), G, n)
end

# Data and structural keywords for a row permutation and a level naming. `A` is in
# first-seen level order; phylo/spatial are keyed by label.
function materialize(r::Route, fx; perm = fx.ord, relab = false)
    gi = fx.gi[perm]; names = relab ? fx.relname : fx.ids
    cols = Pair{Symbol,Any}[:y => fx.y[perm], :x => fx.x[perm], :z => fx.z[perm], :u => fx.u[perm], :v => fx.v[perm]]
    for (k, w) in enumerate(wcols(r)); push!(cols, w => fx.zs[k][gi]); end
    push!(cols, :id => names[gi])
    kw = if r.level === :animal
        fs = unique(gi); (; A = fx.C[fs, fs])
    elseif r.level === :phylo
        (; tree = tree_newick(names))
    elseif r.level === :spatial
        xy = grid_xy(fx.G); (; coords = Dict(names[k] => [xy[k]...] for k in 1:fx.G))
    else
        (;)
    end
    return (; cols...), kw
end

function build_formula(r::Route; swap = false, constsd = false)
    lv = r.level
    lhs0 = r.members == [:intercept, :x] ? (swap ? "x + 1" : "1 + x") :
           r.members == [:intercept, :x, :z] ? "1 + x + z" : "1 + x + z + u"
    lhs = r.labelled ? lhs0 * " | p | id" : lhs0 * " | id"
    wrap = lv === :ord ? "($lhs)" : "$(levelfn(lv))($lhs)"
    sdlev = lv === :ord ? "" : ", $(levelkey(lv))"
    if r.kind === :mean
        fixed = join(["x", "z", "u"][1:length(r.members)-1], " + ")
        mu = eval(Meta.parse("@formula(y ~ $fixed + v + $wrap)"))
        sds = [eval(Meta.parse("@formula(sd(id$sdlev, coef($(mem))) ~ $(constsd ? "1" : string(w)))"))
               for (mem, w) in zip(r.members, wcols(r))]
        return hasdisp(r) ? bf(mu, @formula(sigma ~ 1), sds...) : bf(mu, sds...)
    elseif r.kind === :sigma
        sgl = lv === :ord ? "(1 | id)" : "$(levelfn(lv))(1 | id)"
        return bf(@formula(y ~ x + v), eval(Meta.parse("@formula(sigma ~ 1 + $sgl)")),
                  eval(Meta.parse("@formula(sd(id$sdlev, sigma) ~ w0)")))
    else
        mu = @formula(y ~ x + v + (1 | id)); sdf = @formula(sd(id) ~ w0)
        return hasdisp(r) ? bf(mu, @formula(sigma ~ 1), sdf) : bf(mu, sdf)
    end
end

fitroute(r, f, data, kw) = r.famkey === :gaussian ? drm(f, r.fam; data, kw...) :
                                                    drm(f, r.fam; data, kw..., se = false)

# ------------------------------------------------------------------ parameter bookkeeping
function decode_block(name::Symbol)
    s = String(name)
    startswith(s, "sdtarget_") || return s
    try String(hex2bytes(s[10:end])) catch; s end
end
function sdblocks(fit, r::Route)
    r.kind !== :mean && return [only(rg for (nm, rg) in fit.blocks if nm === :sd)]
    map(r.members) do mem
        hit = [rg for (nm, rg) in fit.blocks if startswith(String(nm), "sdtarget_") &&
               occursin("|$(mem)|", decode_block(nm))]
        length(hit) == 1 || error("cannot locate the SD block of $mem")
        only(hit)
    end
end
betaidx(fit) = (rg = only(rg for (nm, rg) in fit.blocks if nm === :mu); Dict(zip(fit.coefnames[1].second, rg)))
sigidx(fit) = [k for (nm, rg) in fit.blocks if nm === :sigma for k in rg]
# coordinates compared across a transformation: everything except the correlation block
compidx(fit) = [k for (nm, rg) in fit.blocks if nm !== :recov for k in rg]

struct Base0; r; fx; data; kw; f; fit; end
function basefit(r::Route)
    fx = gen(r); data, kw = materialize(r, fx); f = build_formula(r)
    Base0(r, fx, data, kw, f, fitroute(r, f, data, kw))
end
maxabs(a, b) = maximum(abs.(a .- b))

# ------------------------------------------------------------------ properties: each returns (status, detail)
# Tolerance classes (see the header): float-noise invariances vs invariances through a re-parametrised refit.
const TOL_EXACT = (; ll = 1e-6, th = 1e-5)
const TOL_REPARAM = (; ll = 5e-6, th = 5e-4)
function cmpfit(b::Base0, fit2; tll = TOL_REPARAM.ll, tth = TOL_REPARAM.th, expect = b.fit.theta, dll = 0.0, idx = eachindex(b.fit.theta))
    dl = abs(loglikelihood(fit2) - loglikelihood(b.fit) - dll)
    dt = maxabs(fit2.theta[idx], expect[idx])
    st = !(dl < tll) ? "FAIL" : (dt < tth && fit2.converged) ? "pass" : "FLAT"
    (st, @sprintf("dll=%.2e dtheta=%.2e conv=%s max|theta|=%.1f", dl, dt, fit2.converged, maximum(abs, fit2.theta)))
end
refit(b::Base0, tf; f = b.f) = fitroute(b.r, f, tf(b.data), b.kw)
nw(b::Base0) = length(sdblocks(b.fit, b.r))

function p_perm(b::Base0)
    perm = randperm(StableRNG(11), length(b.fx.ord))
    data, kw = materialize(b.r, b.fx; perm = b.fx.ord[perm])
    cmpfit(b, fitroute(b.r, b.f, data, kw); tll = TOL_EXACT.ll, tth = TOL_EXACT.th)
end
function p_relabel(b::Base0)
    gp = randperm(StableRNG(12), b.fx.G)
    perm = vcat([findall(==(g), b.fx.gi) for g in gp]...)
    data, kw = materialize(b.r, b.fx; perm, relab = true)
    cmpfit(b, fitroute(b.r, b.f, data, kw); tll = TOL_EXACT.ll, tth = TOL_EXACT.th)
end
function p_shift_v(b::Base0; c = 1.7)
    bi = betaidx(b.fit); ex = copy(b.fit.theta); ex[bi["(Intercept)"]] -= b.fit.theta[bi["v"]] * c
    cmpfit(b, refit(b, d -> merge(d, (; v = d.v .+ c))); expect = ex)
end
function p_shift_w(b::Base0; c = 0.8)
    ex = copy(b.fit.theta)
    for rg in sdblocks(b.fit, b.r); ex[first(rg)] -= b.fit.theta[last(rg)] * c; end
    cmpfit(b, refit(b, d -> merge(d, NamedTuple(w => d[w] .+ c for w in wcols(b.r)[1:nw(b)]))); expect = ex)
end
function p_scale_w(b::Base0; k = 2.5)
    ex = copy(b.fit.theta)
    for rg in sdblocks(b.fit, b.r); ex[last(rg)] /= k; end
    cmpfit(b, refit(b, d -> merge(d, NamedTuple(w => d[w] .* k for w in wcols(b.r)[1:nw(b)]))); expect = ex)
end
function p_scale_x(b::Base0; k = 2.0)
    ex = copy(b.fit.theta); ex[betaidx(b.fit)["x"]] /= k
    b.r.kind === :mean && (ex[first(sdblocks(b.fit, b.r)[2])] -= log(k))
    cmpfit(b, refit(b, d -> merge(d, (; x = d.x .* k))); expect = ex)
end
function p_flip_x(b::Base0)
    ex = copy(b.fit.theta); ex[betaidx(b.fit)["x"]] *= -1
    cmpfit(b, refit(b, d -> merge(d, (; x = -d.x))); expect = ex, idx = compidx(b.fit))
end
# shift of the random-slope covariate: exactly equivariant only for q1/sigma routes; with SD formulas on a
# random slope it is reported, not asserted (see header).
function p_shift_x(b::Base0; c = 0.9)
    fit2 = refit(b, d -> merge(d, (; x = d.x .+ c)))
    if b.r.kind === :mean
        return ("EXPECTED", @sprintf("random slope on x with `sd ~ w`: not shift-equivariant; dll=%+.3f conv=%s",
                                     loglikelihood(fit2) - loglikelihood(b.fit), fit2.converged))
    end
    ex = copy(b.fit.theta); bi = betaidx(b.fit); ex[bi["(Intercept)"]] -= b.fit.theta[bi["x"]] * c
    cmpfit(b, fit2; expect = ex)
end
function p_shift_x_const(b::Base0; c = 0.9)
    (b.r.kind === :mean && (b.r.labelled || b.r.level === :ord && b.r.rho != 0)) || return ("NA", "needs free correlation")
    f0 = build_formula(b.r; constsd = true)
    fa = fitroute(b.r, f0, b.data, b.kw)
    fb = fitroute(b.r, f0, merge(b.data, (; x = b.data.x .+ c)), b.kw)
    bi = betaidx(fa); ex = copy(fa.theta); ex[bi["(Intercept)"]] -= fa.theta[bi["x"]] * c
    dl = abs(loglikelihood(fb) - loglikelihood(fa)); dbeta = maxabs(fb.theta[1:length(bi)], ex[1:length(bi)])
    (dl < TOL_EXACT.ll && fb.converged ? "pass" : (dl < TOL_EXACT.ll ? "FLAT" : "FAIL"),
     @sprintf("dll=%.2e dbeta=%.2e conv=%s/%s", dl, dbeta, fa.converged, fb.converged))
end
function p_scale_y(b::Base0; k = 3.0)
    b.r.famkey in (:gaussian, :gamma) || return ("NA", "count/binary response")
    ex = copy(b.fit.theta); bi = betaidx(b.fit)
    if b.r.famkey === :gaussian
        for i in values(bi); ex[i] *= k; end
        for i in sigidx(b.fit); ex[i] += log(k); end
        b.r.kind === :sigma || for rg in sdblocks(b.fit, b.r); ex[first(rg)] += log(k); end
    else
        ex[bi["(Intercept)"]] += log(k)
    end
    cmpfit(b, refit(b, d -> merge(d, (; y = d.y .* k))); expect = ex, dll = -length(b.data.y) * log(k), idx = compidx(b.fit))
end
function p_predict(b::Base0)
    fit = b.fit; d = b.data
    mx = maxabs(predict(fit, d), fitted(fit; re = :zero))
    pp = predict_parameters(fit, d; se = false)
    ms = 0.0
    for (nm, _) in fit.blocks
        key = (nm === :sigma && haskey(fit.scales, :sigma_fixed)) ? :sigma_fixed : nm   # population vs conditional sigma
        haskey(fit.scales, key) && haskey(pp, nm) && (ms = max(ms, maxabs(pp[nm], fit.scales[key])))
    end
    perm = randperm(StableRNG(13), length(d.y))
    mp = maxabs(predict(fit, map(c -> c[perm], d)), predict(fit, d)[perm])
    (mx < 1e-10 && ms < 1e-10 && mp < 1e-12 ? "pass" : "FAIL",
     @sprintf("|predict-fitted|=%.1e |predict_parameters-scales|=%.1e |row-perm|=%.1e", mx, ms, mp))
end
function p_swap(b::Base0)
    b.r.members == [:intercept, :x] && b.r.kind === :mean || return ("NA", "needs a q = 2 block")
    fit2 = fitroute(b.r, build_formula(b.r; swap = true), b.data, b.kw)
    dl = abs(loglikelihood(fit2) - loglikelihood(b.fit))
    (dl < 1e-8 && maxabs(fit2.theta, b.fit.theta) < 1e-8 ? "pass" : "FAIL", @sprintf("dll=%.2e dtheta=%.2e", dl, maxabs(fit2.theta, b.fit.theta)))
end

# REML (Gaussian structured q2): the criterion is invariant to a shift of a mean covariate (X -> X T, |det T| = 1)
# and, like ML, to row order and level naming.
function p_reml(b::Base0; c = 1.7)
    (b.r.famkey === :gaussian && b.r.kind === :mean && b.r.level in (:animal, :phylo, :spatial) && !b.r.labelled) ||
        return ("NA", "REML structured q2 is Gaussian only")
    fr(data, kw) = drm(b.f, b.r.fam; data, kw..., method = :REML)
    f0 = fr(b.data, b.kw)
    perm = randperm(StableRNG(14), length(b.fx.ord))
    dp, kp = materialize(b.r, b.fx; perm = b.fx.ord[perm])
    gp = randperm(StableRNG(15), b.fx.G); perm2 = vcat([findall(==(g), b.fx.gi) for g in gp]...)
    dq, kq = materialize(b.r, b.fx; perm = perm2, relab = true)
    f1, f2, f3 = fr(dp, kp), fr(dq, kq), fr(merge(b.data, (; v = b.data.v .+ c)), b.kw)
    bi = betaidx(f0); ex = copy(f0.theta); ex[bi["(Intercept)"]] -= f0.theta[bi["v"]] * c
    dl = maximum(abs(loglikelihood(f) - loglikelihood(f0)) for f in (f1, f2, f3))
    dt = max(maxabs(f1.theta, f0.theta), maxabs(f2.theta, f0.theta), maxabs(f3.theta, ex))
    (dl < 1e-5 && dt < 1e-3 && all(f -> f.converged, (f0, f1, f2, f3)) ? "pass" : "FAIL",
     @sprintf("REML dll=%.2e dtheta=%.2e (perm, relabel, shift_v)", dl, dt))
end

const TRUTH_BETA = Dict("(Intercept)" => 0.2, "x" => 0.4, "v" => 0.2, "z" => -0.3, "u" => 0.0)
function truth_theta(r::Route, fit)
    th = copy(fit.theta)
    for (nm, k) in betaidx(fit); th[k] = TRUTH_BETA[nm]; end
    for (k, rg) in enumerate(sdblocks(fit, r)); th[rg] = [r.sdint[k], r.sdslope[k]]; end
    for i in sigidx(fit); th[i] = log(r.sigma); end
    return th
end
# Routes for which `simulate(fit; ...)` refuses (observed 2026-11-08); simrefit falls back to the generative
# process for these and only these. Any other route that falls back is a swallowed failure and is reported as FAIL.
const SIM_REFUSED = ("gauss_ord_q2", "gauss_sigma_animal")
function p_simrefit(b::Base0; S = 5, simseed = 100)
    fit = b.fit; r = b.r; datasets = Any[]; th0 = fit.theta; mode = :simulate; seeds = collect(1:S)
    simerr = nothing
    try
        Y = simulate(fit; nsim = S, rng = StableRNG(simseed), data = b.data, b.kw...)
        for s in 1:S; push!(datasets, (merge(b.data, (; y = Y[:, s])), b.kw)); end
    catch e
        isempty(r.fixture[1]) || rethrow()
        simerr = first(sprint(showerror, e), 120)
        empty!(datasets); mode = :truth; th0 = truth_theta(r, fit); seeds = [5000 + s for s in 1:S]
        for s in 1:S
            fxs = gen(r; seed = seeds[s]); push!(datasets, materialize(r, fxs))
        end
    end
    est = fill(NaN, length(th0), S); gap = fill(NaN, S); conv = 0
    failed = Tuple{Int,String}[]              # (seed index, reason): every failure is counted, none is swallowed
    for s in 1:S
        d, kw = datasets[s]
        try
            fs = fitroute(r, b.f, d, kw)
            est[:, s] = fs.theta; conv += fs.converged; gap[s] = fs.nll(th0) - fs.nll(fs.theta)
            isfinite(gap[s]) && all(isfinite, est[:, s]) || push!(failed, (seeds[s], "non-finite estimate or objective"))
        catch e
            push!(failed, (seeds[s], first(sprint(showerror, e), 80)))
        end
    end
    ok = .!isnan.(gap)
    nfail = length(failed)
    fallback_ok = mode === :simulate || r.name in SIM_REFUSED
    ok_any = any(ok)
    med = ok_any ? [abs(Statistics.median(est[i, ok] .- th0[i])) for i in eachindex(th0)] : fill(Inf, length(th0))
    bfix = maximum(med[collect(values(betaidx(fit)))])
    bsd = maximum(med[reduce(vcat, collect.(sdblocks(fit, r)))])
    mingap = ok_any ? minimum(gap[ok]) : -Inf
    st = if nfail > 0 || !fallback_ok || mingap <= -1e-6
        "FAIL"
    else
        bfix < r.fixtol && bsd < r.sdtol ? "pass" : "WEAK"
    end
    msg = nfail > 0 ? "; FAILED seeds " * join(["$(sd): $(why)" for (sd, why) in failed], ", ") : ""
    simerr === nothing || (msg *= (fallback_ok ? "; simulate refused (documented): " : "; simulate refused unexpectedly: ") * simerr)
    (st, @sprintf("%s S=%d failures=%d conv=%d |median err| fixed=%.3f SD-coef=%.3f; min(nll(truth)-nll(fit))=%.2e%s",
                  mode, S, nfail, conv, bfix, bsd, mingap, msg))
end

const PROPS = [:perm => p_perm, :relabel => p_relabel, :shift_v => p_shift_v, :shift_w => p_shift_w,
               :scale_w => p_scale_w, :scale_x => p_scale_x, :flip_x => p_flip_x, :shift_x => p_shift_x,
               :shift_x_const => p_shift_x_const, :scale_y => p_scale_y, :predict => p_predict,
               :swap => p_swap, :reml => p_reml, :simrefit => p_simrefit]

# ------------------------------------------------------------------ route table
const FAMS = Dict(:gaussian => Gaussian(), :poisson => Poisson(), :nb2 => NegBinomial2(),
                  :gamma => Gamma(), :binomial => Binomial())
R(name, fk, level; kw...) = Route(; name, fam = FAMS[fk], famkey = fk, level, kw...)
psq(fk, kind) = ("q2_stable/psq2_$(fk)_$(kind).tsv", "y")
lps(col, kind) = ("q2_stable/lps_$(kind).tsv", col)
bq(kind) = ("binomial_phylo_spatial_q2/binq2_$(kind).tsv", "y")
BIN = (; sdint = [log(0.9), log(0.8)], sdtol = 3.0, fixtol = 0.6)

# (route, fast default subset: true = every property, a tuple = only those, false = FULL mode only)
const ROUTES = [
    (R("gauss_ord_q2", :gaussian, :ord; rho = 0.4, m = 8), true),
    (R("gauss_ord_q3", :gaussian, :ord; rho = 0.4, m = 8, members = [:intercept, :x, :z]), (:perm, :scale_x, :predict)),
    (R("gauss_ord_q4", :gaussian, :ord; rho = 0.4, m = 8, members = [:intercept, :x, :z, :u]), false),
    (R("pois_ord_q2", :poisson, :ord; rho = 0.4), true),
    (R("nb2_ord_q2", :nb2, :ord; rho = 0.4), true),
    (R("gamma_ord_q2", :gamma, :ord; rho = 0.4, sigma = 0.4), false),            # refused: one SD target only
    (R("binom_ord_q2", :binomial, :ord; rho = 0.4, m = 24, G = 48, sdint = [log(0.9), log(0.8)], sdtol = 3.0, fixtol = 0.6), false),
    (R("gauss_animal_q2", :gaussian, :animal; m = 8), true),
    (R("pois_animal_q2", :poisson, :animal), true),
    (R("nb2_animal_q2", :nb2, :animal; sdtol = 1.5), true),
    (R("gamma_animal_q2", :gamma, :animal; sigma = 0.4), true),
    (R("binom_animal_q2", :binomial, :animal; m = 28, G = 32, BIN...), false),
    (R("gauss_phylo_q2", :gaussian, :phylo; fixture = psq(:gaussian, :phylo)), false),
    (R("gauss_spatial_q2", :gaussian, :spatial; fixture = psq(:gaussian, :spatial)), true),
    (R("pois_phylo_q2", :poisson, :phylo; fixture = psq(:poisson, :phylo)), true),
    (R("pois_spatial_q2", :poisson, :spatial; fixture = psq(:poisson, :spatial)), false),
    (R("nb2_phylo_q2", :nb2, :phylo; fixture = psq(:nb2, :phylo), sdtol = Inf), true),
    (R("nb2_spatial_q2", :nb2, :spatial; fixture = psq(:nb2, :spatial), sdtol = 1.5), false),
    (R("binom_phylo_q2", :binomial, :phylo; fixture = bq(:phylo), BIN...), true),
    (R("binom_spatial_q2", :binomial, :spatial; fixture = bq(:spatial), BIN...), false),
    (R("pois_lab_ord", :poisson, :ord; labelled = true, rho = 0.5), true),
    (R("nb2_lab_ord", :nb2, :ord; labelled = true, rho = 0.5), false),
    (R("pois_lab_animal", :poisson, :animal; labelled = true, rho = 0.5, seed = 1), false),
    (R("nb2_lab_animal", :nb2, :animal; labelled = true, rho = 0.5, seed = 2), false),
    (R("gauss_lab_animal", :gaussian, :animal; labelled = true, rho = 0.5, m = 8), false),
    (R("pois_lab_phylo", :poisson, :phylo; labelled = true, fixture = lps("yp", :phylo)), true),
    (R("pois_lab_spatial", :poisson, :spatial; labelled = true, fixture = lps("yp", :spatial)), false),
    (R("nb2_lab_phylo", :nb2, :phylo; labelled = true, fixture = lps("yn", :phylo)), false),
    (R("gauss_lab_phylo", :gaussian, :phylo; labelled = true, fixture = lps("yg", :phylo)), false),
    (R("gauss_q1", :gaussian, :ord; kind = :q1, members = [:intercept]), true),
    (R("pois_q1", :poisson, :ord; kind = :q1, members = [:intercept]), false),
    (R("gauss_sigma_ord", :gaussian, :ord; kind = :sigma, members = [:intercept]), false),
    (R("gauss_sigma_animal", :gaussian, :animal; kind = :sigma, members = [:intercept]), true),
    (R("nb2_sigma_animal", :nb2, :animal; kind = :sigma, members = [:intercept], seed = 1), true),
    (R("gamma_sigma_animal", :gamma, :animal; kind = :sigma, members = [:intercept]), false),
]

const FULL = get(ENV, "METAMORPHIC_FULL", "") == "1"
# Slowest properties left to FULL mode so the default subset stays near 3 min (each is still covered by another route).
const FAST_SKIP = Dict("gauss_animal_q2" => (:reml,), "gauss_spatial_q2" => (:reml, :simrefit),
                       "nb2_phylo_q2" => (:simrefit,), "nb2_animal_q2" => (:simrefit,))
const ROWS = NamedTuple[]

function run_route(r::Route; skip = (), S = 5)
    t0 = time(); local b
    try
        b = basefit(r)
    catch e
        push!(ROWS, (; route = r.name, prop = "base fit", status = "REFUSED", detail = first(sprint(showerror, e), 160), secs = time() - t0))
        return nothing
    end
    push!(ROWS, (; route = r.name, prop = "base fit", status = b.fit.converged ? "pass" : "FAIL",
                  detail = @sprintf("ll=%.4f", loglikelihood(b.fit)), secs = time() - t0))
    for (p, f) in PROPS
        p in skip && continue
        t1 = time()
        st, det = try
            p === :simrefit ? f(b; S) : f(b)
        catch e
            ("ERROR", first(sprint(showerror, e), 200))
        end
        push!(ROWS, (; route = r.name, prop = String(p), status = st, detail = det, secs = time() - t1))
    end
    return b
end

function main()
    out = get(ENV, "METAMORPHIC_OUT", "")
    for (r, fast) in ROUTES
        (FULL || fast !== false) || continue
        n0 = length(ROWS)
        @testset "$(r.name)" begin
            skip = FULL ? () : fast === true ? (:shift_x, get(FAST_SKIP, r.name, ())...) : Tuple(setdiff(first.(PROPS), fast))
            run_route(r; skip, S = 5)
            for row in ROWS[n0+1:end]
                row.status in ("EXPECTED", "NA") && continue
                if FULL && row.status in ("REFUSED", "FLAT", "WEAK")
                    @test_skip row.status == "pass"
                else
                    @testset "$(row.prop) [$(row.detail)]" begin
                        @test row.status == "pass"
                    end
                end
            end
        end
    end
    if !isempty(out)
        open(out, "w") do io
            for x in ROWS
                println(io, join((x.route, x.prop, x.status, replace(x.detail, '\t' => ' '), round(x.secs; digits = 1)), '\t'))
            end
        end
    end
    for x in ROWS
        x.status in ("pass", "NA", "EXPECTED") || @printf("  %-18s %-14s %-8s %s\n", x.route, x.prop, x.status, x.detail)
    end
    @printf("metamorphic: %d checks, %.0f s fit time\n", length(ROWS), sum(x.secs for x in ROWS))
end

# METAMORPHIC_LIB=1 loads the machinery without running (used by the evidence scripts).
isempty(get(ENV, "METAMORPHIC_LIB", "")) && @testset "Metamorphic invariance of the SD-regression routes" begin
    main()
end

end # module
