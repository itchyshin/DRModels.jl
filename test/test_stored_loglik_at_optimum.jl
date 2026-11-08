# The stored log-likelihood of a fit must be the objective at the converged
# parameters: |fit.loglik - (-fit.nll(theta_hat))| < 1e-10.
#
# Several Laplace fitters kept the objective in a variable that the `nll` closure
# also assigns (Julia boxes it), so a finite-difference gradient or Hessian call
# placed after `value = nll(theta_hat)` silently replaced the stored value with
# the objective at a stencil point near theta_hat (typically 1e-6 to 1e-5 away,
# once 28). This file guards every route that was measured to have that defect.
using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions
using SpecialFunctions: loggamma

const _SLL_TOL = 1e-10

_sll_gap(fit) = fit.loglik - (-fit.nll(fit.theta))

function _sll_check(label, fit)
    gap = _sll_gap(fit)
    @info "stored-loglik gap" label gap
    @test abs(gap) < _SLL_TOL
    # `fit.nll` returns the stored value at exactly theta_hat, so also evaluate it
    # one ulp away (an unpinned evaluation). The defect this file guards moved the
    # stored value by 1e-6 to 1e-4; warm-start noise in the inner mode solve is
    # below 1e-8, so 1e-7 separates the two without loosening the check above.
    near = -fit.nll(nextfloat.(fit.theta))
    @info "stored loglik vs unpinned nll one ulp from theta_hat" label gap_unpinned = fit.loglik - near
    # The unpinned objective carries the inner mode solve's acceptance noise
    # (step norm <= 1e-7 relative), a deterministic ~1e-8 to 1e-7 offset that is
    # independent of g_tol (Binomial animal q2 on Julia 1.10: 1.2e-7 at g_tol
    # 1e-5 through 1e-12, and 1.5e-8 on 1.12 where the fit converges to a
    # loglik 19 higher). A fit that did not converge sits where that noise
    # dominates, so it gets 5e-7; still far below the 1e-6 to 6e-4 the
    # shared-variable defect produced. The pinned check above is unchanged.
    @test abs(fit.loglik - near) < (fit.converged ? 1e-7 : 5e-7)
    return gap
end

# ---------------------------------------------------------------- NB2 structured
function _sll_nb2_fixture(; G = 20, m = 12, alpha = [-0.65, 0.4], seed = 20261102,
                          slope = false)
    rng = MersenneTwister(seed)
    g = repeat(1:G, inner = m)
    zgroup = repeat([-1.0, 1.0], inner = G ÷ 2)
    x = randn(rng, length(g))
    R = 0.35 .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    latent = exp.(alpha[1] .+ alpha[2] .* zgroup) .* (cholesky(Symmetric(R)).L * randn(rng, G))
    load = slope ? x : ones(length(x))
    mu = exp.(0.35 .+ 0.25 .* x .+ load .* latent[g])
    sz = 1 / 0.42^2
    y = Float64[rand(rng, Distributions.NegativeBinomial(sz, sz / (sz + v))) for v in mu]
    coords = hcat(mod.(0:(G - 1), 5), div.(0:(G - 1), 5)) .* 1.3
    return (; data = (; y, x, z = zgroup[g], g = ["id_$(i)" for i in g]),
            R, zgroup, gidx = g, coords, load)
end

# Independent dense Laplace for NB2 with analytic derivatives: no Optim, no
# ForwardDiff and none of the fitter's code.
function _sll_dense_nb2_laplace(theta, y, Xmu, gidx, load, zgroup, R)
    Q = inv(Symmetric(R))
    pmu = size(Xmu, 2)
    r = exp(-2 * theta[pmu + 1])
    sdl = hcat(ones(length(zgroup)), zgroup) * theta[(pmu + 2):end]
    eta0 = Xmu * theta[1:pmu]
    scale = exp.(sdl)[gidx] .* load
    function joint(b)
        eta = eta0 .+ scale .* b[gidx]
        s = 0.5 * dot(b, Q * b)
        for i in eachindex(y)
            s -= loggamma(y[i] + r) - loggamma(r) - loggamma(y[i] + 1) +
                 r * log(r) + y[i] * eta[i] - (r + y[i]) * log(r + exp(eta[i]))
        end
        s
    end
    b = zeros(size(R, 1))
    H = Q
    for _ in 1:200
        eta = eta0 .+ scale .* b[gidx]
        mu = exp.(eta)
        d1 = y .- (r .+ y) .* mu ./ (r .+ mu)
        w = (r .+ y) .* r .* mu ./ (r .+ mu) .^ 2
        grad = Q * b
        H = Matrix(Q)
        for i in eachindex(y)
            grad[gidx[i]] -= scale[i] * d1[i]
            H[gidx[i], gidx[i]] += scale[i]^2 * w[i]
        end
        step = Symmetric(H) \ grad
        f0 = joint(b)
        t = 1.0
        while t > 1e-8 && joint(b .- t .* step) > f0
            t /= 2
        end
        b .-= t .* step
        norm(step) < 1e-13 * (1 + norm(b)) && break
    end
    eta = eta0 .+ scale .* b[gidx]
    mu = exp.(eta)
    w = (r .+ y) .* r .* mu ./ (r .+ mu) .^ 2
    H = Matrix(Q)
    for i in eachindex(y)
        H[gidx[i], gidx[i]] += scale[i]^2 * w[i]
    end
    return joint(b) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Q))
end

@testset "stored loglik is the objective at theta_hat" begin
@testset "NB2 structured SD regression" begin
    fx = _sll_nb2_fixture()
    fs = _sll_nb2_fixture(slope = true, seed = 20261103)
    sd_rel = @formula(sd(g, relmat) ~ z)
    sd_ani = @formula(sd(g, animal) ~ z)
    sd_spa = @formula(sd(g, spatial) ~ z)

    fit = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1), sd_rel),
              NegBinomial2(); data = fx.data, K = fx.R, se = false)
    _sll_check("relmat intercept", fit)
    # Independent dense Laplace at theta_hat agrees with the fitter's nll there.
    Xmu = hcat(ones(length(fx.data.y)), fx.data.x)
    dense = _sll_dense_nb2_laplace(fit.theta, fx.data.y, Xmu, fx.gidx, fx.load,
                                    fx.zgroup, fx.R)
    @info "dense Laplace vs fitter nll at theta_hat" dense nll = fit.nll(fit.theta) diff = dense - fit.nll(fit.theta)
    @test dense ≈ fit.nll(fit.theta) atol = 2e-3
    @test abs(fit.loglik - (-dense)) < 2e-3

    fit = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sigma ~ 1), sd_rel),
              NegBinomial2(); data = fx.data, K = fx.R, se = true)
    _sll_check("relmat intercept, se = true", fit)

    fit = drm(bf(@formula(y ~ x + relmat(0 + x | g)), @formula(sigma ~ 1), sd_rel),
              NegBinomial2(); data = fs.data, K = fs.R, se = false)
    _sll_check("relmat slope", fit)

    fit = drm(bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1), sd_ani),
              NegBinomial2(); data = fx.data, A = fx.R, se = false)
    _sll_check("animal intercept (A)", fit)

    fit = drm(bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1), sd_ani),
              NegBinomial2(); data = fx.data, Ainv = inv(fx.R), se = false)
    _sll_check("animal intercept (Ainv)", fit)

    fit = drm(bf(@formula(y ~ x + animal(0 + x | g)), @formula(sigma ~ 1), sd_ani),
              NegBinomial2(); data = fs.data, A = fs.R, se = false)
    _sll_check("animal slope", fit)

    fit = drm(bf(@formula(y ~ x + spatial(0 + x | g)), @formula(sigma ~ 1), sd_spa),
              NegBinomial2(); data = fs.data, coords = fs.coords, se = false)
    _sll_check("spatial slope", fit)
end

# --------------------------------------------------- count and binomial routes
_sll_bern(rng, eta) = Float64[rand(rng, Distributions.Bernoulli(1 / (1 + exp(-v)))) for v in eta]

function _sll_animal_fixture(; G = 24, m = 12, seed = 20261104, rho = 0.4)
    rng = MersenneTwister(seed)
    ids = ["animal_$(i)" for i in randperm(rng, G)]
    R = rho .^ abs.(reshape(1:G, :, 1) .- reshape(1:G, 1, :))
    scale = collect(range(0.7, 1.5; length = G))
    A = Diagonal(scale) * R * Diagonal(scale)
    gidx = repeat(1:G, inner = m)
    z0 = repeat([-1.0, 1.0], inner = G ÷ 2)[randperm(rng, G)]
    z1 = collect(range(-1.0, 1.0; length = G))[randperm(rng, G)]
    x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
    L = cholesky(Symmetric(A)).L
    return (; rng, G, m, ids, A, gidx, z0, z1, x, L, group = repeat(ids, inner = m))
end

@testset "Other Laplace SD routes" begin
    # Poisson structured (animal) SD regression
    f = _sll_animal_fixture()
    eta = 0.25 .+ 0.2 .* f.x .+ (exp.(-0.55 .+ 0.38 .* f.z0) .* (f.L * randn(f.rng, f.G)))[f.gidx]
    y = Float64[rand(f.rng, Distributions.Poisson(exp(v))) for v in eta]
    data = (; y, x = f.x, z = f.z0[f.gidx], g = f.group)
    fit = drm(bf(@formula(y ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z)),
              Poisson(); data, A = f.A, se = true)
    _sll_check("Poisson animal SD", fit)

    # Gamma structured (animal) SD regression and the independent
    # intercept-and-slope SD regression
    f = _sll_animal_fixture(seed = 20261105)
    eta = 0.2 .+ 0.18 .* f.x .+ (exp.(-0.65 .+ 0.4 .* f.z0) .* (f.L * randn(f.rng, f.G)))[f.gidx]
    shape = 1 / 0.32^2
    y = Float64[rand(f.rng, Distributions.Gamma(shape, exp(v) / shape)) for v in eta]
    data = (; y, x = f.x, z = f.z0[f.gidx], g = f.group)
    fit = drm(bf(@formula(y ~ x + animal(1 | g)), @formula(sigma ~ 1),
                 @formula(sd(g, animal) ~ z)), Gamma(); data, A = f.A, se = true)
    _sll_check("Gamma animal SD", fit)

    u0, u1 = f.L * randn(f.rng, f.G), f.L * randn(f.rng, f.G)
    eta = 0.2 .+ 0.3 .* f.x .+ (exp.(-0.65 .+ 0.35 .* f.z0) .* u0)[f.gidx] .+
          f.x .* (exp.(-1.05 .+ 0.45 .* f.z1) .* u1)[f.gidx]
    y = Float64[rand(f.rng, Distributions.Gamma(1 / 0.38^2, exp(v) * 0.38^2)) for v in eta]
    data = (; y, x = f.x, z0 = f.z0[f.gidx], z1 = f.z1[f.gidx], g = f.group)
    fit = drm(bf(@formula(y ~ x + animal(1 + x | g)), @formula(sigma ~ 1),
                 @formula(sd(g, animal, coef(intercept)) ~ z0),
                 @formula(sd(g, animal, coef(x)) ~ z1)), Gamma(); data, A = f.A)
    _sll_check("Gamma animal intercept+slope SD", fit)

    # Binomial animal SD (related-matrix route, covariance A)
    f = _sll_animal_fixture(seed = 20261106)
    eta = -0.25 .+ 0.2 .* f.x .+ (exp.(-0.9 .+ 0.35 .* f.z1) .* (f.L * randn(f.rng, f.G)))[f.gidx]
    data = (; y = _sll_bern(f.rng, eta), x = f.x, z = f.z1[f.gidx], g = f.group)
    fit = drm(bf(@formula(y ~ x + animal(1 | g)), @formula(sd(g, animal) ~ z)),
              DRModels.Binomial(); data, A = f.A, se = true)
    _sll_check("Binomial animal SD (related)", fit)

    # Binomial animal q2 (intercept and slope, correlated)
    f = _sll_animal_fixture(G = 24, m = 16, seed = 20261107, rho = 0.3)
    l0 = f.L * randn(f.rng, f.G); l1 = 0.4 .* l0 .+ sqrt(1 - 0.4^2) .* (f.L * randn(f.rng, f.G))
    eta = -0.1 .+ 0.25 .* f.x .+ (exp.(-0.6 .+ 0.5 .* f.z0) .* l0)[f.gidx] .+
          f.x .* (exp.(-0.75 .+ 0.6 .* f.z0) .* l1)[f.gidx]
    data = (; y = _sll_bern(f.rng, eta), x = f.x, z0 = f.z0[f.gidx], z1 = f.z0[f.gidx], group = f.group)
    fit = drm(bf(@formula(y ~ x + animal(1 + x | group)),
                 @formula(sd(group, animal, coef(intercept)) ~ z0),
                 @formula(sd(group, animal, coef(x)) ~ z1)),
              DRModels.Binomial(); data, A = f.A, se = false)
    _sll_check("Binomial animal q2", fit)

    # Binomial animal + IID SD regressions
    rng = MersenneTwister(20261108)
    G, H, m = 18, 14, 4
    ai = repeat(1:G, inner = H * m); si = repeat(repeat(1:H, inner = m), outer = G)
    zl, wl = collect(range(-1.0, 1.0; length = G)), collect(range(-1.0, 1.0; length = H))
    A = [0.45^abs(i - j) for i in 1:G, j in 1:G]
    xx = randn(rng, length(ai))
    eta = -0.2 .+ 0.3 .* xx .+ (exp.(-0.75 .+ 0.5 .* zl) .* (cholesky(Symmetric(A)).L * randn(rng, G)))[ai] .+
          (exp.(-0.35 .- 0.6 .* wl) .* randn(rng, H))[si]
    data = (; y = _sll_bern(rng, eta), x = xx, animal_id = ["animal_$(i)" for i in ai],
            site = ["site_$(i)" for i in si], z = zl[ai], w = wl[si])
    fit = drm(bf(@formula(y ~ x + animal(1 | animal_id) + (1 | site)),
                 @formula(sd(animal_id, animal) ~ z), @formula(sd(site) ~ w)),
              DRModels.Binomial(); data, A, se = true)
    _sll_check("Binomial animal + IID SD", fit)

    # Binomial relmat + IID SD
    K = [0.45^abs(i - j) for i in 1:G, j in 1:G]
    rg, ig = repeat(1:G, inner = H * m), repeat(repeat(1:H, inner = m), outer = G)
    rz, iw = rand(rng, [-1.0, 1.0], G), rand(rng, [-1.0, 1.0], H)
    eta = -0.1 .+ 0.25 .* xx .+ (cholesky(Symmetric(K)).L * randn(rng, G) .* exp.(-0.65 .+ 0.35 .* rz))[rg] .+
          (randn(rng, H) .* exp.(-0.8 .- 0.3 .* iw))[ig]
    data = (; y = _sll_bern(rng, eta), x = xx, rel_z = rz[rg], iid_w = iw[ig],
            rel_group = rg, iid_group = ig)
    fit = drm(bf(@formula(y ~ x + (1 | iid_group) + relmat(1 | rel_group)),
                 @formula(sd(iid_group) ~ iid_w), @formula(sd(rel_group, relmat) ~ rel_z)),
              DRModels.Binomial(); data, K, se = true)
    _sll_check("Binomial relmat + IID SD", fit)

    # Binomial crossed intercepts
    G2 = H2 = 14
    g = repeat(1:G2, inner = H2 * 4); h = repeat(repeat(1:H2, inner = 4), outer = G2)
    wg, vh = repeat([-1.0, 1.0], inner = G2 ÷ 2), repeat([-1.0, 1.0], inner = H2 ÷ 2)
    xc = randn(rng, length(g))
    eta = -0.2 .+ 0.35 .* xc .+ (exp.(log(0.45) .+ 0.3 .* wg) .* randn(rng, G2))[g] .+
          (exp.(log(0.5) .- 0.25 .* vh) .* randn(rng, H2))[h]
    data = (; y = _sll_bern(rng, eta), x = xc, g, h, w = wg[g], v = vh[h])
    fit = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sd(g) ~ w), @formula(sd(h) ~ v)),
              DRModels.Binomial(); data)
    _sll_check("Binomial crossed SD", fit)

    # Binomial phylogenetic intercept SD
    function newick(depth; height = 1.0)
        node(prefix, d) = d == 0 ? "$(prefix):$(height / depth)" :
            string("(", node(prefix * "a", d - 1), ",", node(prefix * "b", d - 1), ")",
                   d == depth ? ";" : ":$(height / depth)")
        return node("tip", depth)
    end
    tree = newick(4; height = 2.5)
    phy = DRModels.augmented_phy(tree)
    Gt, mt = phy.n_leaves, 20
    tip_order = randperm(rng, Gt)
    species = repeat(phy.leaf_names[tip_order], inner = mt)
    tip_index = repeat(tip_order, inner = mt)
    ztip = repeat([-1.0, 1.0], Gt ÷ 2)
    xt = repeat(collect(range(-1.0, 1.0; length = mt)), outer = Gt)
    Kt = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    b = exp.(-0.6 .+ 0.35 .* ztip) .* (cholesky(Symmetric(Kt)).L * randn(rng, Gt))
    succ = _sll_bern(rng, -0.2 .+ 0.3 .* xt .+ b[tip_index])
    data = (; successes = succ, failures = 1 .- succ, x = xt, z = ztip[tip_index], species)
    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo(1 | species)),
                 @formula(sd(species, phylogenetic) ~ z)),
              DRModels.Binomial(); data, tree, se = true, g_tol = 1e-6)
    _sll_check("Binomial phylo SD", fit)

    # Binomial phylogenetic-interaction SD
    function inter_tree(prefix; depth = 3, height = 1.0)
        node(name, d) = d == 0 ? "$(name):$(height / 3)" :
            string("(", node(name * "a", d - 1), ",", node(name * "b", d - 1), ")",
                   d == 3 ? "" : ":$(height / 3)")
        return node(prefix, depth) * ";"
    end
    tree1, tree2 = inter_tree("host"), inter_tree("poll")
    phy1, phy2 = DRModels.augmented_phy(tree1), DRModels.augmented_phy(tree2)
    host = String[]; pollinator = String[]; pair = String[]; xi = Float64[]; zi = Float64[]
    zpair = shuffle!(rng, vcat(fill(-1.0, 32), fill(1.0, 32)))
    pidx = 0
    for h in phy1.leaf_names, p in phy2.leaf_names
        pidx += 1
        for _ in 1:12
            push!(host, h); push!(pollinator, p); push!(pair, "$(h)__$(p)")
            push!(xi, 2rand(rng) - 1); push!(zi, zpair[pidx])
        end
    end
    Q1, node1, _ = DRModels._poisson_phylo_setup(tree1, host)
    Q2, node2, _ = DRModels._poisson_phylo_setup(tree2, pollinator)
    latent = cholesky(Symmetric(kron(Q2, Q1))).U \ randn(rng, size(Q1, 1) * size(Q2, 1))
    node = (node2 .- 1) .* size(Q1, 1) .+ node1
    succ = _sll_bern(rng, -0.2 .+ 0.3 .* xi .+
                     DRModels._binomial_sd_scale(-0.2 .+ 0.8 .* zi) .* latent[node])
    order = randperm(rng, length(xi))
    data = (; successes = succ[order], failures = 1 .- succ[order], x = xi[order],
            z = zi[order], host = host[order], pollinator = pollinator[order], pair = pair[order])
    fit = drm(bf(@formula(cbind(successes, failures) ~ x + phylo_interaction(1 | host & pollinator)),
                 @formula(sd(pair, phylogenetic_interaction) ~ z)),
              DRModels.Binomial(); data, tree1, tree2, g_tol = 1e-5)
    _sll_check("Binomial phylo-interaction SD", fit)

    # Binomial mesh spatial SD
    Gm, mm = 30, 16
    site_index = repeat(1:Gm; inner = mm)
    zg = repeat([-1.0, 1.0], inner = Gm ÷ 2)
    rows = collect(1:(Gm * mm))
    Qm = spdiagm(-1 => fill(-0.35, Gm - 1), 0 => fill(2.0, Gm), 1 => fill(-0.35, Gm - 1))
    projection = sparse(vcat(rows, rows), vcat(site_index, min.(site_index .+ 1, Gm)),
                        vcat(fill(0.75, length(rows)), fill(0.25, length(rows))), Gm * mm, Gm)
    mesh = (; projection, precision = Qm, alignment_ids = rows)
    field = cholesky(Symmetric(Matrix(Qm))).U \ randn(rng, Gm)
    xm = randn(rng, Gm * mm)
    eta = 0.15 .+ 0.25 .* xm .+ exp.(-0.55 .+ 0.45 .* zg)[site_index] .* (projection * field)
    data = (; y = _sll_bern(rng, eta), x = xm, z = zg[site_index],
            site = ["site_$g" for g in site_index], row_id = rows)
    fit = drm(bf(@formula(y ~ x + spatial(1 | site)), @formula(sd(site, spatial) ~ z)),
              DRModels.Binomial(); data, mesh, se = false, g_tol = 1e-5)
    _sll_check("Binomial mesh SD", fit)
end
end
