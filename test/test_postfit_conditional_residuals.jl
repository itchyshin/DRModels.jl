# Quantile residuals on the SD-regression routes condition on the Laplace modes.
#
# drmTMB contract (R 0.8.1, `drm_quantile_residuals()` -> `fitted_distribution()` ->
# `predict_parameters(newdata = NULL)`): the distribution F_i is evaluated at the
# FITTED-ROW parameters, and for a fit with random effects those include the
# conditional modes at the fitted theta (the Laplace modes). So for a random-effect
# SD-regression fit the residual is not the population-level one. DRModels.jl used
# the population level (random effects set to 0) on these routes, which left the
# residuals over-dispersed (SD 1.3-2.5). It now records the per-row link-scale mode
# contribution `scales[:cond_eta_re]` at theta_hat on every route below and
# `residuals(fit; type = :quantile)` uses it. `fitted(fit)` stays population level.
#
# Oracle (shares no fitter code): from the fixture's own fields and the fitted
# coefficients build the dense latent design L (n x m) and prior precision P (m x m),
# find the mode of  -sum_i log p(y_i | eta0_i + (L u)_i) + u'Pu/2  by an own damped
# Newton on ForwardDiff derivatives and Distributions.jl log densities, form the
# conditional parameters, and compute the residuals from Distributions.jl. Continuous
# families (Gaussian, Gamma) are deterministic and are compared elementwise; count
# families replicate the Dunn-Smyth draw u = F(y-) + (F(y) - F(y-)) * rand(rng) with
# the same seeded generator.
#
# Tolerances fixed before the first run: |r_julia - r_oracle| < 1e-6 elementwise;
# |cond_eta_re - L u| < 1e-6. Wrong-answer variant: the same oracle at u = 0 (the
# population level, the old behaviour) must differ from the Julia residuals by > 0.2
# somewhere, so the check can tell the two apart.
using DRModels, Test, LinearAlgebra, Random, Statistics
import Distributions, ForwardDiff
using SpecialFunctions: loggamma

# Reuse the fixture builders (not the fits) of the route test files.
if !isdefined(Main, :tf_case)
    let src = read(joinpath(@__DIR__, "test_gamma_nb2_twofield_sd.jl"), String)
        include_string(Main, src[1:first(findfirst("for (fam, shape) in", src)) - 1])
    end
end
if !isdefined(Main, :_psq2_fixture)
    let src = read(joinpath(@__DIR__, "test_phylo_spatial_q2_sd.jl"), String)
        include_string(Main, src[1:first(findfirst("@testset", src)) - 1])
    end
end
if !isdefined(Main, :_lord_fixture)
    let src = read(joinpath(@__DIR__, "test_labelled_ordinary_q2_sd.jl"), String)
        include_string(Main, src[1:first(findfirst("@testset", src)) - 1])
    end
end
if !isdefined(Main, :_q34_fixture)
    let src = read(joinpath(@__DIR__, "test_gaussian_ordinary_q3_q4_sd.jl"), String)
        include_string(Main, src[1:first(findfirst("@testset", src)) - 1])
    end
end
if !isdefined(Main, :_gq2_fixture)
    let src = read(joinpath(@__DIR__, "test_gaussian_structured_q2_sd.jl"), String)
        include_string(Main, src[1:first(findfirst("@testset", src)) - 1])
    end
end

# ---- oracle machinery --------------------------------------------------------

# Per-observation log density at (linear predictor on the mean link, log sigma).
function cr_logpdf(fam::Symbol, y, lin, ls)
    k = exp(-2ls)
    fam === :gaussian && return Distributions.logpdf(Distributions.Normal(lin, exp(ls)), y)
    fam === :poisson && return Distributions.logpdf(Distributions.Poisson(exp(lin)), y)
    fam === :nb2 && return Distributions.logpdf(
        Distributions.NegativeBinomial(k, k / (k + exp(lin))), y)
    return Distributions.logpdf(Distributions.Gamma(k, exp(lin) / k), y)
end

# Damped Newton for the minimiser of f (own code).
function cr_newton(f, m)
    u = zeros(m)
    for _ in 1:300
        g = ForwardDiff.gradient(f, u)
        maximum(abs, g) < 1e-11 && break
        H = ForwardDiff.hessian(f, u)
        C = cholesky(Symmetric(H); check = false)
        lam = 1e-4
        while !issuccess(C)
            C = cholesky(Symmetric(H + lam * I); check = false); lam *= 10
        end
        step = C \ g
        t = 1.0; f0 = f(u)
        while t > 1e-12 && f(u .- t .* step) > f0 - 1e-4 * t * dot(g, step)
            t /= 2
        end
        u .-= (t > 1e-12 ? t : 1.0) .* step
    end
    return u
end

# Mode of the unit-scale latent field u, prior precision P, rows eta = eta0 + L u,
# log sigma fixed at ls (a scalar).
function cr_modes(fam::Symbol, y, eta0, L, P, ls)
    nj(u) = begin
        eta = eta0 .+ L * u
        s = 0.5 * dot(u, P * u)
        for i in eachindex(y)
            s -= cr_logpdf(fam, y[i], eta[i], ls)
        end
        s
    end
    return cr_newton(nj, size(L, 2))
end

# Dunn-Smyth residuals from Distributions.jl at conditional (lin, ls) per row.
function cr_resid(fam::Symbol, y, lin, ls; seed)
    n = length(y)
    ls = ls isa Real ? fill(float(ls), n) : ls
    rng = Random.MersenneTwister(seed)
    lo, hi = eps(), 1 - eps()
    N = Distributions.Normal()
    r = zeros(n)
    for i in 1:n
        k = exp(-2ls[i])
        if fam === :gaussian
            u = Distributions.cdf(Distributions.Normal(lin[i], exp(ls[i])), y[i])
        elseif fam === :gamma
            u = Distributions.cdf(Distributions.Gamma(k, exp(lin[i]) / k), y[i])
        else
            d = fam === :poisson ? Distributions.Poisson(exp(lin[i])) :
                Distributions.NegativeBinomial(k, k / (k + exp(lin[i])))
            a = y[i] > 0 ? Distributions.cdf(d, y[i] - 1) : 0.0
            b = Distributions.cdf(d, y[i])
            u = a + (b - a) * rand(rng)
        end
        r[i] = Distributions.quantile(N, clamp(u, lo, hi))
    end
    return r
end

const CR_SEED = 4242

# One check: fit, oracle modes, residual agreement, wrong-answer variant.
function cr_check(fit, fam::Symbol, y, eta0, L, P, ls; label)
    @testset "$label" begin
        @test fit.converged
        @test haskey(fit.scales, :cond_eta_re)
        u = cr_modes(fam, y, eta0, L, P, ls)
        @test fit.scales[:cond_eta_re] ≈ L * u atol = 1e-6
        rj = residuals(fit; type = :quantile, rng = Random.MersenneTwister(CR_SEED))
        ro = cr_resid(fam, y, eta0 .+ L * u, ls; seed = CR_SEED)
        r0 = cr_resid(fam, y, eta0, ls; seed = CR_SEED)
        @test maximum(abs, rj .- ro) < 1e-6
        # the population-level (old) residuals differ, so the comparison has power
        @test maximum(abs, rj .- r0) > 0.2
        println("CRRESID $label sd_julia=$(round(std(rj); digits = 3)) ",
                "sd_population=$(round(std(r0); digits = 3)) ",
                "maxdiff_oracle=$(round(maximum(abs, rj .- ro); sigdigits = 2))")
        # fitted() is still the population-level mean
        @test haskey(fit.means, :mu)
    end
end

# ---- A. ordinary labelled (1 + x | p | g) Poisson / NB2 ---------------------

@testset "conditional quantile residuals: ordinary labelled q2 (Poisson, NB2)" begin
    fx = _lord_fixture(); data = fx.data; G = fx.G; gi = fx.gi
    n = length(gi)
    for (family, resp, fsym) in ((Poisson(), :yp, :poisson), (NegBinomial2(), :yn, :nb2))
        fit = drm(_lord_formula(resp, family), family; data)
        th = fit.theta
        off = fsym === :poisson ? 0 : 1
        beta = th[1:2]
        a0, a1 = th[(3 + off):(4 + off)], th[(5 + off):(6 + off)]
        rho = 0.999999 * tanh(th[7 + off])
        ls = fsym === :poisson ? 0.0 : th[3]
        l0 = exp.(hcat(ones(n), data.z0) * a0)
        l1 = data.x .* exp.(hcat(ones(n), data.z1) * a1)
        L = zeros(n, 2G)
        for i in 1:n
            L[i, gi[i]] = l0[i]; L[i, G + gi[i]] = l1[i]
        end
        P = inv(Symmetric(kron([1.0 rho; rho 1.0], Matrix{Float64}(I, G, G))))
        eta0 = hcat(ones(n), data.x) * beta
        y = getproperty(data, resp)
        cr_check(fit, fsym, y, eta0, L, Matrix(P), ls; label = "labelled ordinary $fsym")
        @test fit.means[:mu] ≈ exp.(eta0) atol = 1e-12
    end
end

# ---- B. Gamma / NB2 two-field routes ----------------------------------------

@testset "conditional quantile residuals: Gamma / NB2 two-field routes" begin
    for (fam, shape) in ((:nb2, :samegroup), (:gamma, :samegroup), (:gamma, :crossed),
                         (:gamma, :ordrelmat), (:gamma, :relmatq2), (:nb2, :crossed),
                         (:gamma, :labanimal), (:nb2, :corr))
        case = tf_case(fam, shape)
        fit = drm(case.formula, TF_FAMS[fam]; data = case.data, case.kw...)
        targets = DRModels._sd_target_parts(case.formula)
        blocks = [DRModels._sd_target_block(t.target; disambiguate = true) for t in targets]
        beta = coef(fit, :mu); ls = only(coef(fit, :sigma))
        n = length(case.data.y)
        fields = map(enumerate(case.fields)) do (k, (labels, levels, C, w, z))
            a = coef(fit, blocks[k])
            (indexin(labels, levels), Matrix{Float64}(C), w, exp.(a[1] .+ a[2] .* z))
        end
        Gs = [size(f[2], 1) for f in fields]
        offs = cumsum([0; Gs[1:end-1]])
        L = zeros(n, sum(Gs))
        for (k, (gix, C, w, sd)) in enumerate(fields), i in 1:n
            L[i, offs[k] + gix[i]] = w[i] * sd[gix[i]]
        end
        rho = any(p -> first(p) === :recov, fit.blocks) ? tanh(only(coef(fit, :recov))) : nothing
        P = if rho === nothing
            Matrix(cat(inv(Symmetric(fields[1][2])), inv(Symmetric(fields[2][2])); dims = (1, 2)))
        else
            Matrix(kron(inv([1.0 rho; rho 1.0]), inv(Symmetric(fields[1][2]))))
        end
        eta0 = hcat(ones(n), case.data.x) * beta
        cr_check(fit, fam, case.data.y, eta0, L, P, ls; label = "$fam $shape")
    end
end

# ---- C. phylo / spatial (1 + x | g) q2, Gaussian / Poisson / NB2 ------------

@testset "conditional quantile residuals: phylo / spatial q2" begin
    nwk, ids, Cphy = _psq2_tree()
    xy, Csp = _psq2_grid(length(ids))
    coords = Dict(ids[k] => [xy[k]...] for k in eachindex(ids))
    for (kind, C) in ((:phylo, Cphy), (:spatial, Csp)),
        (fam, fsym) in ((Gaussian(), :gaussian), (Poisson(), :poisson), (NegBinomial2(), :nb2))
        fx = _psq2_fixture(C, ids; family = fsym)
        kw = kind === :phylo ? (; tree = nwk) : (; coords)
        fit = drm(_psq2_formula(kind, fam), fam; data = fx.data, kw...)
        beta, logsig, a0, a1 = _psq2_theta(fit)
        d = fx.data; gi = fx.gi; n = length(d.y); G = size(C, 1)
        s0 = exp.(a0[1] .+ a0[2] .* fx.z0); s1 = exp.(a1[1] .+ a1[2] .* fx.z1)
        L = zeros(n, 2G)
        for i in 1:n
            L[i, gi[i]] = s0[gi[i]]; L[i, G + gi[i]] = d.x[i] * s1[gi[i]]
        end
        Ci = Matrix(inv(Symmetric(C)))
        P = Matrix(cat(Ci, Ci; dims = (1, 2)))
        eta0 = beta[1] .+ beta[2] .* d.x
        cr_check(fit, fsym, d.y, eta0, L, P, fsym === :poisson ? 0.0 : logsig;
                 label = "$kind $fsym")
    end
end

# ---- D. Gaussian ordinary q3 / q4 correlated block --------------------------

@testset "conditional quantile residuals: Gaussian ordinary q3 / q4" begin
    for q in (3, 4)
        fx = _q34_fixture(q); d = fx.data; n = length(d.y); G = fx.G
        fit = drm(_q34_formula(fx), Gaussian(); data = d)
        A, Rm, beta, sigma = _q34_parts(fit, fx)
        L = zeros(n, q * G)
        for i in 1:n, a in 1:q
            L[i, (a - 1) * G + fx.gi[i]] = fx.U[i, a] * exp(A[a, 1] + A[a, 2] * fx.w[fx.gi[i]])
        end
        P = Matrix(inv(Symmetric(kron(Matrix(Rm), Matrix{Float64}(I, G, G)))))
        eta0 = fx.U[:, 1:3] * beta
        cr_check(fit, :gaussian, d.y, eta0, L, P, log(sigma); label = "Gaussian q$q")
    end
end

# ---- E. Gaussian structured q2 (animal, relmat) -----------------------------

@testset "conditional quantile residuals: Gaussian structured q2" begin
    for kind in (:animal, :relmat)
        fx = _gq2_fixture()
        data, formula, kw = fx.data, _gq2_formula(kind), _gq2_kw(kind, fx)
        fit = drm(formula, Gaussian(); data, kw...)
        sdnames = [first(p) for p in fit.blocks if first(p) ∉ (:mu, :sigma)]
        a0 = coef(fit, sdnames[1]); a1 = coef(fit, sdnames[2])
        n, G = length(data.y), length(fx.z0)
        d0 = exp.(hcat(ones(G), fx.z0) * a0); d1 = exp.(hcat(ones(G), fx.z1) * a1)
        L = zeros(n, 2G)
        for i in 1:n
            L[i, fx.gidx[i]] = d0[fx.gidx[i]]; L[i, G + fx.gidx[i]] = data.x[i] * d1[fx.gidx[i]]
        end
        Ci = Matrix(inv(Symmetric(fx.C)))
        eta0 = hcat(ones(n), data.x) * coef(fit, :mu)
        cr_check(fit, :gaussian, data.y, eta0, L, Matrix(cat(Ci, Ci; dims = (1, 2))),
                 only(coef(fit, :sigma)); label = "Gaussian $kind q2")
    end
end

# ---- F. joint mu + sigma ordinary random intercepts, each with an SD formula --

function cr_joint_fixture(fam::Symbol; seed, n_id = 30, n_g = 24, n = 600, same = false)
    rng = Random.MersenneTwister(seed)
    n_g = same ? n_id : n_g
    gi = vcat(1:n_id, rand(rng, 1:n_id, n - n_id))
    gs = same ? gi : vcat(1:n_g, rand(rng, 1:n_g, n - n_g))
    w_lvl = sin.((1:n_id) .* 1.7)
    z_lvl = same ? w_lvl : cos.((1:n_g) .* 0.9 .+ 1)
    a = exp.(log(0.45) .+ 0.5 .* w_lvl) .* randn(rng, n_id)
    b = exp.(log(0.35) .- 0.6 .* z_lvl) .* randn(rng, n_g)
    x = randn(rng, n)
    lin = (fam === :gaussian ? 0.2 : 0.8) .+ 0.4 .* x .+ a[gi]
    ls = (fam === :gaussian ? log(0.5) : -0.7) .+ b[gs]
    y = map(1:n) do i
        k = exp(-2ls[i])
        fam === :gaussian ? lin[i] + exp(ls[i]) * randn(rng) :
        fam === :nb2 ? Float64(rand(rng, Distributions.NegativeBinomial(k, k / (k + exp(lin[i]))))) :
                       rand(rng, Distributions.Gamma(k, exp(lin[i]) / k))
    end
    id = [string("i", lpad(string(g), 2, '0')) for g in gi]
    g = same ? id : [string("g", lpad(string(j), 2, '0')) for j in gs]
    return (; data = (; y, x, id, g, w = w_lvl[gi], z = z_lvl[gs]), gi, gs, w_lvl, z_lvl)
end

cr_joint_formula(same) = same ?
    bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | id)), @formula(sd(id) ~ w),
       @formula(sd(id, sigma) ~ z)) :
    bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g)), @formula(sd(id) ~ w),
       @formula(sd(g, sigma) ~ z))

@testset "conditional quantile residuals: joint mu + sigma (modes on both axes)" begin
    for (fam, same, seed) in ((:gaussian, true, 2026110705), (:nb2, false, 2026110701),
                              (:gamma, false, 2026110703), (:nb2, true, 2026110702))
        fx = cr_joint_fixture(fam; seed, same); d = fx.data; n = length(d.y)
        F = fam === :gaussian ? Gaussian() : fam === :nb2 ? NegBinomial2() : DRModels.Gamma()
        fit = drm(cr_joint_formula(same), F; data = d)
        @testset "$fam $(same ? "same group" : "crossed")" begin
            @test fit.converged
            ks = [first(p) for p in fit.blocks if startswith(String(first(p)), "sdtarget_")]
            key_mu = only(filter(k -> occursin(bytes2hex(codeunits("|mu|")), String(k)), ks))
            key_sg = only(filter(k -> k !== key_mu, ks))
            am, asg = coef(fit, key_mu), coef(fit, key_sg)
            sd_a = exp.(am[1] .+ am[2] .* fx.w_lvl); sd_b = exp.(asg[1] .+ asg[2] .* fx.z_lvl)
            mu0 = hcat(ones(n), d.x) * coef(fit, :mu); s0 = only(coef(fit, :sigma))
            na, nb = length(sd_a), length(sd_b)
            nj(u) = begin
                s = zero(eltype(u))
                for i in 1:n
                    s -= cr_logpdf(fam, d.y[i], mu0[i] + u[fx.gi[i]], s0 + u[na + fx.gs[i]])
                end
                s - sum(Distributions.logpdf.(Distributions.Normal.(0, sd_a), u[1:na])) -
                    sum(Distributions.logpdf.(Distributions.Normal.(0, sd_b), u[(na + 1):(na + nb)]))
            end
            u = cr_newton(nj, na + nb)
            lin_c = mu0 .+ u[fx.gi]; ls_c = s0 .+ u[na .+ fx.gs]
            rj = residuals(fit; type = :quantile, rng = Random.MersenneTwister(CR_SEED))
            ro = cr_resid(fam, d.y, lin_c, ls_c; seed = CR_SEED)
            r0 = cr_resid(fam, d.y, mu0, s0; seed = CR_SEED)
            @test maximum(abs, rj .- ro) < 1e-6
            @test maximum(abs, rj .- r0) > 0.2
            println("CRRESID joint $fam same=$same sd_julia=$(round(std(rj); digits = 3)) ",
                    "sd_population=$(round(std(r0); digits = 3)) ",
                    "maxdiff_oracle=$(round(maximum(abs, rj .- ro); sigdigits = 2))")
        end
    end
end

# ---- G. Ainv on the labelled / unlabelled animal q2 routes -------------------
# drmTMB accepts `animal(1 + x | p | g, Ainv = Ainv)` for Poisson, NB2, Gamma and
# Gaussian and fits the same model as `A = solve(Ainv)`. DRModels.jl accepted `Ainv`
# for Poisson / NB2 only; Gamma and Gaussian ended in a MethodError. They now convert
# the precision to the labelled covariance. Oracle: the same fit from `A` (the
# log-likelihood and every coefficient agree), and a wrong-answer variant (a different
# precision) that must not.
@testset "Ainv on animal q2 routes (labelled and unlabelled)" begin
    for (fam, shape) in ((:gamma, :labanimal), (:nb2, :labanimal))
        case = tf_case(fam, shape)
        A = case.kw.A
        Ainv = Matrix(Symmetric(inv(A)))
        fA = drm(case.formula, TF_FAMS[fam]; data = case.data, A)
        fP = drm(case.formula, TF_FAMS[fam]; data = case.data, Ainv)
        @test fP.converged
        @test loglikelihood(fP) ≈ loglikelihood(fA) atol = 1e-6
        @test coef(fP) ≈ coef(fA) atol = 1e-5
        # Rows matched to the wrong animals. A fixed odds-then-evens order, not a random
        # draw: `randperm(rng, n)` differs between Julia versions, and on 1.12 the old draw
        # gave a gap of only 0.36 for the Gamma fit, below the 0.5 bound.
        perm = vcat(1:2:size(A, 1), 2:2:size(A, 1))
        fW = drm(case.formula, TF_FAMS[fam]; data = case.data,
                 Ainv = Matrix(Symmetric(Ainv[perm, perm])))
        @test abs(loglikelihood(fW) - loglikelihood(fA)) > 0.5
        @test_throws ArgumentError drm(case.formula, TF_FAMS[fam]; data = case.data, A, Ainv)
        @test_throws ArgumentError drm(case.formula, TF_FAMS[fam]; data = case.data,
                                       Ainv = Ainv[1:end-1, 1:end-1])
    end
    # Gaussian labelled and unlabelled animal q2 (fixture of the structured Gaussian route).
    fx = _gq2_fixture()
    A = Matrix(Symmetric(fx.C)); Ainv = Matrix(Symmetric(inv(A)))
    for formula in (_gq2_formula(:animal),
                    bf(@formula(y ~ x + animal(1 + x | p | g)), @formula(sigma ~ 1),
                       @formula(sd(g, animal, coef(intercept)) ~ z0),
                       @formula(sd(g, animal, coef(x)) ~ z1)))
        fA = drm(formula, Gaussian(); data = fx.data, A, ids = unique(fx.data.g))
        fP = drm(formula, Gaussian(); data = fx.data, Ainv)
        @test fP.converged
        @test loglikelihood(fP) ≈ loglikelihood(fA) atol = 1e-6
        @test coef(fP) ≈ coef(fA) atol = 1e-5
    end
end
