# Post-fit methods on the wave-2 SD-regression fits:
#   Gamma / NB2 two-field shapes (src/twofield_sd_laplace.jl; fixtures `tf_case`)
#   phylo / coordinate-spatial (1 + x | g) q2 for Gaussian, Poisson, NB2
#   (src/structured_q2_phylo_spatial.jl; fixtures `_psq2_*`).
#
# Checked per shape:
#   predict            population level: link = X beta, response = g^{-1}(X beta),
#                      on newdata that carries only `x` (no grouping or label column).
#   predict_parameters mu = g^{-1}(X beta); each SD block = exp(Z alpha) of its row.
#   simulate           default re = :marginal, against an INDEPENDENT reference
#                      simulator written here from the fixture's own fields
#                      (labels, kernel, loading, SD predictor) and Distributions.jl.
#                      Statistic: per-row means of h(y) (h = log for Gamma,
#                      log1p for counts, identity for Gaussian), two-sample z_i.
#                      Pass: mean(z_i^2) < 1.5 and mean per-row variance of h(y)
#                      within 10% of the reference. Wrong-answer variant: the
#                      reference with the random effects removed must differ in
#                      that variance by > 12%.
#                      Correlated fits: per-row variance of h(y) on rows x > 0.5
#                      and x < -0.5 must match the reference within 15%, and the
#                      reference with rho -> -rho must miss by > 15% in a group.
#   bootstrap_result   B = 2, failures = :skip: no simulator failure.
#   drm_bridge         same coefficients as drm(); the label `p` is never a data
#                      column; dpars_newdata[mu] = g^{-1}(X beta).
#   ranef              refuses (no stored conditional modes), as for `:sd` (#759).
# Tolerances were fixed before the first run. The no-random-effect wrong-answer
# check was first written on the mean of h(y); the first run showed that it
# cannot discriminate (E log y does not depend on a mean-zero latent field for
# Gamma), so it was moved to the variance of h(y). A 25% threshold then failed
# on the phylo Poisson/NB2 fixtures, whose latent SDs are small (no-RE variance
# off by 17-21% while the correct reference matched within 1-3%), so the
# wrong-answer threshold is 12%: about 4x the largest correct-reference miss.
using DRModels, Test, LinearAlgebra, Random, Statistics
import Distributions, ForwardDiff
using SpecialFunctions: loggamma

# Reuse the fixture builders (not the fits) of the two route test files.
if !isdefined(Main, :tf_case)
    let f = joinpath(@__DIR__, "test_gamma_nb2_twofield_sd.jl"), src = read(f, String)
        include_string(Main, src[1:first(findfirst("for (fam, shape) in", src)) - 1], f)
    end
end
if !isdefined(Main, :_psq2_fixture)
    let f = joinpath(@__DIR__, "test_phylo_spatial_q2_sd.jl"), src = read(f, String)
        # file name passed so @__DIR__ in the fixture reader is test/, not the cwd
        include_string(Main, src[1:first(findfirst("@testset", src)) - 1], f)
    end
end

const W2_FSTR = Dict(
    :samegroup => "y ~ x + (1 | g) + (0 + x | g); sigma ~ 1; sd(g, coef(intercept)) ~ w1; sd(g, coef(x)) ~ w2",
    :crossed => "y ~ x + (1 | g) + (1 | h); sigma ~ 1; sd(g) ~ w1; sd(h) ~ w2",
    :ordrelmat => "y ~ x + (1 | g) + relmat(1 | p); sigma ~ 1; sd(g) ~ w1; sd(p, relmat) ~ w2",
    :ordanimal => "y ~ x + (1 | g) + animal(1 | p); sigma ~ 1; sd(g) ~ w1; sd(p, animal) ~ w2",
    :relmatq2 => "y ~ x + relmat(1 + x | g); sigma ~ 1; sd(g, relmat, coef(intercept)) ~ w1; sd(g, relmat, coef(x)) ~ w2",
    :labanimal => "y ~ x + animal(1 + x | p | g); sigma ~ 1; sd(g, animal, coef(intercept)) ~ w1; sd(g, animal, coef(x)) ~ w2",
    :labrelmat => "y ~ x + relmat(1 + x | p | g); sigma ~ 1; sd(g, relmat, coef(intercept)) ~ w1; sd(g, relmat, coef(x)) ~ w2",
    :corr => "y ~ x + (1 + x | g); sigma ~ 1; sd(g, coef(intercept)) ~ w1; sd(g, coef(x)) ~ w2")

# Independent reference draw. `fields` = [(gi, C, w, sd)] with gi the row -> level
# index into C / sd; latent u_k ~ N(0, C_k); correlated pair u2 = rho u1 + sqrt(1-rho^2) e.
function w2_ref_draw(rng, famsym, eta0, fields, rho, sigma)
    eta = copy(eta0)
    us = Vector{Vector{Float64}}(undef, length(fields))
    for (k, (gi, C, w, sd)) in enumerate(fields)
        us[k] = cholesky(Symmetric(C)).L * randn(rng, size(C, 1))
    end
    if rho !== nothing
        us[2] = rho .* us[1] .+ sqrt(1 - rho^2) .* us[2]
    end
    for (k, (gi, C, w, sd)) in enumerate(fields)
        eta .+= w .* sd[gi] .* us[k][gi]
    end
    famsym === :gaussian && return eta .+ sigma .* randn(rng, length(eta))
    mu = exp.(eta)
    famsym === :poisson && return Float64[rand(rng, Distributions.Poisson(m)) for m in mu]
    famsym === :gamma && return [rand(rng, Distributions.Gamma(1 / sigma^2, m * sigma^2)) for m in mu]
    k = 1 / sigma^2
    return Float64[rand(rng, Distributions.NegativeBinomial(k, k / (k + m))) for m in mu]
end

w2_h(famsym) = famsym === :gamma ? log : famsym === :gaussian ? identity : log1p

function w2_compare(Y, Yref, famsym)
    h = w2_h(famsym)
    H, R = h.(Y), h.(Yref)
    m1, m2 = vec(mean(H; dims = 2)), vec(mean(R; dims = 2))
    v1, v2 = vec(var(H; dims = 2)), vec(var(R; dims = 2))
    z = (m1 .- m2) ./ sqrt.((v1 ./ size(H, 2)) .+ (v2 ./ size(R, 2)) .+ 1e-12)
    return mean(abs2, z), v1, v2
end

w2_refmat(nsim, seed, args...) =
    (rng = MersenneTwister(seed); reduce(hcat, [w2_ref_draw(rng, args...) for _ in 1:nsim]))

# Shared post-fit checks. `fields`: reference fields with fitted SDs; `rho` the
# fitted correlation or `nothing`; `link` the inverse link.
function w2_check_postfit(fit, data, kw, famsym, fields, rho; label)
    n = length(data.y)
    beta = coef(fit, :mu)
    sigma = famsym === :poisson ? NaN : exp(only(coef(fit, :sigma)))
    eta0 = beta[1] .+ beta[2] .* data.x
    inv_link = famsym === :gaussian ? identity : exp

    @testset "$label: predict / predict_parameters" begin
        nd = (; x = [-1.0, 0.0, 0.7, 1.3])
        eta_nd = beta[1] .+ beta[2] .* nd.x
        @test predict(fit, nd; type = :link) ≈ eta_nd atol = 1e-10
        @test predict(fit, nd) ≈ inv_link.(eta_nd) atol = 1e-10
        @test predict(fit, data; type = :link) ≈ eta0 atol = 1e-10
        # wrong answer: the conditional mean would add a random effect
        @test !(predict(fit, data; type = :link) ≈ eta0 .+ 0.1)
        pp = predict_parameters(fit, data)
        @test pp[:mu] ≈ inv_link.(eta0) atol = 1e-10
        famsym === :poisson || @test pp[:sigma] ≈ fill(sigma, n) atol = 1e-10
        targets = DRModels._sd_target_parts(fit.formula)
        for (k, t) in enumerate(targets)
            blk = DRModels._sd_target_block(t.target; disambiguate = true)
            gi, _, _, sd = fields[k]
            @test pp[blk] ≈ sd[gi] rtol = 1e-8
        end
    end

    @testset "$label: simulate (re = :marginal)" begin
        nsim = 400
        Y = simulate(fit; nsim, rng = MersenneTwister(91), data, kw...)
        @test size(Y) == (n, nsim)
        @test all(isfinite, Y)
        famsym in (:poisson, :nb2) && @test all(y -> y >= 0 && isinteger(y), Y)
        famsym === :gamma && @test all(>(0), Y)
        Yref = w2_refmat(nsim, 92, famsym, eta0, fields, rho, sigma)
        stat, v1, v2 = w2_compare(Y, Yref, famsym)
        @test stat < 1.5
        @test abs(mean(v1) / mean(v2) - 1) < 0.10
        # wrong answer: no random effects (E h(y) barely moves for Gamma, so the
        # variance of h(y) is the discriminating moment)
        noRE = [(gi, C, w, zero(sd)) for (gi, C, w, sd) in fields]
        _, _, v0 = w2_compare(Y, w2_refmat(nsim, 93, famsym, eta0, noRE, nothing, sigma), famsym)
        @test abs(mean(v1) / mean(v0) - 1) > 0.12
        println("W2STAT $label stat=$(round(stat; digits = 3)) vratio=$(round(mean(v1) / mean(v2); digits = 3)) vratio_noRE=$(round(mean(v1) / mean(v0); digits = 3))")
        if rho !== nothing
            pos, neg = findall(>(0.5), data.x), findall(<(-0.5), data.x)
            ratio(a, b, rows) = mean(a[rows]) / mean(b[rows])
            @test abs(ratio(v1, v2, pos) - 1) < 0.15
            @test abs(ratio(v1, v2, neg) - 1) < 0.15
            _, _, vf = w2_compare(Y, w2_refmat(nsim, 94, famsym, eta0, fields, -rho, sigma), famsym)
            @test max(abs(ratio(v1, vf, pos) - 1), abs(ratio(v1, vf, neg) - 1)) > 0.15
            println("W2RHO $label rho=$(round(rho; digits = 3)) pos=$(round(ratio(v1, v2, pos); digits = 3)) neg=$(round(ratio(v1, v2, neg); digits = 3)) flip_pos=$(round(ratio(v1, vf, pos); digits = 3)) flip_neg=$(round(ratio(v1, vf, neg); digits = 3))")
        end
        # re = :zero still draws without the random effects
        Y0 = simulate(fit; nsim = 50, rng = MersenneTwister(95), data, re = :zero)
        @test size(Y0) == (n, 50)
    end

    @testset "$label: bootstrap_result" begin
        boot = bootstrap_result(fit; data, B = 2, rng = MersenneTwister(96),
                                failures = :skip, kw...)
        @test boot.attempted == 2
        @test boot.used + boot.failed == 2
        @test all(!occursin(r"simulat|marginal", string(r.message)) for r in boot.failures)
    end

    @testset "$label: ranef refuses" begin
        @test_throws ArgumentError ranef(fit)
    end
end

@testset "wave-2 Gamma/NB2 two-field fits: post-fit methods" begin
    for (fam, shape) in ((:nb2, :samegroup), (:gamma, :samegroup), (:gamma, :crossed),
                         (:gamma, :ordrelmat), (:gamma, :ordanimal), (:gamma, :relmatq2),
                         (:nb2, :crossed), (:nb2, :ordrelmat), (:gamma, :labanimal),
                         (:gamma, :labrelmat), (:nb2, :labanimal), (:nb2, :corr))
        case = tf_case(fam, shape)
        fit = drm(case.formula, TF_FAMS[fam]; data = case.data, case.kw...)
        @test fit.converged
        targets = DRModels._sd_target_parts(case.formula)
        blocks = [DRModels._sd_target_block(t.target; disambiguate = true) for t in targets]
        fields = map(enumerate(case.fields)) do (k, (labels, levels, C, w, z))
            a = coef(fit, blocks[k])
            (indexin(labels, levels), Matrix{Float64}(C), w, exp.(a[1] .+ a[2] .* z))
        end
        rho = any(p -> first(p) === :recov, fit.blocks) ? tanh(only(coef(fit, :recov))) : nothing
        (shape in (:labanimal, :labrelmat, :corr)) == (rho !== nothing) || error("fixture")
        w2_check_postfit(fit, case.data, case.kw, fam, fields, rho; label = "$fam $shape")
        rho === nothing || @testset "$fam $shape: rho label" begin
            @test only(last(only(filter(p -> first(p) === :recov, fit.coefnames)))) == "g:atanh_rho"
        end

        @testset "$fam $shape: drm_bridge" begin
            nd = map(v -> v[1:5], case.data)
            br = drm_bridge(; formula = W2_FSTR[shape], family = fam === :gamma ? "gamma" : "nbinom2",
                            data = case.data, case.kw..., newdata = nd)
            @test br["coefficients"] ≈ coef(fit) atol = 1e-6
            @test length(unique(br["coef_names"])) == length(fit.theta)
            # the label `p` adds no mean column: only (Intercept) and x
            @test count(startswith("mu_"), br["coef_names"]) == 2
            beta = coef(fit, :mu)
            @test br["dpars_newdata"]["mu"] ≈ exp.(beta[1] .+ beta[2] .* nd.x) atol = 1e-8
            @test !(br["dpars_newdata"]["mu"] ≈ exp.(beta[1] .+ beta[2] .* nd.x) .* 1.01)
        end
    end
end

@testset "wave-2 phylo/spatial q2 fits: post-fit methods" begin
    nwk, ids, Cphy = _psq2_tree()
    xy, Csp = _psq2_grid(length(ids))
    coords = Dict(ids[k] => [xy[k]...] for k in eachindex(ids))
    for (kind, C) in ((:phylo, Cphy), (:spatial, Csp)),
        (fam, fsym) in ((Gaussian(), :gaussian), (Poisson(), :poisson), (NegBinomial2(), :nb2))
        fx = _psq2_fixture(C, ids; family = fsym)
        kw = kind === :phylo ? (; tree = nwk) : (; coords)
        fit = drm(_psq2_formula(kind, fam), fam; data = fx.data, kw...)
        @test fit.converged
        _, _, a0, a1 = _psq2_theta(fit)
        fields = [(fx.gi, Matrix{Float64}(C), ones(length(fx.data.y)), exp.(a0[1] .+ a0[2] .* fx.z0)),
                  (fx.gi, Matrix{Float64}(C), fx.data.x, exp.(a1[1] .+ a1[2] .* fx.z1))]
        w2_check_postfit(fit, fx.data, kw, fsym, fields, nothing; label = "$kind $fsym")

        @testset "$kind $fsym: simulate needs the kernel" begin
            @test_throws ArgumentError simulate(fit; data = fx.data)
        end
        @testset "$kind $fsym: drm_bridge" begin
            lev = kind === :phylo ? "phylogenetic" : "spatial"
            fstr = "y ~ x + $kind(1 + x | id); " * (fsym === :poisson ? "" : "sigma ~ 1; ") *
                   "sd(id, $lev, coef(intercept)) ~ z0; sd(id, $lev, coef(x)) ~ z1"
            nd = map(v -> v[1:5], fx.data)
            br = drm_bridge(; formula = fstr, family = fsym === :nb2 ? "nbinom2" : String(fsym),
                            data = fx.data, kw..., newdata = nd)
            @test br["coefficients"] ≈ coef(fit) atol = 1e-6
            beta = coef(fit, :mu)
            il = fsym === :gaussian ? identity : exp
            @test br["dpars_newdata"]["mu"] ≈ il.(beta[1] .+ beta[2] .* nd.x) atol = 1e-8
        end
    end
end
