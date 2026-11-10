# Joint mean + sigma random effects, each with an SD formula, wave 2:
#   * NB2 and Gamma joint ordinary intercepts, crossed groups and one shared
#     group (twins of drmTMB test-nbinom2-mu-sigma-joint-sd.R,
#     test-nbinom2-mu-sigma-same-group-sd.R, test-gamma-mu-sigma-joint-sd.R);
#   * Gaussian joint intercepts on ONE shared grouping factor (independent
#     latent fields, as native drmTMB; checked to fit natively in drmTMB 0.8.1);
#   * the per-observation kernels against ForwardDiff.
# Oracle: an independent dense Laplace over the RAW effects u = (a, b), written
# from Distributions.logpdf with ForwardDiff derivatives and its own Newton mode
# search (no DRModels internals), evaluated at the fitted parameters.
# Tolerances fixed before the first run: |logLik - oracle| < 1e-5; oracle mode
# |grad|_inf < 1e-8 and Hessian PD; kernel derivatives rtol 1e-8. The mode
# gradient bar was relaxed to 1e-6 after the first run (Gamma same-group oracle
# stalled at 8.5e-8 in Float64 roundoff; a gradient g moves the Laplace value by
# about g^2 / min eigenvalue, far below the 1e-5 logLik bar).
using DRModels, Random, Test, LinearAlgebra, ForwardDiff
import Distributions
using Distributions: Normal, NegativeBinomial, logpdf
using Statistics: median

function _jsw_fixture(fam::Symbol; seed, n_id = 30, n_g = 24, n = 600, same = false,
                      alpha_mu = [log(0.45), 0.5], alpha_sg = [log(0.35), -0.6])
    rng = MersenneTwister(seed)
    n_g = same ? n_id : n_g
    gi = vcat(1:n_id, rand(rng, 1:n_id, n - n_id))
    gs = same ? gi : vcat(1:n_g, rand(rng, 1:n_g, n - n_g))
    w_lvl = sin.((1:n_id) .* 1.7)
    z_lvl = same ? w_lvl : cos.((1:n_g) .* 0.9 .+ 1)
    a = exp.(alpha_mu[1] .+ alpha_mu[2] .* w_lvl) .* randn(rng, n_id)
    b = exp.(alpha_sg[1] .+ alpha_sg[2] .* z_lvl) .* randn(rng, n_g)
    x = randn(rng, n)
    lin = (fam === :gaussian ? 0.2 : 0.8) .+ 0.4 .* x .+ a[gi]
    ls = (fam === :gaussian ? log(0.5) : -0.7) .+ b[gs]
    y = map(1:n) do i
        k = exp(-2ls[i])
        fam === :gaussian ? lin[i] + exp(ls[i]) * randn(rng) :
        fam === :nb2 ? Float64(rand(rng, NegativeBinomial(k, k / (k + exp(lin[i]))))) :
                       rand(rng, Distributions.Gamma(k, exp(lin[i]) / k))
    end
    id = [string("i", lpad(string(g), 2, '0')) for g in gi]
    g = same ? id : [string("g", lpad(string(j), 2, '0')) for j in gs]
    data = (; y, x, id, g, w = w_lvl[gi], z = z_lvl[gs])
    return (; data, gi, gs, w_lvl, z_lvl, same)
end

# Per-observation log density, written from Distributions only.
_jsw_logpdf(fam, y, lin, ls) = begin
    k = exp(-2ls)
    fam === :gaussian ? logpdf(Normal(lin, exp(ls)), y) :
    fam === :nb2 ? logpdf(NegativeBinomial(k, k / (k + exp(lin))), y) :
                   logpdf(Distributions.Gamma(k, exp(lin) / k), y)
end

function _jsw_oracle(fam, y, mu0, s0, gi, gs, sd_a, sd_b; same = false)
    na, nb = length(sd_a), length(sd_b)
    m = na + nb
    negjoint(u) = begin
        a = u[1:na]; b = u[na+1:m]
        s = zero(eltype(u))
        for i in eachindex(y)
            s -= _jsw_logpdf(fam, y[i], mu0[i] + a[gi[i]], s0 + b[gs[i]])
        end
        s - sum(logpdf.(Normal.(0, sd_a), a)) - sum(logpdf.(Normal.(0, sd_b), b))
    end
    u = zeros(m)
    for it in 1:200
        g = ForwardDiff.gradient(negjoint, u)
        maximum(abs, g) < 1e-11 && break
        H = ForwardDiff.hessian(negjoint, u)
        C = cholesky(Symmetric(H); check = false)
        lam = 1e-3
        while !issuccess(C)
            C = cholesky(Symmetric(H + lam * I); check = false); lam *= 10
        end
        step = C \ g
        t = 1.0; f0 = negjoint(u)
        while t > 1e-12 && negjoint(u .- t .* step) > f0 - 1e-4 * t * dot(g, step)
            t /= 2
        end
        u .-= (t > 1e-12 ? t : 1.0) .* step
    end
    H = ForwardDiff.hessian(negjoint, u)
    ll = -negjoint(u) + 0.5 * m * log(2π) - 0.5 * logdet(cholesky(Symmetric(H)))
    return (; loglik = ll, max_grad = maximum(abs, ForwardDiff.gradient(negjoint, u)),
            min_eig = minimum(eigvals(Symmetric(H))))
end

function _jsw_oracle_from_fit(fit, fx, fam)
    d = fx.data
    ks = [first(p) for p in fit.blocks if startswith(String(first(p)), "sdtarget_")]
    @assert length(ks) == 2
    # the mean-side block is the one whose identity contains "|mu|"
    key_mu = only(filter(k -> occursin(bytes2hex(codeunits("|mu|")), String(k)), ks))
    key_sg = only(filter(k -> k !== key_mu, ks))
    am = coef(fit, key_mu); asg = coef(fit, key_sg)
    sd_a = exp.(am[1] .+ am[2] .* fx.w_lvl)
    sd_b = exp.(asg[1] .+ asg[2] .* fx.z_lvl)
    X = hcat(ones(length(d.y)), d.x)
    return _jsw_oracle(fam, d.y, X * coef(fit, :mu), coef(fit, :sigma)[1], fx.gi, fx.gs, sd_a, sd_b)
end

_jsw_formula(same) = same ?
    bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | id)), @formula(sd(id) ~ w),
       @formula(sd(id, sigma) ~ z)) :
    bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g)), @formula(sd(id) ~ w),
       @formula(sd(g, sigma) ~ z))

const _JSW_FAMS = Dict(:gaussian => Gaussian(), :nb2 => NegBinomial2(), :gamma => DRModels.Gamma())

@testset "joint sigma wave 2: per-observation kernels match ForwardDiff" begin
    for (fam, y) in ((:gaussian, 0.7), (:nb2, 3.0), (:nb2, 0.0), (:gamma, 1.3))
        for (lin, eta) in ((0.3, -0.4), (-1.1, 0.6), (1.5, -1.2))
            k = DRModels._gsl_kernel(_JSW_FAMS[fam], y, lin, eta)
            nf(v) = -_jsw_logpdf(fam, y, v[1], v[2])
            gr = ForwardDiff.gradient(nf, [lin, eta]); H = ForwardDiff.hessian(nf, [lin, eta])
            @test k[1] ≈ nf([lin, eta]) rtol = 1e-10
            @test collect(k[2:3]) ≈ gr rtol = 1e-8 atol = 1e-12
            @test collect(k[4:6]) ≈ [H[1, 1], H[1, 2], H[2, 2]] rtol = 1e-8 atol = 1e-12
        end
    end
end

for (fam, same, seed) in ((:nb2, false, 2026110701), (:nb2, true, 2026110702),
                          (:gamma, false, 2026110703), (:gamma, true, 2026110704),
                          (:gaussian, true, 2026110705))
    @testset "joint sigma wave 2: $fam $(same ? "same group" : "crossed groups") matches dense Laplace oracle" begin
        fx = _jsw_fixture(fam; seed, same)
        fit = drm(_jsw_formula(same), _JSW_FAMS[fam]; data = fx.data)
        @test fit.converged
        @test all(isfinite, stderror(fit))
        o = _jsw_oracle_from_fit(fit, fx, fam)
        @test o.max_grad < 1e-6
        @test o.min_eig > 0
        @test abs(loglik(fit) - o.loglik) < 1e-5
    end
end

@testset "joint sigma wave 2: shapes native drmTMB refuses stay refused" begin
    fx = _jsw_fixture(:nb2; seed = 2026110701)
    d = fx.data
    for fam in (NegBinomial2(), DRModels.Gamma())
        dd = fam isa NegBinomial2 ? d : merge(d, (; y = d.y .+ 0.5))
        # no SD formula on either side / only one side
        @test_throws Exception drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g))), fam; data = dd)
        @test_throws Exception drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g)),
                                      @formula(sd(id) ~ w)), fam; data = dd)
        @test_throws Exception drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g)),
                                      @formula(sd(g, sigma) ~ z)), fam; data = dd)
        # sigma random slope
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (0 + x | g)),
                                          @formula(sd(id) ~ w), @formula(sd(g, sigma, coef(x)) ~ z)), fam; data = dd)
        # mean random slope
        @test_throws ArgumentError drm(bf(@formula(y ~ x + (0 + x | id)), @formula(sigma ~ 1 + (1 | g)),
                                          @formula(sd(id, coef(x)) ~ w), @formula(sd(g, sigma) ~ z)), fam; data = dd)
    end
    # SD predictor varying within group
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | id)), @formula(sigma ~ 1 + (1 | g)),
                                      @formula(sd(id) ~ x), @formula(sd(g, sigma) ~ z)), NegBinomial2(); data = d)
end

# Gaussian correlated mean + sigma intercepts on one group, `(1 | p | id)` in both
# (native drmTMB fits it; no SD formula). Oracle: per-group 2-D Laplace over raw
# (b_g, delta_g) ~ N(0, Sigma) with Sigma from the fitted SDs and rho = tanh(eta),
# ForwardDiff derivatives, own Newton. Same tolerances as above.
function _jsw_corr_fixture(; seed = 2026110706, G = 40, n = 800, rho = 0.5)
    rng = MersenneTwister(seed)
    gi = vcat(1:G, rand(rng, 1:G, n - G))
    S = [0.6^2 rho*0.6*0.3; rho*0.6*0.3 0.3^2]
    L = cholesky(Symmetric(S)).L
    u = L * randn(rng, 2, G)
    x = randn(rng, n)
    y = 0.2 .+ 0.5 .* x .+ u[1, gi] .+ exp.(-0.7 .+ u[2, gi]) .* randn(rng, n)
    return (; data = (; y, x, id = string.("g", gi)), gi, G)
end

@testset "joint sigma wave 2: Gaussian correlated (1 | p | id) mu + sigma matches oracle" begin
    fx = _jsw_corr_fixture()
    fit = drm(bf(@formula(y ~ x + (1 | p | id)), @formula(sigma ~ 1 + (1 | p | id))), Gaussian();
              data = fx.data)
    @test fit.converged
    @test all(isfinite, stderror(fit))
    bm = coef(fit, :mu); s0 = coef(fit, :sigma)[1]; lsd = coef(fit, :resd); rho = tanh(coef(fit, :cor)[1])
    @test -1 < rho < 1
    S = [exp(2lsd[1]) rho*exp(lsd[1] + lsd[2]); rho*exp(lsd[1] + lsd[2]) exp(2lsd[2])]
    Sinv = inv(S); ldS = logdet(S)
    # first-seen group order, as DRModels' _group_index; the oracle does not care
    labs = unique(fx.data.id); gidx = [findfirst(==(l), labs) for l in fx.data.id]
    mu = bm[1] .+ bm[2] .* fx.data.x
    total = 0.0; worst_g = 0.0; min_eig = Inf
    for g in 1:length(labs)
        idx = findall(==(g), gidx)
        nj(u) = -sum(logpdf(Normal(mu[i] + u[1], exp(s0 + u[2])), fx.data.y[i]) for i in idx) +
                0.5 * dot(u, Sinv * u) + 0.5 * ldS + log(2π)
        u = zeros(2)
        for _ in 1:100
            gr = ForwardDiff.gradient(nj, u); maximum(abs, gr) < 1e-12 && break
            H = ForwardDiff.hessian(nj, u)
            C = cholesky(Symmetric(H); check = false); lam = 1e-3
            while !issuccess(C); C = cholesky(Symmetric(H + lam * I); check = false); lam *= 10; end
            st = C \ gr; t = 1.0; f0 = nj(u)
            while t > 1e-12 && nj(u .- t .* st) > f0 - 1e-4 * t * dot(gr, st); t /= 2; end
            u .-= (t > 1e-12 ? t : 1.0) .* st
        end
        H = ForwardDiff.hessian(nj, u)
        total += -nj(u) + log(2π) - 0.5 * logdet(H)
        worst_g = max(worst_g, maximum(abs, ForwardDiff.gradient(nj, u)))
        min_eig = min(min_eig, minimum(eigvals(Symmetric(H))))
    end
    @test worst_g < 1e-6
    @test min_eig > 0
    @test abs(loglik(fit) - total) < 1e-5
    # an SD formula with the labelled pair is refused, as in drmTMB
    @test_throws Exception drm(bf(@formula(y ~ x + (1 | p | id)), @formula(sigma ~ 1 + (1 | p | id)),
                                  @formula(sd(id) ~ 1), @formula(sd(id, sigma) ~ 1)), Gaussian(); data = fx.data)
    # label mismatch
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | p | id)), @formula(sigma ~ 1 + (1 | q | id))),
                                   Gaussian(); data = fx.data)
end
