# Arc 2 — REML for the Gaussian location-scale model with phylo() on sigma is ONE
# joint Laplace approximation over (phylo effects, β_μ, β_σ), the restricted
# likelihood native drmTMB maximises (`REML = TRUE` puts beta_mu and beta_sigma
# in TMB's random vector when sigma carries a variance component).
#
#   1. Relationship (D-277): `_glsp_joint_reml_nll` equals a DENSE oracle — the
#      joint mode of jn(a, β) found by a plain ForwardDiff Newton, and the Laplace
#      value from the full dense Hessian — on all three Λ blocks.
#   2. Relationship: the coupled block with L21 = 0 IS the separate block.
#   3. Same target: drm(method = :REML) reproduces native drmTMB (engine = "tmb")
#      df, logLik (<= 1e-6) and estimates (<= 1e-5 rel) on the committed fixtures;
#      the numbers come from docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/
#      native-fit.R, committed next to them.
#   4. Guard (D-273): the neighbouring routes that already matched native — ML on
#      both shapes and the mean-only phylo REML cell — still do.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays, ForwardDiff

const _D = DRModels
const _ARC2_EV = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                          "arc2-gaussian-sigma-phylo-reml")

# Dense oracle: joint mode of jn(z), z = [a; β], by undamped-then-damped Newton on
# ForwardDiff derivatives; Laplace value with the flat-β constant.
function _arc2_dense_reml(kind, y, Xμ, Xψ, gidx, P, Zη, Zψ, β0)
    pμ = size(Xμ, 2); p = pμ + size(Xψ, 2); na = size(P, 1)
    f(z) = _D._ls_joint(kind, y, Xμ * z[na+1:na+pμ], Xψ * z[na+pμ+1:na+p], gidx,
                        z[1:na], P, Zη, Zψ)
    z = vcat(zeros(na), β0)
    for _ in 1:200
        g = ForwardDiff.gradient(f, z)
        norm(g) < 1e-11 && break
        H = Symmetric(ForwardDiff.hessian(f, z))
        λ = 0.0                                   # Levenberg damping until H + λI is PD
        while !isposdef(H + λ * I); λ = λ == 0.0 ? 1e-6 : 10λ; end
        d = (H + λ * I) \ g; α = 1.0
        while f(z .- α .* d) > f(z) && α > 1e-8; α /= 2; end
        z = z .- α .* d
    end
    H = ForwardDiff.hessian(f, z)
    nll = f(z) + 0.5 * logdet(Symmetric(H)) - 0.5 * logdet(Symmetric(Matrix(P))) -
          0.5 * p * log(2π)
    return nll, z[na+1:end]
end

@testset "Arc 2 joint-Laplace REML: sparse value == dense oracle (3 blocks)" begin
    Random.seed!(20260924)
    ntip = 8; m = 4; n = ntip * m
    phy = random_balanced_tree(ntip; branch_length = 0.25)
    species = repeat(1:ntip, inner = m)
    x = randn(n)
    y = 0.2 .+ 0.5 .* x .+ 0.4 .* randn(ntip)[species] .+ exp.(-0.3 .+ 0.3 .* randn(ntip)[species]) .* randn(n)
    Xμ = hcat(ones(n), x); Xψ = hcat(ones(n), randn(n))     # pψ = 2: a β_σ slope too
    Q, gidx, G = _D._locscale_phylo_setup(phy, species)
    kind = Val(:gaussian_mean)
    Zη0, Zψ0 = _D._glsp_asym_loadings(n)
    Zη, Zψ = _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n)
    cases = [
        ("asymmetric", _D._glsp_asym_Λ(log(0.4)), Zη0, Zψ0),
        ("separate",   _D._glsp_sep_Λ([log(0.5), log(0.35)]), Zη, Zψ),
        ("coupled",    _D._glsp_coupled_Λ([log(0.5), -0.2, log(0.3)]), Zη, Zψ),
    ]
    for (label, Λ, Ze, Zp) in cases
        P = _D.prior_precision(Q, _D._ls_inv2x2(Λ))
        nll, β̂, _, S, ok = _D._glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P, Ze, Zp,
                                                   zeros(4), zeros(2G))
        nll_d, β_d = _arc2_dense_reml(kind, y, Xμ, Xψ, gidx, P, Ze, Zp, zeros(4))
        @testset "$label" begin
            @test ok
            @test nll ≈ nll_d atol = 1e-8
            @test β̂ ≈ β_d atol = 1e-7
            @test isposdef(Symmetric(S))
        end
    end
end

@testset "Arc 2 joint-Laplace REML: coupled with L21 = 0 is the separate block" begin
    Random.seed!(7)
    ntip = 12; m = 3; n = ntip * m
    phy = random_balanced_tree(ntip; branch_length = 0.3)
    species = repeat(1:ntip, inner = m)
    x = randn(n); y = 1 .+ 0.3 .* x .+ randn(n)
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    Q, gidx, G = _D._locscale_phylo_setup(phy, species)
    Zη, Zψ = _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n)
    kind = Val(:gaussian_mean)
    P_sep = _D.prior_precision(Q, _D._ls_inv2x2(_D._glsp_sep_Λ([log(0.45), log(0.25)])))
    P_cou = _D.prior_precision(Q, _D._ls_inv2x2(_D._glsp_coupled_Λ([log(0.45), 0.0, log(0.25)])))
    r_sep = _D._glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P_sep, Zη, Zψ, zeros(3), zeros(2G))
    r_cou = _D._glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P_cou, Zη, Zψ, zeros(3), zeros(2G))
    @test r_sep[5] && r_cou[5]
    @test r_sep[1] ≈ r_cou[1] atol = 1e-10
    @test r_sep[2] ≈ r_cou[2] atol = 1e-10
end

# ---- same target: native drmTMB numbers from the committed R script ----------
function _arc2_fixture(fx)
    lines = readlines(joinpath(_ARC2_EV, "fixture-$fx.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    data = (y  = [parse(Float64, r[1]) for r in rows],
            x  = [parse(Float64, r[2]) for r in rows],
            sp = [String(strip(r[3], '"')) for r in rows])
    return data, String(strip(read(joinpath(_ARC2_EV, "fixture-$fx.nwk"), String)))
end
function _arc2_native()
    lines = readlines(joinpath(_ARC2_EV, "native.tsv"))
    hdr = split(lines[1], '\t')
    Dict((r["fixture"], r["shape"], r["estimator"]) => r
         for r in (Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]))
end
_arc2_num(s) = s == "NA" ? NaN : parse(Float64, s)

const _ARC2_FORMS = Dict(
    "mu_only"    => (bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ 1)), false),
    "sigma_only" => (bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | sp))), false),
    "mu_sigma"   => (bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ phylo(1 | sp))), true),
)

function _arc2_check(native, fx, shape, estimator; se = false)
    data, nwk = _arc2_fixture(fx)
    form, coupled = _ARC2_FORMS[shape]
    fit = drm(form, Gaussian(); data = data, tree = nwk, method = Symbol(estimator),
              phylo_coupled = coupled, g_tol = 1e-8)
    nr = native[(fx, shape, estimator)]
    @test is_converged(fit)
    @test estimation_method(fit) == Symbol(estimator)
    @test dof(fit) == parse(Int, nr["df"])
    @test loglik(fit) ≈ _arc2_num(nr["logLik"]) atol = 1e-6
    β = vcat(coef(fit, :mu), coef(fit, :sigma))
    @test β ≈ _arc2_num.([nr["mu_intercept"], nr["mu_x"], nr["sigma_intercept"]]) rtol = 1e-5
    if shape == "sigma_only"
        @test exp(coef(fit, :resd_sigma)[1]) ≈ _arc2_num(nr["sd_sigma"]) rtol = 1e-5
    elseif shape == "mu_sigma"
        @test fit.scales[:lambda_sd_mu][1] ≈ _arc2_num(nr["sd_mu"]) rtol = 1e-5
        @test fit.scales[:lambda_sd_sigma][1] ≈ _arc2_num(nr["sd_sigma"]) rtol = 1e-5
        @test fit.scales[:lambda_cor][1] ≈ _arc2_num(nr["cor"]) atol = 1e-5
    else
        @test re_sd(fit)[:sp] ≈ _arc2_num(nr["sd_mu"]) rtol = 1e-5
    end
    if se   # Wald SEs of the fixed effects (sdreport rule for a REML-integrated β)
        @test stderror(fit)[1:3] ≈ _arc2_num.([nr["se_mu_intercept"], nr["se_mu_x"],
                                               nr["se_sigma_intercept"]]) rtol = 1e-3
    end
    return fit
end

@testset "Arc 2 same target: σ-phylo REML == native drmTMB REML" begin
    native = _arc2_native()
    # F1 is the Arc 1 probe fixture (sigma-only: −143.3750 native vs −146.1213 before).
    @testset "F1 sigma_only REML" begin _arc2_check(native, "F1", "sigma_only", "REML"; se = true) end
    @testset "F1 mu_sigma REML (coupled)" begin _arc2_check(native, "F1", "mu_sigma", "REML") end
    @testset "F2 sigma_only REML" begin _arc2_check(native, "F2", "sigma_only", "REML"; se = true) end
    @testset "F2 mu_sigma REML (coupled)" begin _arc2_check(native, "F2", "mu_sigma", "REML"; se = true) end
end

@testset "Arc 2 guard: neighbouring routes still match native" begin
    native = _arc2_native()
    @testset "F1 sigma_only ML" begin _arc2_check(native, "F1", "sigma_only", "ML") end
    @testset "F2 mu_sigma ML (coupled)" begin _arc2_check(native, "F2", "mu_sigma", "ML") end
    # The mean-only phylo REML cell: a different route (sparse location-only spine).
    @testset "F2 mu_only REML" begin _arc2_check(native, "F2", "mu_only", "REML") end
end

@testset "Arc 2 separate block: REML is the same joint-Laplace quantity" begin
    data, nwk = _arc2_fixture("F2")
    fit = drm(_ARC2_FORMS["mu_sigma"][1], Gaussian(); data = data, tree = nwk, method = :REML)
    @test estimation_method(fit) == :REML
    @test is_converged(fit)
    @test dof(fit) == 5
    # Reported restricted logLik == −nll_R at the reported variance parameters.
    n = length(data.y)
    Xμ = hcat(ones(n), data.x); Xψ = ones(n, 1)
    phy = augmented_phy(nwk)
    Q, gidx, G = _D._locscale_phylo_setup(phy, data.sp)
    P = _D.prior_precision(Q, _D._ls_inv2x2(_D._glsp_sep_Λ(fit.theta[4:5])))
    r = _D._glsp_joint_reml_nll(Val(:gaussian_mean), data.y, Xμ, Xψ, gidx, G, P,
                                _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n),
                                zeros(3), zeros(2G))
    @test loglik(fit) ≈ -r[1] atol = 1e-8
    @test fit.theta[1:3] ≈ r[2] atol = 1e-6
    # Constraining the correlation to 0 cannot raise the restricted likelihood.
    fit_c = drm(_ARC2_FORMS["mu_sigma"][1], Gaussian(); data = data, tree = nwk,
                method = :REML, phylo_coupled = true)
    @test loglik(fit_c) >= loglik(fit) - 1e-8
end
