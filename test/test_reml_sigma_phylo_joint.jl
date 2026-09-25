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
#   5. Speed: the coupled REML stage reaches native's optimum on H2 and G1 within
#      a time bound. Cells with a slow coupled ML seed need DRM_SLOW_TESTS=1.
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

# The correlation-bound candidate evaluates the coupled block in whitened latent
# coordinates, a = L·u with u ~ N(0, Q⁻¹ ⊗ I) and loadings (Zη·L, Zψ·L). The Laplace
# approximation is invariant to that linear change of variables, so at a
# well-conditioned Λ the two forms must agree to rounding.
@testset "Arc 2 joint-Laplace REML: whitened coupled form == direct form" begin
    Random.seed!(11)
    ntip = 10; m = 4; n = ntip * m
    phy = random_balanced_tree(ntip; branch_length = 0.3)
    species = repeat(1:ntip, inner = m)
    x = randn(n); y = 0.4 .+ 0.5 .* x .+ 0.5 .* randn(ntip)[species] .+ randn(n)
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    Q, gidx, G = _D._locscale_phylo_setup(phy, species)
    Zη, Zψ = _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n)
    kind = Val(:gaussian_mean)
    for v in ([log(0.5), 0.3, log(0.2)], [log(0.4), -0.35, log(0.05)])
        L = [exp(v[1]) 0.0; v[2] exp(v[3])]
        P = _D.prior_precision(Q, _D._ls_inv2x2(_D._glsp_coupled_Λ(v)))
        P_I = _D.prior_precision(Q, Matrix(1.0I, 2, 2))
        r_d = _D._glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P, Zη, Zψ, zeros(3), zeros(2G))
        r_w = _D._glsp_joint_reml_nll(kind, y, Xμ, Xψ, gidx, G, P_I, Zη * L, Zψ * L,
                                      zeros(3), zeros(2G))
        @test r_d[5] && r_w[5]
        @test r_w[1] ≈ r_d[1] atol = 1e-8
        @test r_w[2] ≈ r_d[2] atol = 1e-7
    end
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

# Slow cells: every end-to-end coupled REML fit first runs the coupled ML fit for its
# start, and on the F1, G1 and G2 fixtures that ML fit takes 11-72 s locally and
# roughly ten times that on CI (DRModels issue #818 tracks the ML route). Those
# cells run only with DRM_SLOW_TESTS=1, as in test_locscale_profile.jl. The default
# run keeps a coupled REML native match end to end (F2, with Wald SEs) and checks
# the coupled REML stage itself on H2 and G1 from a cached ML start (below).
const _ARC2_SLOW = get(ENV, "DRM_SLOW_TESTS", "0") == "1"

@testset "Arc 2 same target: σ-phylo REML == native drmTMB REML" begin
    native = _arc2_native()
    # F1 is the Arc 1 probe fixture (sigma-only: −143.3750 native vs −146.1213 before).
    @testset "F1 sigma_only REML" begin _arc2_check(native, "F1", "sigma_only", "REML"; se = true) end
    if _ARC2_SLOW
        @testset "F1 mu_sigma REML (coupled)" begin _arc2_check(native, "F1", "mu_sigma", "REML") end
    end
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

# ---- correlation bound: coupled REML where native sits on |cor| = 0.999999 ------
# Two fixtures (native-fit-boundary.R, unequal species sizes) where native drmTMB's
# coupled REML optimum lies on its correlation bound, rho = 0.999999·tanh(eta).
# Julia used to stop at a worse local optimum: G1 at sd_mu → 0, cor = −0.45, logLik
# 0.97 below native; G2 7.6e-4 below. G2's tree is 7.22 tall (not unit): its phylo
# SDs are on the raw branch-length scale, native's on the unit-height scale, so the
# SDs are compared after that rescaling (the logLik does not depend on it).
function _arc2_boundary_fixture(fx)
    lines = readlines(joinpath(_ARC2_EV, "fixture-$fx.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    data = (y  = [parse(Float64, r[1]) for r in rows],
            x  = [parse(Float64, r[2]) for r in rows],
            z  = [parse(Float64, r[3]) for r in rows],
            sp = [String(strip(r[4], '"')) for r in rows])
    return data, String(strip(read(joinpath(_ARC2_EV, "fixture-$fx.nwk"), String)))
end

const _ARC2_BOUNDARY_FIXTURES = [("G1", @formula(sigma ~ phylo(1 | sp))),
                                 ("G2", @formula(sigma ~ z + phylo(1 | sp)))]

function _arc2_native_tsv(file)
    lines = readlines(joinpath(_ARC2_EV, file))
    hdr = split(lines[1], '\t')
    Dict((r["fixture"], r["shape"], r["estimator"]) => r
         for r in (Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]))
end

if _ARC2_SLOW
@testset "Arc 2 correlation bound: coupled REML == native drmTMB on its bound" begin
    native = _arc2_native_tsv("native-boundary.tsv")
    for (fx, sform) in _ARC2_BOUNDARY_FIXTURES
        data, nwk = _arc2_boundary_fixture(fx)
        nr = native[(fx, "mu_sigma", "REML")]
        fit = drm(bf(@formula(y ~ x + phylo(1 | sp)), sform), Gaussian(); data = data,
                  tree = nwk, method = :REML, phylo_coupled = true, g_tol = 1e-8)
        @testset "$fx" begin
            @test is_converged(fit)
            @test dof(fit) == parse(Int, nr["df"])
            ll_n = _arc2_num(nr["logLik"])
            @test abs(loglik(fit) - ll_n) <= 1e-6
            β_n = _arc2_num.([nr["mu_intercept"], nr["mu_x"], nr["sigma_intercept"]])
            fx == "G2" && push!(β_n, _arc2_num(nr["sigma_z"]))
            @test vcat(coef(fit, :mu), coef(fit, :sigma)) ≈ β_n rtol = 1e-5
            # Julia's SDs are on the raw branch-length scale (see above).
            h = _D.phylo_tree_height(augmented_phy(nwk))
            @test fit.scales[:lambda_sd_mu][1] * sqrt(h) ≈ _arc2_num(nr["sd_mu"]) rtol = 1e-4
            @test fit.scales[:lambda_sd_sigma][1] * sqrt(h) ≈ _arc2_num(nr["sd_sigma"]) rtol = 1e-4
            @test fit.scales[:lambda_cor][1] ≈ _D._GLSP_COR_CAP atol = 1e-9
            @test _arc2_num(nr["cor"]) > 0.99999          # native: on (or at) its bound
            @test all(isnan, vcov(fit)[end, :])           # boundary fit: no Wald covariance
        end
    end
end
else
    @info "Arc 2 end-to-end correlation-bound cells skipped (coupled ML seeds, ~minutes); set DRM_SLOW_TESTS=1 to run"
end

# ---- speed: the coupled REML stage from a cached ML start --------------------
# The coupled REML fit re-solves the joint mode thousands of times, each from the
# neighbouring one. A warm inner solve used to fail at the exact mode on a few-ULP
# rise of the objective and spin for 1-2 s before a cold solve rescued it; on H2
# (strong negative phylo correlation, 1-12 rows per species, tree height 3.58) the
# REML stage took about 29 minutes against native's 1-2 s (native-fit-speed.R). It
# now takes a few seconds. `_glsp_joint_reml_fit` is called as the coupled route
# calls it, from the Julia coupled ML estimate (drm(..., method = :ML,
# phylo_coupled = true, g_tol = 1e-8), recorded below in the route's internal order
# [β; logL11, L21, logL22]), so the ML seed's own cost stays out of this check.
# The elapsed-time bound is loose (CI runs several times slower than a laptop) but
# far below the old cost: it fails if the slowdown returns.
const _ARC2_ML_START = Dict(
    "H2" => [1.2882697910798269, 0.6077084992687934, -1.2916003662446642, 0.4860840655836412,
             -1.0231783198724216, -0.22161799723466435, -1.8624001464918072],
    "G1" => [-0.16511401166476766, 0.5865758079562735, 0.3268961769415294,
             -9.315395271195959, -0.657631802472501, -1.3033928365601821])

function _arc2_coupled_reml_stage(fx, has_z)
    data, nwk = _arc2_boundary_fixture(fx)
    n = length(data.y)
    Xμ = hcat(ones(n), data.x)
    Xψ = has_z ? hcat(ones(n), data.z) : ones(n, 1)
    p = size(Xμ, 2) + size(Xψ, 2)
    phy = augmented_phy(nwk)
    Q, gidx, G = _D._locscale_phylo_setup(phy, data.sp)
    θml = _ARC2_ML_START[fx]
    starts = [[log(0.3), 0.0, log(0.3)], _D._glsp_reml_start(θml[p+1:end], (1, 3))]
    t = @elapsed rf = _D._glsp_joint_reml_fit(Val(:gaussian_mean), data.y, Xμ, Xψ, gidx, G, Q,
                                              _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n),
                                              _D._glsp_coupled_Λ, starts, θml[1:p];
                                              se = true, logsd_idx = (1, 3), shrink_idx = (2,),
                                              cor_edge = true)
    Λ = _D._glsp_coupled_Λ(rf.v)
    sd = sqrt.([Λ[1, 1], Λ[2, 2]])
    return rf, t, sd .* sqrt(_D.phylo_tree_height(phy)), Λ[1, 2] / prod(sd)
end

@testset "Arc 2 speed: coupled REML stage matches native in seconds" begin
    native = merge(_arc2_native_tsv("native-boundary.tsv"), _arc2_native_tsv("native-speed.tsv"))
    for (fx, has_z) in (("H2", true), ("G1", false))
        nr = native[(fx, "mu_sigma", "REML")]
        rf, t, sd_unit, cor = _arc2_coupled_reml_stage(fx, has_z)
        @info "Arc 2 coupled REML stage (cached ML start)" fixture = fx seconds = round(t, digits = 2)
        @testset "$fx" begin
            @test rf.converged
            @test abs(-rf.reml_nll - _arc2_num(nr["logLik"])) <= 1e-6
            β_n = _arc2_num.([nr["mu_intercept"], nr["mu_x"], nr["sigma_intercept"]])
            has_z && push!(β_n, _arc2_num(nr["sigma_z"]))
            @test rf.β ≈ β_n rtol = 1e-5
            @test sd_unit ≈ _arc2_num.([nr["sd_mu"], nr["sd_sigma"]]) rtol = 1e-4
            if fx == "G1"
                @test cor ≈ _D._GLSP_COR_CAP atol = 1e-9     # native's optimum is on its bound
            else
                @test cor ≈ _arc2_num(nr["cor"]) atol = 1e-5  # H2: interior, cor = -0.784
            end
            @test t < 120.0
        end
    end
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

# ---- variance boundary: a zero-signal REML fit converges on the plateau ------
# With no scale-phylogeny signal the restricted NLL flattens as the log-SD runs to
# −∞; its gradient falls like SD², below the FD gradient's rounding noise, so the
# outer Newton used to creep until the iteration cap and report non-convergence
# (9 of 15 zero-signal seeds). It must now report convergence AND sit on the
# plateau's supremum, the restricted logLik at SD → 0.
function _arc2_zero_signal(seed)
    Random.seed!(seed)
    ntip = 20; m = 4; n = ntip * m
    phy = random_balanced_tree(ntip; branch_length = 0.25)
    sp = repeat(1:ntip, inner = m)
    x = randn(n); y = 0.5 .+ 0.3 .* x .+ randn(n)
    return (y = y, x = x, sp = sp), phy
end
function _arc2_nllR(data, phy, Λ, Zη, Zψ)
    n = length(data.y)
    Q, gidx, G = _D._locscale_phylo_setup(phy, data.sp)
    P = _D.prior_precision(Q, _D._ls_inv2x2(Λ))
    r = _D._glsp_joint_reml_nll(Val(:gaussian_mean), data.y, hcat(ones(n), data.x),
                                ones(n, 1), gidx, G, P, Zη, Zψ, zeros(3), zeros(2G))
    @assert r[5]
    return r[1]
end

@testset "Arc 2 boundary: zero-signal REML converges on the plateau supremum" begin
    # 1011 was non-converged before the plateau rule; 1019 (aarch64) and 1011
    # (Julia 1.10 x86-64) were still non-converged under its first version.
    for seed in (1011, 1002, 1019)
        data, phy = _arc2_zero_signal(seed)
        n = length(data.y)
        fit = drm(_ARC2_FORMS["sigma_only"][1], Gaussian(); data = data, tree = phy,
                  method = :REML)
        @test is_converged(fit)
        @test exp(coef(fit, :resd_sigma)[1]) < 1e-3
        Zη, Zψ = _D._glsp_asym_loadings(n)
        sup = -_arc2_nllR(data, phy, _D._glsp_asym_Λ(-20.0), Zη, Zψ)
        @test loglik(fit) ≈ sup atol = 1e-6
        @test loglik(fit) <= sup + 1e-8
        @test all(isnan, vcov(fit)[end, :])  # no Wald curvature on the boundary
    end
    # Coupled block, no phylo signal on either axis (both SDs and L21 → 0).
    data, phy = _arc2_zero_signal(3)
    n = length(data.y)
    fit = drm(_ARC2_FORMS["mu_sigma"][1], Gaussian(); data = data, tree = phy,
              method = :REML, phylo_coupled = true)
    @test is_converged(fit)
    @test fit.scales[:lambda_sd_mu][1] < 1e-3 && fit.scales[:lambda_sd_sigma][1] < 1e-3
    sup = -_arc2_nllR(data, phy, _D._glsp_coupled_Λ([-20.0, 0.0, -20.0]),
                      _D._ls_canonical_Zeta(n), _D._ls_canonical_Zpsi(n))
    @test loglik(fit) ≈ sup atol = 1e-6
end

# ---- profile_ci under REML profiles the RESTRICTED surface -------------------
# Before, `profile_ci = true` under REML profiled the ML NLL from the REML point
# (which is not the ML minimum): neither an ML nor a REML interval.
@testset "Arc 2 profile_ci under REML: restricted-likelihood profile" begin
    thr = 0.5 * 3.841458820694124     # χ²₁(0.95)/2
    # Scale-only block: β is integrated out and the SD is the only variance
    # parameter, so the profile IS nll_R; each finite endpoint sits at the threshold.
    data, nwk = _arc2_fixture("F1")
    n = length(data.y)
    fit = drm(_ARC2_FORMS["sigma_only"][1], Gaussian(); data = data, tree = nwk,
              method = :REML, profile_ci = true)
    lo, hi = fit.scales[:profile_ci_sd_sigma]
    sd = exp(coef(fit, :resd_sigma)[1])
    @test 0 <= lo < sd < hi < Inf
    phy = augmented_phy(nwk)
    Zη, Zψ = _D._glsp_asym_loadings(n)
    nllR(s) = _arc2_nllR(data, phy, _D._glsp_asym_Λ(log(s)), Zη, Zψ)
    @test -nllR(sd) ≈ loglik(fit) atol = 1e-8
    @test nllR(hi) - nllR(sd) ≈ thr atol = 1e-4
    if lo > 0
        @test nllR(lo) - nllR(sd) ≈ thr atol = 1e-4
    else   # an honest [0, hi]: the restricted NLL never rises by thr as SD → 0
        @test nllR(sd * exp(-8)) - nllR(sd) < thr
    end
    # Guard: the ML route's profile is unchanged — still the ML surface, from the ML fit.
    fit_ml = drm(_ARC2_FORMS["sigma_only"][1], Gaussian(); data = data, tree = nwk,
                 method = :ML, profile_ci = true)
    @test fit_ml.scales[:profile_ci_sd_sigma] != fit.scales[:profile_ci_sd_sigma]
    # Separate block (small, for speed): each SD's interval brackets it, and at an
    # endpoint the restricted NLL with the OTHER SD held at its estimate is at least
    # the threshold (the profile re-optimises that SD, so it can only be lower).
    Random.seed!(21)
    ntip = 10; m = 4; ns = ntip * m
    phy_s = random_balanced_tree(ntip; branch_length = 0.25)
    sp = repeat(1:ntip, inner = m); xs = randn(ns)
    ys = 0.2 .+ 0.4 .* xs .+ 0.5 .* randn(ntip)[sp] .+ exp.(0.4 .* randn(ntip)[sp]) .* randn(ns)
    ds = (y = ys, x = xs, sp = sp)
    fs = drm(_ARC2_FORMS["mu_sigma"][1], Gaussian(); data = ds, tree = phy_s,
             method = :REML, profile_ci = true)
    v̂ = fs.theta[4:5]
    Zc = (_D._ls_canonical_Zeta(ns), _D._ls_canonical_Zpsi(ns))
    nllS(v) = _arc2_nllR(ds, phy_s, _D._glsp_sep_Λ(v), Zc...)
    base = nllS(v̂)
    for (key, j) in ((:profile_ci_sd_mu, 1), (:profile_ci_sd_sigma, 2))
        l, u = fs.scales[key]
        @test l < exp(v̂[j]) < u
        for e in (l, u)
            (0 < e < Inf) || continue
            v = copy(v̂); v[j] = log(e)
            @test nllS(v) - base >= thr - 1e-4
        end
    end
end
