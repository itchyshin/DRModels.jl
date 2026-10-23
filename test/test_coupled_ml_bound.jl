# Issue #818 — coupled sigma-phylo ML (`phylo(1 | g)` on mu and sigma,
# `phylo_coupled = true`) must reach native drmTMB's ML optimum when that optimum
# sits on native's correlation bound, rho = 0.999999·tanh(eta) (drmTMB.cpp).
#
# Before the fix the route stopped on the sd_mu -> 0 plateau (G1 logLik 1.016
# below native, G2 0.685 below) and reported converged = true. Two defects:
#   1. Near |cor| -> 1, P = Q ⊗ Λ⁻¹ reaches ~1e10, the inner mode's absolute 1e-9
#      stationarity bound sits below the joint gradient's rounding noise, and the
#      marginal NLL is Inf: at native's own G1 optimum the old objective could not
#      be evaluated at all.
#   2. No candidate on the bound itself.
# The objective is now evaluated with a noise-floor inner bound, and the bound is
# fitted as its own candidate in whitened coordinates (a = L·u), where the
# precision is Q ⊗ I and the value is the same Laplace approximation without the
# ill-conditioned prior quadratic.
#
# Fixtures and native numbers: docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/
# (native-fit-boundary.R -> native-boundary.tsv for G1/G2; native-fit.R ->
# native.tsv for F1). Integrator on both sides: Laplace (DRModels.jl sparse
# Laplace vs TMB's Laplace).
using DRModels
using Test, LinearAlgebra, SparseArrays

const _D818 = DRModels
const _EV818 = joinpath(@__DIR__, "..", "docs", "dev-log", "evidence",
                        "arc2-gaussian-sigma-phylo-reml")
const _SLOW818 = get(ENV, "DRM_SLOW_TESTS", "0") == "1"

function _fx818(fx)
    lines = readlines(joinpath(_EV818, "fixture-$fx.csv"))
    hdr = split(lines[1], ",")
    rows = [split(l, ",") for l in lines[2:end]]
    col(i) = [parse(Float64, r[i]) for r in rows]
    data = length(hdr) == 4 ?
        (y = col(1), x = col(2), z = col(3), sp = [String(strip(r[4], '"')) for r in rows]) :
        (y = col(1), x = col(2), sp = [String(strip(r[3], '"')) for r in rows])
    return data, String(strip(read(joinpath(_EV818, "fixture-$fx.nwk"), String)))
end

function _native818(file)
    lines = readlines(joinpath(_EV818, file))
    hdr = split(lines[1], '\t')
    Dict((r["fixture"], r["shape"], r["estimator"]) => r
         for r in (Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]))
end
_num818(s) = s == "NA" ? NaN : parse(Float64, s)

const _NATIVE818 = merge(_native818("native-boundary.tsv"), _native818("native.tsv"))
_sform818(fx) = fx == "G2" ? @formula(sigma ~ z + phylo(1 | sp)) : @formula(sigma ~ phylo(1 | sp))

# ---- the objective at native's optimum --------------------------------------
# Relationship: at native's G1 ML optimum (Λ from the reported SDs and cor, SDs
# on the unit-height tree), the direct-precision marginal NLL at the noise-floor
# inner bound and the whitened one agree, and both are finite. The old fixed
# 1e-9 inner bound returns Inf there (that is defect 1).
@testset "#818: coupled ML objective is finite at native's bound optimum" begin
    data, nwk = _fx818("G1"); n = length(data.y)
    nr = _NATIVE818[("G1", "mu_sigma", "ML")]
    Xμ = hcat(ones(n), data.x); Xψ = ones(n, 1)
    Q, gidx, G = _D818._locscale_phylo_setup(augmented_phy(nwk), data.sp)
    Zη, Zψ = _D818._ls_canonical_Zeta(n), _D818._ls_canonical_Zpsi(n)
    kind = Val(:gaussian_mean)
    sm, ss, c = _num818(nr["sd_mu"]), _num818(nr["sd_sigma"]), _num818(nr["cor"])
    Λ = [sm^2 c*sm*ss; c*sm*ss ss^2]
    η0 = Xμ * _num818.([nr["mu_intercept"], nr["mu_x"]])
    ψ0 = Xψ * [_num818(nr["sigma_intercept"])]
    P = _D818.prior_precision(Q, _D818._ls_inv2x2(Λ))
    old, _, ok_old = _D818._ls_marginal_nll(kind, data.y, η0, ψ0, gidx, G, P, Zη, Zψ)
    @test !ok_old && old == Inf                       # the pre-fix objective
    v_d, _, ok_d = _D818._glsp_ml_nll(kind, data.y, η0, ψ0, gidx, G, P, Zη, Zψ)
    L = Matrix(cholesky(Symmetric(Λ)).L)
    P_I = _D818.prior_precision(Q, Matrix(1.0I, 2, 2))
    v_w, _, ok_w = _D818._glsp_ml_nll(kind, data.y, η0, ψ0, gidx, G, P_I, Zη * L, Zψ * L)
    @test ok_d && ok_w
    @test v_w ≈ v_d atol = 1e-5                       # direct form carries eps·‖P‖ noise
    @test abs(-v_w - _num818(nr["logLik"])) <= 1e-6   # whitened == native at native's point
end

# ---- end to end: drm(method = :ML, phylo_coupled = true) --------------------
# G1: fails before the fix (logLik -331.94079, 1.016 below native).
# G2 (SLOW): native's optimum is at cor 0.99999894, just inside the bound; Julia's
# is on it. Native's own objective is noisy there (~5e-7 in eta, measured with
# drmTMB 0.7.1: fn at Julia's point is 3e-7 ABOVE native's reported optimum), so
# G2 is checked by logLik >= native - 1e-6 and betas at rtol 1e-4.
# F1 (SLOW): native reports non-convergence at cor = -0.99999793; Julia must be at
# least as high.
function _check818(fx; rtol_β = 1e-5, rtol_sd = 1e-4)
    data, nwk = _fx818(fx)
    nr = _NATIVE818[(fx, "mu_sigma", "ML")]
    fit = drm(bf(@formula(y ~ x + phylo(1 | sp)), _sform818(fx)), Gaussian(); data = data,
              tree = nwk, method = :ML, phylo_coupled = true, g_tol = 1e-8)
    ll_n = _num818(nr["logLik"])
    @testset "$fx" begin
        @test is_converged(fit)
        @test dof(fit) == parse(Int, nr["df"])
        @test loglik(fit) >= ll_n - 1e-6
        @test abs(loglik(fit) - ll_n) <= 1e-6
        β_n = _num818.([nr["mu_intercept"], nr["mu_x"], nr["sigma_intercept"]])
        fx == "G2" && push!(β_n, _num818(nr["sigma_z"]))
        @test vcat(coef(fit, :mu), coef(fit, :sigma)) ≈ β_n rtol = rtol_β
        h = _D818.phylo_tree_height(augmented_phy(nwk))   # Julia SDs: raw branch-length scale
        @test fit.scales[:lambda_sd_mu][1] * sqrt(h) ≈ _num818(nr["sd_mu"]) rtol = rtol_sd
        @test fit.scales[:lambda_sd_sigma][1] * sqrt(h) ≈ _num818(nr["sd_sigma"]) rtol = rtol_sd
        @test abs(fit.scales[:lambda_cor][1]) ≈ _D818._GLSP_COR_CAP atol = 1e-9
        @test sign(fit.scales[:lambda_cor][1]) == sign(_num818(nr["cor"]))
        @test all(isnan, vcov(fit)[end, :])               # on the bound: no Wald covariance
    end
    return fit
end

@testset "#818: coupled ML reaches native drmTMB's optimum on its correlation bound" begin
    _check818("G1")
    if _SLOW818
        _check818("G2"; rtol_β = 1e-4)
        _check818("F1"; rtol_β = 1e-2, rtol_sd = 1e-2)
    else
        @info "#818 G2/F1 end-to-end coupled ML cells skipped; set DRM_SLOW_TESTS=1 to run"
    end
end
