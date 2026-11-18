# Issue #952 — the Gaussian sigma-phylo location-scale route (`phylo(1 | g)` on
# sigma; separate, coupled and sigma-only blocks) must not depend on the tree height.
#
# On ade4::carni70 (ultrametric, height 53.8) the default separate fit of
# `lsize ~ 1 + lrange + phylo(1 | species)`, `sigma ~ 1 + phylo(1 | species)`
# returned logLik = -1e18 at its start values, and the coupled fit stopped at
# -119.531, 7.7 nats below drmTMB; on the same tree with every branch divided by
# 53.8 the separate fit reached drmTMB's optimum with the phylo correlation held
# at 0 (-111.94867803) and the coupled fit drmTMB's native optimum
# (-111.84106326). Cause: the route fitted on the raw branch
# lengths with absolute log-SD constants (start log 0.3, i.e. SD 0.3·√53.8 = 2.2
# on the unit-height scale, where the inner mode runs into the sigma -> 0
# collapse). The route now fits on the unit-height tree and maps the estimates
# back to the raw scale (`_glsp_tree_scale`, `_glsp_unscale_fit`).
#
# The issue's "0.107 below drmTMB" on the unit tree is a model difference, not an
# early stop: DRModels.jl's default for this formula is the SEPARATE block (no
# mu-sigma phylo correlation, 5 parameters), drmTMB fits the correlation (df = 6).
# drmTMB's own objective with the correlation held at 0 gives -111.94867803, the
# separate fit's value; `phylo_coupled = true` gives drmTMB's -111.84106326.
# Reference numbers and provenance: test/fixtures/ade4/README.md.
using DRModels
using Test, LinearAlgebra, SparseArrays, Logging

const _D952 = DRModels
const _FX952 = joinpath(@__DIR__, "fixtures", "ade4")
const _SLOW952 = get(ENV, "DRM_SLOW_TESTS", "0") == "1"
const _H952 = 53.8

function _data952()
    lines = readlines(joinpath(_FX952, "carni70.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    return (species = [String(strip(r[1], '"')) for r in rows],
            lsize = [parse(Float64, r[2]) for r in rows],
            lrange = [parse(Float64, r[3]) for r in rows])
end

const _NWK952 = String(strip(read(joinpath(_FX952, "carni70.nwk"), String)))
# The same tree with every branch length divided by its height.
const _NWK952_UNIT = replace(_NWK952, r":[0-9.eE+-]+" =>
                             m -> ":" * repr(parse(Float64, m[2:end]) / _H952))

@testset "#952: tree scale of the sigma-phylo precision" begin
    d = _data952()
    Qr, _, _ = _D952._locscale_phylo_setup(augmented_phy(_NWK952), d.species)
    Qu, _, _ = _D952._locscale_phylo_setup(augmented_phy(_NWK952_UNIT), d.species)
    # diag(Q⁻¹) is the root-to-node path length, so its maximum is the tree height.
    @test _D952._glsp_tree_scale(Qr) ≈ phylo_tree_height(augmented_phy(_NWK952)) rtol = 1e-12
    @test _D952._glsp_tree_scale(Qr) ≈ _H952 rtol = 1e-12
    # Unit height snaps to exactly 1, so a unit-height tree is fitted as before.
    @test _D952._glsp_tree_scale(Qu) === 1.0
    @test _D952._glsp_tree_scale(sparse(1.0I, 5, 5)) === 1.0
    # Not positive definite: no rescaling rather than an error.
    @test _D952._glsp_tree_scale(sparse([1.0 2.0; 2.0 1.0])) === 1.0
end

@testset "#952: the rescaled tree is an exact reparameterisation" begin
    d = _data952(); n = length(d.lsize)
    Xμ = hcat(ones(n), d.lrange); Xψ = ones(n, 1)
    kind = Val(:gaussian_mean)
    Qr, gidx, G = _D952._locscale_phylo_setup(augmented_phy(_NWK952), d.species)
    c = _D952._glsp_tree_scale(Qr)
    Zη = _D952._ls_canonical_Zeta(n); Zψ = _D952._ls_canonical_Zpsi(n)
    # Separate block at its optimum region: the marginal NLL on (c·Q, log SD) equals
    # the one on (Q, log SD − ½ log c).
    θu = [1.36, 0.434, -0.003, -0.65, -0.18]
    θr = vcat(θu[1:3], θu[4:5] .- 0.5 * log(c))
    @test _D952._glsp_sep_nll(kind, d.lsize, Xμ, Xψ, gidx, G, c * Qr, θu, Zη, Zψ) ≈
          _D952._glsp_sep_nll(kind, d.lsize, Xμ, Xψ, gidx, G, Qr, θr, Zη, Zψ) rtol = 1e-9
    # Coupled block, λ = [log L11, L21, log L22]: L21 scales by 1/√c.
    λu = [-0.68, 0.21, -0.18]; λr = [λu[1] - 0.5 * log(c), λu[2] / sqrt(c), λu[3] - 0.5 * log(c)]
    η0 = Xμ * [1.40, 0.42]; ψ0 = Xψ * [-0.011]
    Pu = prior_precision(c * Qr, _D952._ls_lc_inv2x2(λu))
    Pr = prior_precision(Qr, _D952._ls_lc_inv2x2(λr))
    vu = _D952._glsp_ml_nll(kind, d.lsize, η0, ψ0, gidx, G, Pu, Zη, Zψ)
    vr = _D952._glsp_ml_nll(kind, d.lsize, η0, ψ0, gidx, G, Pr, Zη, Zψ)
    @test vu[3] && vr[3]
    @test vu[1] ≈ vr[1] rtol = 1e-9
    # The route's start values (log SD = log 0.3) on the unit-height scale evaluate.
    # On the RAW tree the same start is SD 2.2 on that scale, where the inner mode
    # ran into the sigma -> 0 collapse and returned the 1e18 sentinel (measured on
    # f71b292; not asserted here, since a better inner solver may evaluate it).
    θ0 = vcat(Xμ \ d.lsize, 0.0, log(0.3), log(0.3))
    @test _D952._glsp_sep_nll(kind, d.lsize, Xμ, Xψ, gidx, G, c * Qr, θ0, Zη, Zψ) < 1e17
end

@testset "#952: the penalty is the user's raw-scale penalty on the rescaled tree" begin
    c = _H952
    pen = drm_phylo_penalty(sd_u = 0.4, sd_alpha = 0.05, cor_sd = 0.7)
    pc = _D952._glsp_rescale_penalty(pen, c)
    @test _D952._glsp_rescale_penalty(nothing, c) === nothing
    @test pc.sd_u ≈ pen.sd_u * sqrt(c) && pc.sd_alpha == pen.sd_alpha && pc.cor_sd == pen.cor_sd
    @test pc.rate ≈ -log(pc.sd_alpha) / pc.sd_u rtol = 1e-14
    for v in (-3.0, -0.7, 0.0, 1.2)
        a = _D952._phylo_pen_sd(pen.rate, v - 0.5 * log(c))   # raw-scale SD
        b = _D952._phylo_pen_sd(pc.rate, v)                    # unit-scale SD
        @test a[1] ≈ b[1] rtol = 1e-12
        @test a[2] ≈ b[2] rtol = 1e-12
    end
    # Coupled block: value and gradient in the route's own coordinates.
    i0 = 4
    θu = [0.1, 0.2, 0.3, -0.6, 0.25, -0.4]
    θr = copy(θu); θr[i0] -= 0.5 * log(c); θr[i0+1] /= sqrt(c); θr[i0+2] -= 0.5 * log(c)
    gu = zeros(6); gr = zeros(6)
    vu = _D952._phylo_pen_apply_coupled!(gu, pc, θu, i0)
    vr = _D952._phylo_pen_apply_coupled!(gr, pen, θr, i0)
    @test vu ≈ vr rtol = 1e-12
    # Chain rule through θr(θu): ∂/∂L21_unit = ∂/∂L21_raw / √c, identity elsewhere.
    @test gu[i0] ≈ gr[i0] rtol = 1e-10
    @test gu[i0+1] ≈ gr[i0+1] / sqrt(c) rtol = 1e-10
    @test gu[i0+2] ≈ gr[i0+2] rtol = 1e-10
end

@testset "#952: back-transform of a fit made on the rescaled tree" begin
    c = 4.0
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:2, :sigma => 3:3, :recov => 4:6]
    names = Pair{Symbol,Vector{String}}[:mu => ["(Intercept)", "x"], :sigma => ["(Intercept)"],
                                        :recov => ["g:L11", "g:L22", "g:L21"]]
    θ = [1.0, 2.0, 0.5, -0.2, -0.4, 0.3]
    V = Matrix{Float64}(I, 6, 6) .+ 0.1
    scales = Dict(:sigma => [1.0], :lambda_sd_mu => [0.8], :lambda_sd_sigma => [0.7],
                  :lambda_cor => [0.4], :profile_ci_sd_mu => [0.0, 2.0])
    fit = _D952.DrmFit(Gaussian(), blocks, names, θ, V, -10.0, 5, true,
                       Dict(:mu => zeros(5)), Dict(:mu => zeros(5)), scales)
    fr = _D952._glsp_unscale_fit(fit, c, nothing)
    @test fr.theta ≈ [1.0, 2.0, 0.5, -0.2 - log(2), -0.4 - log(2), 0.15]
    J = Diagonal([1, 1, 1, 1, 1, 0.5])
    @test fr.vcov ≈ J * V * J
    @test fr.scales[:lambda_sd_mu] ≈ [0.4] && fr.scales[:lambda_sd_sigma] ≈ [0.35]
    @test fr.scales[:lambda_cor] == [0.4] && fr.scales[:sigma] == [1.0]
    @test fr.scales[:profile_ci_sd_mu] ≈ [0.0, 1.0]
    @test fit.scales[:lambda_sd_mu] == [0.8]            # the input fit is not mutated
    @test fr.loglik == fit.loglik && fr.converged && fr.blocks == fit.blocks
    # Separate block: both log SDs move, nothing else.
    fs = _D952.DrmFit(Gaussian(), Pair{Symbol,UnitRange{Int}}[:mu => 1:1, :sigma => 2:2,
                      :resd_mu => 3:3, :resd_sigma => 4:4],
                      Pair{Symbol,Vector{String}}[:mu => ["a"], :sigma => ["b"],
                      :resd_mu => ["g:sd_mu"], :resd_sigma => ["g:sd_sigma"]],
                      [0.1, 0.2, 0.3, 0.4], Matrix{Float64}(I, 4, 4), -3.0, 5, true,
                      Dict(:mu => zeros(5)), Dict(:mu => zeros(5)), Dict(:sigma => [1.0]))
    @test _D952._glsp_unscale_fit(fs, c, nothing).theta ≈ [0.1, 0.2, 0.3 - log(2), 0.4 - log(2)]
    @test _D952._glsp_unscale_fit(fs, c, nothing).vcov == Matrix{Float64}(I, 4, 4)
end

@testset "#952: the fit-failed warning prints is_converged, not the raw flag" begin
    # A sentinel fit (loglik -1e18): the `DrmFit` constructor clears the optimiser flag
    # on a sentinel (#1009 #1012 #1019), so the warning must print that cleared flag.
    blocks = Pair{Symbol,UnitRange{Int}}[:mu => 1:1, :sigma => 2:2]
    names = Pair{Symbol,Vector{String}}[:mu => ["(Intercept)"], :sigma => ["(Intercept)"]]
    fit = _D952.DrmFit(Gaussian(), blocks, names, [0.0, 0.0], fill(NaN, 2, 2), -1e18, 5, true,
                       Dict(:mu => zeros(5)), Dict(:mu => zeros(5)), Dict(:sigma => [1.0]))
    @test !fit.converged && !is_converged(fit)
    logs, _ = Test.collect_test_logs(min_level = Logging.Warn) do
        _D952._warn_variance_boundary(fit)
    end
    failed = filter(r -> occursin("the fit FAILED", string(r.message)), logs)
    @test length(failed) == 1
    @test failed[1].kwargs[:converged] === false
end

const _F952 = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)),
                 @formula(sigma ~ 1 + phylo(1 | species)))

_fit952(tree; kw...) = with_logger(NullLogger()) do
    drm(_F952, Gaussian(); data = _data952(), tree = tree, kw...)
end

# The issue's call on the raw tree, in the default suite so that CI catches a return
# of the -1e18 sentinel: a 4 s fit once compiled (the cell took 56.5 s in a fresh
# local shard, compilation included). On the unfixed code the fit took 158 s and
# returned the sentinel (measured 2026-10-16). Raw-vs-unit invariance and the
# drmTMB comparisons other than this one are in the DRM_SLOW_TESTS cells below.
@testset "#952: the issue's default fit on the raw tree" begin
    fit = _fit952(_NWK952)
    @test loglik(fit) > -1e15                                   # never the sentinel
    @test is_converged(fit)
    @test abs(loglik(fit) - (-111.94867803)) < 1e-3   # drmTMB ca3ddef, correlation held at 0
end

# ---- end to end: raw vs unit trees and drmTMB (DRM_SLOW_TESTS=1) -------------
# About 5 s per fit after the fix plus compilation; on the unfixed code the raw
# separate fit took 158 s and the raw coupled fit 87 s (measured, 2026-10-16), so
# these cells run only with DRM_SLOW_TESTS=1.
if _SLOW952
    @testset "#952: separate block (the default) on the raw and unit-height trees" begin
        raw = _fit952(_NWK952); unit = _fit952(_NWK952_UNIT)
        for fit in (raw, unit)
            @test isfinite(loglik(fit)) && loglik(fit) > -1e15     # never the sentinel
            @test is_converged(fit)
            # drmTMB ca3ddef's objective with its phylo correlation held at 0.
            @test abs(loglik(fit) - (-111.94867803)) < 1e-3
        end
        @test loglik(raw) ≈ loglik(unit) atol = 1e-6
        @test raw.theta[1:3] ≈ unit.theta[1:3] atol = 1e-6
        # SDs on the raw scale: sd_raw = sd_unit / √height.
        @test raw.theta[4:5] .+ 0.5 * log(_H952) ≈ unit.theta[4:5] atol = 1e-5
        @test raw.scales[:lambda_sd_mu][1] * sqrt(_H952) ≈ unit.scales[:lambda_sd_mu][1] rtol = 1e-5
        @test sqrt.(diag(raw.vcov)[1:3]) ≈ sqrt.(diag(unit.vcov)[1:3]) rtol = 1e-4
    end

    @testset "#952: coupled block matches drmTMB on both trees" begin
        raw = _fit952(_NWK952; phylo_coupled = true)
        unit = _fit952(_NWK952_UNIT; phylo_coupled = true)
        for fit in (raw, unit)
            @test is_converged(fit)
            @test abs(loglik(fit) - (-111.84106326)) < 1e-3       # drmTMB ca3ddef, df = 6
            @test fit.theta[1:3] ≈ [1.39559462183, 0.42067158068, -0.01110875406] atol = 1e-4
            @test fit.scales[:lambda_cor][1] ≈ 0.999999 * tanh(0.24334775573) atol = 1e-4
        end
        @test loglik(raw) ≈ loglik(unit) atol = 1e-6
        # drmTMB reports its SDs on the unit-tip-variance scale.
        @test log(unit.scales[:lambda_sd_mu][1]) ≈ -0.68444385693 atol = 1e-4
        @test log(unit.scales[:lambda_sd_sigma][1]) ≈ -0.14630323781 atol = 1e-4
        @test raw.scales[:lambda_sd_sigma][1] * sqrt(_H952) ≈ unit.scales[:lambda_sd_sigma][1] rtol = 1e-5
        @test raw.scales[:lambda_cor] ≈ unit.scales[:lambda_cor] atol = 1e-6
    end

    @testset "#952: sigma-only (asymmetric) block on the raw and unit-height trees" begin
        f = bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1 + phylo(1 | species)))
        asym(tree) = with_logger(NullLogger()) do
            drm(f, Gaussian(); data = _data952(), tree = tree)
        end
        raw = asym(_NWK952); unit = asym(_NWK952_UNIT)
        @test is_converged(raw) && is_converged(unit)
        @test loglik(raw) ≈ loglik(unit) atol = 1e-6
        @test raw.theta[end] + 0.5 * log(_H952) ≈ unit.theta[end] atol = 1e-5
    end
else
    @info "#952 end-to-end carni70 cells skipped; set DRM_SLOW_TESTS=1 to run"
end
