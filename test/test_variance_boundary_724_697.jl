# Variance-component boundary diagnostic (#724, #697).
#
# #724: residual sigma is not twin-stable when the phylogenetic variance share -> 1
# (geiger::carnivores, n = 16). The MLE is sigma_e^2 = 0 (a boundary, not a flat
# ridge): the profile deviance is *increasing* in sigma_e^2 (slope c ~ 5.9 here), so
# on log sigma the gradient 2 c sigma^2 vanishes quadratically and any optimiser stops
# wherever it first falls under `g_tol` -- drmTMB at 4.3e-5, DRModels.jl at 3.1e-4 on
# the issue's data. These tests pin (1) the profile geometry, (2) the warning /
# `check_drm` flag, (3) its calibration gap and (4) that bootstrap refits stay quiet.
#
# #697: with ONE observation per tip, sigma_a^2 and sigma_e^2 are separated only by the
# off-diagonal structure of A. The Fisher-information tests pin the numbers quoted in
# docs/src/tutorials/location-scale-scale.md and docs/dev-log/2026-09-30-class3-boundary.md.
#
# The data are a 16-tip ultrametric coalescent tree and y = 2.04 + a, a ~ N(0, 4.42 A),
# simulated in R (ape::rcoal, set.seed(724)); no geiger data is bundled (GPL).

using DRModels
using Test
using LinearAlgebra
using Logging
using Random
using StableRNGs

const _VB = DRModels

const _VB_NWK_RAW = "((((t11:26.92985073,((t1:9.370193811,t16:9.370193811):10.47145133,((t9:11.50636665,t8:11.50636665):3.108531054,t15:14.61489771):5.226747434):7.088205585):1.965468074,(t14:0.2157888031,t5:0.2157888031):28.67953):15.43701817,(t6:42.74982625,(t2:3.481722412,t7:3.481722412):39.26810384):1.582510714):14.86766303,(t4:22.93679668,((t13:10.14301435,(t12:2.949611415,t3:2.949611415):7.19340293):8.204339287,t10:18.34735363):4.589443051):36.26320332);"
# unit-height copy (the raw tree has height 59.2, which makes `drm` warn about the sd_phylo scale)
_vb_unit(nwk, h) = replace(nwk, r":([0-9.]+)" => m -> ":" * string(parse(Float64, m[2:end]) / h))
const _VB_NWK = _vb_unit(_VB_NWK_RAW, 59.2)
const _VB_TIPS = ["t11", "t1", "t16", "t9", "t8", "t15", "t14", "t5", "t6", "t2", "t7", "t4", "t13", "t12", "t3", "t10"]
const _VB_Y = [-2.46969390003231, -0.448315300042628, -0.447302767353372, -1.4552325177712,
               -3.65929529983894, -1.45807028430632, 1.0265224427155, 0.866246033988844,
               1.49070074839223, -1.5047227993107, -1.51541813913103, -1.9078998246347,
               2.28017456732853, 1.17273767163316, 2.80674407060564, 1.58395729601422]

_vb_formula() = bf(@formula(y ~ 1 + phylo(1 | species)), @formula(sigma ~ 1))
_vb_fit(y, nwk, tips) = drm(_vb_formula(), Gaussian(); data = (; y, species = tips), tree = nwk)

# Profile negative log-likelihood over sigma_e (dense Gaussian, mu by GLS, sigma_a^2 by
# golden section): the engine-independent check of the geometry.
function _vb_profile_nll(y, A, σe)
    n = length(y); one = ones(n)
    function nll(la2)
        V = exp(la2) * A + σe^2 * I
        F = cholesky(Symmetric(V)); Vi1 = F \ one
        μ = dot(Vi1, y) / dot(Vi1, one); r = y .- μ
        return 0.5 * (logdet(F) + dot(r, F \ r)) + 0.5 * n * log(2π)
    end
    a, b = log(1e-4), log(1e3); φ = (sqrt(5) - 1) / 2
    c = b - φ * (b - a); d = a + φ * (b - a)
    for _ in 1:80
        if nll(c) < nll(d); b = d else a = c end
        c = b - φ * (b - a); d = a + φ * (b - a)
    end
    return nll((a + b) / 2)
end

@testset "variance-boundary diagnostic (#724 / #697)" begin
    phy = DRModels.augmented_phy(_VB_NWK)
    A = DRModels.sigma_phy_dense(phy; σ²_phy = 1.0)
    @test maximum(diag(A)) ≈ 1.0 atol = 1e-8          # unit-height ultrametric tree
    ord = [findfirst(==(t), String.(phy.leaf_names)) for t in _VB_TIPS]
    Ao = A[ord, ord]

    @testset "#724 geometry: MLE is sigma_e^2 = 0, the profile is increasing in sigma_e^2" begin
        p0 = _vb_profile_nll(_VB_Y, Ao, 1e-8)
        # flat to optimiser precision over the whole range engines report (3e-5 … 3e-4)
        @test abs(_vb_profile_nll(_VB_Y, Ao, 4.3e-5) - p0) < 1e-7
        @test abs(_vb_profile_nll(_VB_Y, Ao, 3.07e-4) - p0) < 1e-5
        # …but not flat in sigma_e^2: slope c = d nll / d sigma_e^2 ≈ 5.9 > 0 (measured 5.94 at sigma_e = 0.01, 6.08 at 0.1)
        c = (_vb_profile_nll(_VB_Y, Ao, 0.1) - p0) / 0.1^2
        @test 4.0 < c < 8.0
        # so d nll / d log sigma_e = 2 c sigma_e^2 -> 0 quadratically: log-scale gradient at
        # sigma_e = 1e-4 is already ~1e-7, below any sensible `g_tol`.
        @test 2 * c * (1e-4)^2 < 1e-6
    end

    fit = Logging.with_logger(Logging.NullLogger()) do
        _vb_fit(_VB_Y, _VB_NWK, _VB_TIPS)
    end

    @testset "#724 the fit lands on the boundary and agrees with the dense likelihood" begin
        σ̂ = exp(coef(fit, :sigma)[1])
        @test σ̂ < 1e-3
        p0 = _vb_profile_nll(_VB_Y, Ao, 1e-8)
        @test loglik(fit) ≈ -p0 atol = 1e-6              # the likelihood is twin-stable
    end

    @testset "#724 warns at fit time, names the cause, and check_drm carries the flag" begin
        msgs = String[]
        lg = Test.TestLogger(min_level = Logging.Warn)
        Logging.with_logger(lg) do
            _vb_fit(_VB_Y, _VB_NWK, _VB_TIPS)
        end
        warns = [string(r.message) for r in lg.logs if r.level == Logging.Warn]
        hit = filter(m -> occursin("lower boundary", m), warns)
        @test length(hit) == 1
        @test occursin("optimiser-stopping artefact", hit[1])
        @test occursin("ONE observation per group", hit[1])       # #697 identification note
        vb = _VB._variance_boundary(fit)
        @test vb.residual_at_boundary
        @test vb.residual_ratio < 1e-3
        @test isempty(vb.structured_at_boundary)
        @test vb.one_obs_per_group
        rep = Logging.with_logger(Logging.NullLogger()) do
            check_drm(fit)
        end
        @test rep.variance_boundary.residual_at_boundary
        lg2 = Test.TestLogger(min_level = Logging.Warn)
        Logging.with_logger(lg2) do
            check_drm(fit)
        end
        @test any(r -> occursin("lower boundary", string(r.message)), lg2.logs)
    end

    @testset "bootstrap refits stay quiet (task-local suppression)" begin
        lg = Test.TestLogger(min_level = Logging.Warn)
        Logging.with_logger(lg) do
            _VB._without_boundary_warnings(() -> _vb_fit(_VB_Y, _VB_NWK, _VB_TIPS))
        end
        @test !any(r -> occursin("lower boundary", string(r.message)), lg.logs)
        # and the switch is restored afterwards
        lg3 = Test.TestLogger(min_level = Logging.Warn)
        Logging.with_logger(lg3) do
            _vb_fit(_VB_Y, _VB_NWK, _VB_TIPS)
        end
        @test any(r -> occursin("lower boundary", string(r.message)), lg3.logs)
    end

    @testset "interior fit: no flag, no boundary warning" begin
        # same tree, residual share 0.5: sigma_e is estimated well inside
        rng = StableRNG(4)
        L = cholesky(Symmetric(0.5 * 4 * Ao + 0.5 * 4 * I)).L
        y = 2.0 .+ L * randn(rng, 16)
        lg = Test.TestLogger(min_level = Logging.Warn)
        fit2 = Logging.with_logger(lg) do
            _vb_fit(y, _VB_NWK, _VB_TIPS)
        end
        vb = _VB._variance_boundary(fit2)
        @test vb.residual_ratio > 1e-2             # measured interior minimum is 2.2e-2
        @test !vb.residual_at_boundary
        @test !any(r -> occursin("lower boundary", string(r.message)), lg.logs)
    end

    @testset "not defined (returns nothing) where the residual is not one scalar" begin
        rng = StableRNG(3)
        n = 60; x = randn(rng, n)
        f0 = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian();
                data = (; y = 1 .+ x .+ exp.(0.3 .* x) .* randn(rng, n), x))
        end
        @test _VB._variance_boundary(f0) === nothing    # no random effect
        @test check_drm(f0).variance_boundary === nothing
    end

    @testset "#697 star tree: sigma_a, sigma_e not separable -> Hessian guard speaks" begin
        n = 16
        nwk = "(" * join(["t$i:1.0" for i in 1:n], ",") * ");"
        rng = StableRNG(5)
        y = 2.0 .+ sqrt(2.0) .* randn(rng, n)
        lg = Test.TestLogger(min_level = Logging.Warn)
        f3 = Logging.with_logger(lg) do
            _vb_fit(y, nwk, ["t$i" for i in 1:n])
        end
        @test any(r -> occursin("numerically singular", string(r.message)), lg.logs)
        # the total variance IS identified: sigma_a^2 + sigma_e^2 = MLE of the iid variance
        tot = exp(2 * f3.theta[3]) * 1.0 + exp(2 * f3.theta[2])   # star tree: A = I, height 1
        @test tot ≈ sum(abs2, y .- sum(y) / n) / n atol = 1e-3
    end

    @testset "#697 near-star tree: structured SD collapses and is flagged" begin
        # 8 cherries, internal branch 0.05, tips 0.95 -> mean off-diagonal correlation 0.0033
        cher(i) = "(t$(2i - 1):0.95,t$(2i):0.95):0.05"
        nwk = "(" * join([cher(i) for i in 1:8], ",") * ");"
        rng = StableRNG(11)
        y = 2.0 .+ 1.5 .* randn(rng, 16)                  # no phylogenetic structure at all
        lg = Test.TestLogger(min_level = Logging.Warn)
        f4 = Logging.with_logger(lg) do
            _vb_fit(y, nwk, ["t$i" for i in 1:16])
        end
        vb = _VB._variance_boundary(f4)
        @test vb.structured_at_boundary == [:species]
        @test vb.structured_ratios[:species] < 1e-3
        @test any(r -> occursin("Structured SD at its lower boundary", string(r.message)), lg.logs)
    end

    @testset "#697 Fisher information: separation comes from A, quantified" begin
        # I = 1/2 [tr((V⁻¹A)²) tr(V⁻¹A V⁻¹); · tr(V⁻²)] for (σ_a², σ_e²); ρ = corr of the estimates.
        function info_mat(A, sa2, se2)
            V = sa2 * A + se2 * I; Vi = inv(V); M = Vi * A
            return 0.5 * [tr(M * M) tr(M * Vi); tr(M * Vi) tr(Vi * Vi)]
        end
        function info_corr(A, sa2, se2)
            C = inv(info_mat(A, sa2, se2)); return C[1, 2] / sqrt(C[1, 1] * C[2, 2]), 0
        end
        ρ_tree, _ = info_corr(Ao, 3.0, 1.0)
        @test -0.9 < ρ_tree < -0.1                    # a real coalescent tree separates them
        cher16(d) = (K = zeros(16, 16); for i in 1:16; K[i, i] = 1; end;
                     for i in 1:8; K[2i - 1, 2i] = K[2i, 2i - 1] = d; end; K)
        ρ1, _ = info_corr(cher16(0.5), 3.0, 1.0)
        ρ2, _ = info_corr(cher16(0.05), 3.0, 1.0)
        @test ρ2 < ρ1 < 0                              # weaker structure, stronger confounding
        @test ρ2 < -0.95                               # near-star: correlation -> -1
        @test abs(det(info_mat(Matrix(1.0I, 16, 16), 3.0, 1.0))) < 1e-12   # A = I: singular, only the sum identified
    end
end
