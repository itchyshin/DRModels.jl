# Cursor coverage 27 Nov 2026 (w3 lane): ranks 5–7 uncovered src files on main
# (`poisson.jl`, `gaussian_locscale_phylo.jl`, `gaussian_sigma_sd_laplace.jl`),
# skipping files already claimed by open 1127cov PRs (w2: inference / gaussian_core /
# location_only; pz; af; comparison).
using DRModels
using Test
using Random
using LinearAlgebra
using StatsModels
import Distributions

@testset "cursor coverage 2026-11-27 (w3)" begin

    @testset "poisson.jl — offset plumbing and Cox–Reid route guards" begin
        DR = DRModels
        @test DR.offset(3.0) === 3.0

        n = 6
        data = (; y = Float64.(rand(Distributions.Poisson(2.0), n)), x = randn(n), g = repeat(1:3, inner = 2))
        off = fill(1.0, n)
        @test DR._eta_with_offset(hcat(ones(n)), [0.5], off) ≈ fill(1.5, n)
        @test DR._add_offset([0.2, 0.3], nothing) == [0.2, 0.3]

        X = hcat(ones(n), data.x)
        θ0 = [0.0, 0.1]
        θ_adj = DR._offset_start(θ0, X, off)
        @test θ_adj[1] ≈ θ0[1] - sum(off) / n

        @test_throws ArgumentError DR._offset_for_fit(DRModels.Poisson(), off[1:end-1], data, n)

        @test_throws ArgumentError drm(
            bf(@formula(y ~ x + (1 | g))),
            DRModels.Poisson();
            data,
            method = :REML,
            marginal = :AGHQ,
            nAGQ = 3,
        )
        msg_aghq = try
            drm(bf(@formula(y ~ x + (1 | g))), DRModels.Poisson();
                  data, method = :REML, marginal = :AGHQ, nAGQ = 3)
        catch e
            sprint(showerror, e)
        end
        @test occursin("AGHQ", msg_aghq) && occursin("REML", msg_aghq)

        @test_throws ArgumentError drm(
            bf(@formula(y ~ x)),
            DRModels.Poisson(); data, method = :REML,
        )

        n_sd = 12
        g_sd = repeat(1:4, inner = 3)
        z = fill(1.0, n_sd)
        dat_sd = (; y = Float64.(rand(Distributions.Poisson(2.0), n_sd)), x = randn(n_sd), g = g_sd, z)
        @test_throws ArgumentError drm(
            bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)),
            DRModels.Poisson();
            data = dat_sd,
            method = :REML,
        )
        msg_sd = try
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)), DRModels.Poisson();
                data = dat_sd, method = :REML)
        catch e
            sprint(showerror, e)
        end
        @test occursin("maximum likelihood only", msg_sd)

        @test_throws ArgumentError drm(
            bf(@formula(y ~ x + (1 | g)), @formula(sd(g) ~ z)),
            DRModels.Poisson();
            data = dat_sd,
            nAGQ = 0,
        )
    end

    @testset "gaussian_locscale_phylo.jl — Λ parameterisation, rank, REML penalty" begin
        DR = DRModels
        λ = [log(0.6), log(0.4)]
        Λ = DR._glsp_sep_Λ(λ)
        @test Λ ≈ [0.36 0.0; 0.0 0.16] atol = 1e-12
        @test DR._glsp_coupled_Λ([λ[1], 0.15, λ[2]]) ≈ DR._ls_lc_to_Λ([λ[1], 0.15, λ[2]])

        @test !DR._glsp_rank_deficient(ones(8, 1))
        X = hcat(ones(8), collect(1:8), 2 .* collect(1:8) .+ 1)
        @test DR._glsp_rank_deficient(X)
        @test DR._glsp_rank_deficient(hcat(ones(8), 2 .* ones(8)))

        pμ = 2
        θ = [0.1, 0.2, 0.0, log(0.5), log(0.4)]
        grad_fn = θ_ -> begin
            g = zeros(length(θ_))
            g[1] = 2θ_[1]
            g[2] = 3θ_[2]
            g[3] = θ_[1] + θ_[2]
            g[4] = 0.5θ_[3]
            g[5] = -0.2θ_[4]
            g
        end
        pen = DR._glsp_reml_penalty(grad_fn, θ, pμ)
        @test isfinite(pen) && pen > 0

        θ_bad = [0.0, 0.0, 0.0, log(0.5), log(0.4)]
        grad_flat = θ_ -> zeros(length(θ_))
        @test DR._glsp_reml_penalty(grad_flat, θ_bad, pμ) == 1e18

        @test DR._glsp_reml_start([log(1e-6), 0.3], [1, 2]) == [log(0.05), 0.3]
    end

    @testset "gaussian_sigma_sd_laplace.jl — logaddexp, joint density, inner mode" begin
        DR = DRModels
        for (a, b) in ((2.0, 1.0), (-3.0, -3.1), (0.5, 0.5))
            @test DR._gsl_logaddexp(a, b) ≈ log(exp(a) + exp(b)) atol = 1e-12
        end

        y = [0.2, -0.1, 0.4]
        μ0 = [0.1, 0.0, 0.2]
        η0 = [log(0.5), log(0.6), log(0.4)]
        ga = [1, 1, 2]
        gb = [1, 2, 2]
        c = ones(3)
        d = ones(3)
        Ga, Gb = 2, 2
        va = [0.05, -0.02]
        vb = [0.03, 0.01]
        Pb = nothing

        F0, _, _ = DR._gsl_joint(Gaussian(), y, μ0, η0, ga, c, gb, d, Pb, va, vb, Ga, Gb; deriv = false)
        F1, g1, H1 = DR._gsl_joint(
            Gaussian(), y, μ0, η0, ga, c, gb, d, Pb, va, vb, Ga, Gb;
            deriv = true, rho = 0.25,
        )
        @test isfinite(F0) && isfinite(F1)
        @test F1 != F0
        @test all(isfinite, g1)
        @test issuccess(cholesky(Symmetric(H1)))

        joint = (v, deriv) -> begin
            vaa = @view v[1:Ga]
            vbb = @view v[(Ga + 1):end]
            DR._gsl_joint(Gaussian(), y, μ0, η0, ga, c, gb, d, Pb, vaa, vbb, Ga, Gb; deriv = deriv)
        end
        vhat, ok = DR._gsl_mode(joint, zeros(Ga + Gb))
        @test ok
        Fhat, ghat, _ = joint(vhat, true)
        @test isfinite(Fhat)
        @test maximum(abs, ghat) < 1e-6

        y2 = [1.0, 2.0, 0.0]
        Xμ = ones(3, 1)
        Xσ = ones(3, 1)
        gb2 = [1, 1, 2]
        d2 = ones(3)
        joint_reml = (w, deriv) -> DR._gsl_joint_reml(Gaussian(), y2, Xμ, Xσ, gb2, d2, nothing, w, 2; deriv = deriv)
        w0 = [0.1, 0.0, 0.2, log(0.5)]
        Fr, gr, Hr = joint_reml(w0, true)
        @test isfinite(Fr) && all(isfinite, gr)
        @test size(Hr) == (4, 4)
    end
end