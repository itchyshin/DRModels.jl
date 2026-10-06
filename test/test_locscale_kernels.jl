# Two-axis (mean + log-dispersion) conditional-likelihood kernels for the
# non-Gaussian phylogenetic location–scale model (#202 groundwork).
# Gates the analytic gradient + Hessian against ForwardDiff. Engine-lane test:
# calls the internal kernels directly (not yet wired into drm()).
using DRModels
using Test
import ForwardDiff

@testset "location–scale kernels: analytic grad/Hessian vs ForwardDiff" begin
    # (family, y-grid) — y > 0 for Gamma, nonnegative counts for NB2.
    cases = [
        (Val(:nb2),   [0.0, 1.0, 3.0, 7.0, 20.0]),
        (Val(:gamma), [0.2, 0.8, 1.5, 4.0, 12.0]),
    ]
    ηs = [-1.2, -0.3, 0.0, 0.6, 1.5]
    ψs = [-1.0, -0.2, 0.4, 1.1, 2.0]

    for (kind, ys) in cases
        for y in ys, η in ηs, ψ in ψs
            f = θ -> DRModels._ls_nll(kind, y, θ[1], θ[2])
            θ = [η, ψ]

            gη, gψ = DRModels._ls_grad(kind, y, η, ψ)
            g_ad = ForwardDiff.gradient(f, θ)
            @test gη ≈ g_ad[1] rtol = 1e-6 atol = 1e-8
            @test gψ ≈ g_ad[2] rtol = 1e-6 atol = 1e-8

            hηη, hηψ, hψψ = DRModels._ls_hess(kind, y, η, ψ)
            H_ad = ForwardDiff.hessian(f, θ)
            @test hηη ≈ H_ad[1, 1] rtol = 1e-6 atol = 1e-8
            @test hηψ ≈ H_ad[1, 2] rtol = 1e-6 atol = 1e-8
            @test hηψ ≈ H_ad[2, 1] rtol = 1e-6 atol = 1e-8
            @test hψψ ≈ H_ad[2, 2] rtol = 1e-6 atol = 1e-8
        end
    end
end

@testset "location–scale kernels: η-axis matches the fixed-dispersion kernels" begin
    # The mean-axis derivatives must agree with the verified fixed-nuisance
    # kernels in sparse_laplace_glmm.jl at the corresponding dispersion value.
    for (η, ψ, y) in [(0.3, 0.5, 4.0), (-0.4, 1.2, 1.0), (0.9, -0.3, 11.0)]
        # NB2: size r = exp(-2ψ).
        r = exp(-2ψ)
        aux_nb = (y = [y], size = r, lconst = [0.0])
        gη, _ = DRModels._ls_grad(Val(:nb2), y, η, ψ)
        hηη, hηψ, _ = DRModels._ls_hess(Val(:nb2), y, η, ψ)
        @test gη ≈ DRModels._laplace_d1(Val(:nb2_fixed), aux_nb, 1, η) rtol = 1e-10
        @test hηη ≈ DRModels._laplace_d2(Val(:nb2_fixed), aux_nb, 1, η) rtol = 1e-10
        @test hηψ ≈ -2 * DRModels._laplace_nuisance_d1(Val(:nb2_fixed), aux_nb, 1, η) rtol = 1e-10

        # Gamma: shape α = exp ψ.
        α = exp(ψ)
        aux_g = (y = [y], shape = α, lconst = [0.0])
        gηg, _ = DRModels._ls_grad(Val(:gamma), y, η, ψ)
        hηηg, _, _ = DRModels._ls_hess(Val(:gamma), y, η, ψ)
        @test gηg ≈ DRModels._laplace_d1(Val(:gamma_fixed), aux_g, 1, η) rtol = 1e-10
        @test hηηg ≈ DRModels._laplace_d2(Val(:gamma_fixed), aux_g, 1, η) rtol = 1e-10
    end
end
