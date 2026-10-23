# End-to-end coverage for the Gamma location–scale family (#202). The Gamma
# kernel and its exact gradient are gated in test_locscale_grad.jl; here we
# confirm the WHOLE pipeline — gradient-driven fit + Wald inference + components —
# works for Gamma, not just NB2. Assertions are seed-robust (stationarity of the
# exact gradient, PD vcov), with only a loose mean-slope recovery check.
using DRModels
using Test, Random, LinearAlgebra, SparseArrays
import Distributions

# Gamma with mean μ = exp η and shape α = exp ψ (rate = α/μ), matching the kernel.
_gamma_draw(η, ψ) = (α = exp(ψ); μ = exp(η); rand(Distributions.Gamma(α, μ / α)))

@testset "location–scale Gamma: end-to-end fit + inference" begin
    Random.seed!(31337)
    G = 30; m = 30; n = G * m
    species = repeat(1:G, inner = m)
    x = randn(n)
    βμ = [0.3, 0.4]; βψ = [0.6]              # shape ≈ exp(0.6) ≈ 1.8
    Λtrue = [0.20 0.03; 0.03 0.12]
    LΛ = cholesky(Symmetric(Λtrue)).L
    A = [LΛ * randn(2) for _ in 1:G]
    Xμ = hcat(ones(n), x); Xψ = ones(n, 1)
    y = [_gamma_draw(βμ[1] + βμ[2] * x[i] + A[species[i]][1],
                     βψ[1] + A[species[i]][2]) for i in 1:n]
    Q = sparse(1.0 * I, G, G)

    fit = DRModels._fit_locscale(Val(:gamma), y, Xμ, Xψ, species, G, Q; se = true)

    gmax = maximum(abs.(DRModels._ls_marginal_grad(Val(:gamma), y, Xμ, Xψ, species, G, Q, fit.θ)))
    @test gmax < 1e-3                          # stationarity of the exact gradient
    @test isposdef(Symmetric(fit.Lambda))
    @test isposdef(Symmetric(fit.vcov))        # valid Wald covariance
    @test all(isfinite, fit.se) && all(fit.se .> 0)
    @test fit.components.sd_mu ≈ sqrt(fit.Lambda[1, 1])
    @test fit.beta_mu[2] ≈ 0.4 atol = 0.15     # mean slope (loose, single seed)
end
