# DJ1: Poisson and log-normal two-axis leaves in src/locscale_kernels.jl.
# The other families are already gated in test_locscale_kernels.jl. These two
# leaves had no caller. Oracles are Distributions densities and ForwardDiff of
# those densities, measured at BLAS = 1 and --check-bounds=yes.
using DRModels, Test, LinearAlgebra
import Distributions
import ForwardDiff

const _ATOL = 1e-8
const _RTOL = 1e-8

_fd1(f, x) = ForwardDiff.derivative(f, x)
_fd2(f, x) = ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), x)

@testset "locscale Poisson leaf ignores the scale axis" begin
    y = 4.0
    η = 0.4
    ψ = -1.2
    μ = exp(η)
    oracle = let y = y
        (e, p) -> -Distributions.logpdf(Distributions.Poisson(exp(e)), y)
    end
    nll = DRModels._ls_nll(Val(:poisson), y, η, ψ)
    gη, gψ = DRModels._ls_grad(Val(:poisson), y, η, ψ)
    hηη, hηψ, hψψ = DRModels._ls_hess(Val(:poisson), y, η, ψ)
    @test nll ≈ oracle(η, ψ) atol = _ATOL rtol = _RTOL
    @test gη ≈ _fd1(e -> oracle(e, ψ), η) atol = _ATOL rtol = _RTOL
    @test gψ == 0
    @test hηη ≈ _fd2(e -> oracle(e, ψ), η) atol = _ATOL rtol = _RTOL
    @test hηψ == 0
    @test hψψ == 0
    # ψ is not a parameter of this leaf: moving it must not move the value.
    @test DRModels._ls_nll(Val(:poisson), y, η, 3.0) == nll
    # Documented clamp: |η| above LS_CLAMP is evaluated at the bound.
    η_hi = DRModels.LS_CLAMP + 10
    clamped = DRModels._ls_nll(Val(:poisson), y, η_hi, ψ)
    @test clamped ≈ oracle(DRModels.LS_CLAMP, ψ) atol = _ATOL rtol = _RTOL
    @test clamped != oracle(η_hi, ψ)
    gη_hi, gψ_hi = DRModels._ls_grad(Val(:poisson), y, η_hi, ψ)
    @test gη_hi ≈ exp(DRModels.LS_CLAMP) - y atol = _ATOL rtol = _RTOL
    @test gψ_hi == 0
end

@testset "locscale log-normal leaf matches LogNormal(η, exp(ψ))" begin
    y = 1.8
    η = -0.3
    ψ = 0.25
    oracle = let y = y
        (e, p) -> -Distributions.logpdf(Distributions.LogNormal(e, exp(p)), y)
    end
    nll = DRModels._ls_nll(Val(:lognormal), y, η, ψ)
    gη, gψ = DRModels._ls_grad(Val(:lognormal), y, η, ψ)
    hηη, hηψ, hψψ = DRModels._ls_hess(Val(:lognormal), y, η, ψ)
    @test nll ≈ oracle(η, ψ) atol = _ATOL rtol = _RTOL
    @test gη ≈ _fd1(e -> oracle(e, ψ), η) atol = _ATOL rtol = _RTOL
    @test gψ ≈ _fd1(p -> oracle(η, p), ψ) atol = _ATOL rtol = _RTOL
    @test hηη ≈ _fd2(e -> oracle(e, ψ), η) atol = _ATOL rtol = _RTOL
    @test hψψ ≈ _fd2(p -> oracle(η, p), ψ) atol = _ATOL rtol = _RTOL
    @test hηψ ≈ ForwardDiff.derivative(p -> _fd1(e -> oracle(e, p), η), ψ) atol = _ATOL rtol = _RTOL
    # Median of the draw is exp(η): the kernel is the log-scale Gaussian, so
    # doubling y at the median η = log(y) is a pure Jacobian plus ½ log(2π).
    y0 = 2.5
    @test DRModels._ls_nll(Val(:lognormal), y0, log(y0), 0.0) ≈
          0.5 * log(2π) + log(y0) atol = _ATOL rtol = _RTOL
end
