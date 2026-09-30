# Value-based finite-difference Hessian guard (`_fd_hessian_from_values`).
# The sparse LSS REML branch differences nll VALUES; a failed probe returns a
# 1e18 sentinel. Unguarded, the sentinel yields a finite garbage Hessian that
# reaches `_vcov_from_hessian` (and the eigenvalue guard) as if it were real.

using DRModels
using Test
using LinearAlgebra

const DSH = DRModels

# The pre-fix inline loop from src/gaussian_sparse_lss.jl (REML branch).
function _legacy_value_hessian(f, θ̂; hstep = 1e-5)
    np = length(θ̂); H = zeros(np, np)
    for k in 1:np, j in k:np
        θpp = copy(θ̂); θpm = copy(θ̂); θmp = copy(θ̂); θmm = copy(θ̂)
        sk = hstep * max(abs(θ̂[k]), 1.0); sj = hstep * max(abs(θ̂[j]), 1.0)
        θpp[k] += sk; θpp[j] += sj; θpm[k] += sk; θpm[j] -= sj
        θmp[k] -= sk; θmp[j] += sj; θmm[k] -= sk; θmm[j] -= sj
        H[k, j] = (f(θpp) - f(θpm) - f(θmp) + f(θmm)) / (4 * sk * sj)
        H[j, k] = H[k, j]
    end
    H
end

@testset "value-based FD Hessian guard" begin
    A = [2.0 0.3; 0.3 1.0]
    quad(θ) = 0.5 * dot(θ, A * θ)
    θ̂ = [0.4, -0.2]
    # Fails whenever the probe leaves a small ball around θ̂ in coordinate 1
    # (mimics a non-PD factor at one probe).
    sentinel_f(θ) = θ[1] > θ̂[1] + 5e-6 && θ[2] < θ̂[2] ? 1e18 : quad(θ)
    nan_f(θ) = θ[1] > θ̂[1] + 5e-6 && θ[2] < θ̂[2] ? NaN : quad(θ)

    @testset "healthy objective reproduces the quadratic Hessian" begin
        H, ok = DSH._fd_hessian_from_values(quad, θ̂)
        @test ok
        @test isapprox(H, A; atol = 1e-4)
    end

    @testset "legacy loop turns a 1e18 sentinel into a garbage finite Hessian" begin
        H = _legacy_value_hessian(sentinel_f, θ̂)
        @test all(isfinite, H)
        @test maximum(abs, H) > 1e6          # not the true curvature (~2)
        # ... which the eigenvalue guard then happily inverts to a garbage vcov
        V = DSH._vcov_from_hessian(H)
        @test all(isfinite, V)
        # NaN probe: legacy path produces a non-finite Hessian
        Hn = _legacy_value_hessian(nan_f, θ̂)
        @test !all(isfinite, Hn)
    end

    @testset "guarded helper marks sentinel / non-finite probes failed" begin
        _, ok1 = DSH._fd_hessian_from_values(sentinel_f, θ̂)
        _, ok2 = DSH._fd_hessian_from_values(nan_f, θ̂)
        _, ok3 = DSH._fd_hessian_from_values(θ -> Inf, θ̂)
        @test !ok1 && !ok2 && !ok3
    end
end
