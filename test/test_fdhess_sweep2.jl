# Finite-difference Hessian guard sweep 2: `_loconly_fd_hessian2` (value-based) and
# `_q4_fd_vcov` (gradient-based). Before the guard a failed probe gave a finite
# garbage Hessian (1e18 sentinel differenced) or a crash (empty gradient broadcast).

using DRModels
using Test
using LinearAlgebra
using Random

const DFS = DRModels

# Pre-fix bodies, verbatim, as the "today" oracles.
function _legacy_loconly_hessian2(f, θ; h = 1e-4)
    H = zeros(2, 2); x = Float64.(θ)
    for i in 1:2, j in 1:2
        ei = zeros(2); ej = zeros(2)
        si = h * max(abs(x[i]), 1.0); sj = h * max(abs(x[j]), 1.0)
        ei[i] = si; ej[j] = sj
        H[i, j] = (f(x .+ ei .+ ej) - f(x .+ ei .- ej) -
                   f(x .- ei .+ ej) + f(x .- ei .- ej)) / (4 * si * sj)
    end
    0.5 .* (H .+ H')
end

function _legacy_q4_hessian(grad_at, θ; h = 1e-4)
    nθ = length(θ); H = zeros(nθ, nθ)
    for k in 1:nθ
        θp = copy(θ); θp[k] += h
        θm = copy(θ); θm[k] -= h
        H[:, k] .= (grad_at(θp) .- grad_at(θm)) ./ (2h)
    end
    H
end

@testset "FD Hessian sweep 2" begin
    @testset "_loconly_fd_hessian2: sentinel probe" begin
        A = [2.0 0.3; 0.3 1.0]
        quad(v) = 0.5 * dot(v, A * v)
        θ = [0.4, -0.2]
        bad(v) = v[1] > θ[1] + 5e-5 && v[2] < θ[2] ? DFS._LOCONLY_PENALTY : quad(v)

        # today: finite garbage that would pass any later finiteness check
        Hold = _legacy_loconly_hessian2(bad, θ)
        @test all(isfinite, Hold)
        @test norm(Hold - A) > 1e3

        # guarded: NaN matrix
        H = DFS._loconly_fd_hessian2(bad, θ)
        @test size(H) == (2, 2) && all(isnan, H)
        @test all(isnan, DFS._loconly_fd_hessian2(v -> v[1] > 0.4 ? NaN : quad(v), θ))

        # healthy: unchanged to 1e-10
        f(v) = exp(v[1]) + sin(v[2]) * v[1]^2 + 0.3v[1] * v[2]
        for h in (1e-3, 1e-4, 1e-5), x in ([0.3, -0.7], [2.5, 3.0])
            @test isapprox(DFS._loconly_fd_hessian2(f, x; h = h),
                           _legacy_loconly_hessian2(f, x; h = h); atol = 1e-10, rtol = 1e-10)
        end
    end

    @testset "loconly public diagnostics at the |lσ| < 50 edge" begin
        Random.seed!(11)
        G = 8
        phy = random_balanced_tree(G; branch_length = 0.25)
        species = repeat(1:G, inner = 2)
        n = length(species)
        X = hcat(ones(n), collect(range(-1, 1; length = n)))
        y = X * [0.2, -0.3] .+ randn(n)
        prob = DFS.make_loc_problem(phy, y, X; species = species)
        # probes at lσ = 49.99999 + 5e-3 leave the valid box -> penalty sentinel
        d = DFS._loconly_reml_fd_stability_diagnostic(prob, 49.9999, 0.0)
        @test !d.finite
        # healthy interior point stays finite
        d2 = DFS._loconly_reml_fd_stability_diagnostic(prob, log(0.5), log(0.5))
        @test d2.finite
    end

    @testset "_q4_fd_vcov_from_grad: failed gradient probe" begin
        θ = [0.3, -0.5, 0.8]
        A = [3.0 0.4 0.1; 0.4 2.0 0.2; 0.1 0.2 1.5]
        g_ok(t) = A * t .+ 0.1 .* t .^ 3
        θbad_gate(t) = t[2] < θ[2] - 5e-5
        g_empty(t) = θbad_gate(t) ? Float64[] : g_ok(t)
        g_nan(t) = θbad_gate(t) ? fill(NaN, 3) : g_ok(t)

        # today: empty gradient -> DimensionMismatch; NaN gradient -> NaN Hessian
        @test_throws DimensionMismatch _legacy_q4_hessian(g_empty, θ)
        @test any(isnan, _legacy_q4_hessian(g_nan, θ))

        for g in (g_empty, g_nan)
            V = DFS._q4_fd_vcov_from_grad(g, θ)
            @test size(V) == (3, 3) && all(isnan, V)
        end

        # healthy: unchanged to 1e-10 vs the legacy inline loop -> _vcov_from_hessian
        Vold = DFS._vcov_from_hessian(_legacy_q4_hessian(g_ok, θ))
        Vnew = DFS._q4_fd_vcov_from_grad(g_ok, θ)
        @test isapprox(Vnew, Vold; atol = 1e-10, rtol = 1e-10)
        θb = [1.5, -3.0, 2.2]      # |θ| > 1: absolute (unscaled) step preserved
        @test isapprox(DFS._q4_fd_vcov_from_grad(g_ok, θb),
                       DFS._vcov_from_hessian(_legacy_q4_hessian(g_ok, θb)); atol = 1e-10, rtol = 1e-10)
    end
end
