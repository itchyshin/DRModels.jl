# Cancellation sweep: the remaining Gaussian marginal likelihoods that formed
# r′V⁻¹r (and, with two components, logdet H) as a DIFFERENCE of two O(1/σ_e²)
# quantities — the #835/#837/#838/#855 class. Each route's shipped `nll` closure
# is evaluated at log σ_e → −16 … −18 with ONE record per level and compared
# with a 256-bit BigFloat dense reassembly V = σ_e² I + Σ_k σ_k² Z_k K_k Z_kᵀ of
# the same marginal. Pre-fix numbers (same fixtures, same RNG stream, origin/main
# bdc4e8a81) are in the comments; post-fix every value is within 1e-8 relative.
using DRModels, Test, LinearAlgebra, Random

const _CS = DRModels

function _cs_dense_nll(y, mu, d, comps)          # comps: (gidx, G, K, σ²)
    setprecision(BigFloat, 256) do
        n = length(y)
        V = Matrix{BigFloat}(Diagonal(BigFloat.(d)))
        for (gidx, G, K, s2) in comps, i in 1:n, j in 1:n
            V[i, j] += BigFloat(s2) * BigFloat(K[gidx[i], gidx[j]])
        end
        r = BigFloat.(y) .- BigFloat.(mu)
        F = cholesky(Symmetric(V))
        Float64(0.5 * (logdet(F) + dot(r, F \ r)) + 0.5 * n * log(2 * BigFloat(pi)))
    end
end
_cs_rel(a, b) = abs(a - b) / max(abs(b), 1.0)

@testset "cancellation sweep: remaining Gaussian marginals" begin
    rng = MersenneTwister(11)

    # (A) _fit_multi_ranef_gaussian, (1 | g) + (1 | h) crossed, one record per g.
    # Pre-fix (Cholesky of I + Z̃ᵀD⁻¹Z̃ + Woodbury quadratic), rel. error vs the
    # BigFloat reference: lσ_e = −12: 1.2e-6; −16: 8.1e-3 (78.18 vs 77.56);
    # −20: Cholesky failure → 1e18 barrier. Cholesky logdet alone was off by
    # −0.44 at −18 (a fake hole). Post-fix (QR of [D^{-1/2}Z̃; I]): ≤ 2e-14.
    let n = 60
        g = collect(1:n); h = repeat(1:6, inner = 10)
        x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ 0.8 .* randn(rng, n)[g] .+ 0.6 .* randn(rng, 6)[h] .+ 0.3 .* randn(rng, n)
        Xμ = hcat(ones(n), x); Xσ = ones(n, 1)
        comps = [(ones(n), g, n, "g"), (ones(n), h, 6, "h")]
        fit = _CS._fit_multi_ranef_gaussian(Gaussian(), y, Xμ, Xσ, comps,
                                            ["(Intercept)", "x"], ["(Intercept)"], 1e-8)
        @test fit.converged
        for lse in (-2.0, -16.0, -18.0, -20.0)
            θ = [1.0, 0.5, lse, log(0.8), log(0.6)]
            ref = _cs_dense_nll(y, Xμ * θ[1:2], fill(exp(2lse), n),
                                [(g, n, Matrix(1.0I, n, n), 0.64), (h, 6, Matrix(1.0I, 6, 6), 0.36)])
            @test _cs_rel(fit.nll(θ), ref) < (lse == -2.0 ? 1e-12 : 1e-8)
        end
        # The analytic gradient (used by LBFGS) still matches ForwardDiff.
        θ = [1.0, 0.5, log(0.3), log(0.8), log(0.6)]
        gout = zeros(5); fit.nllgrad(gout, θ)
        @test isapprox(gout, _CS.ForwardDiff.gradient(fit.nll, θ); rtol = 1e-7, atol = 1e-8)
    end

    # (C) _fit_spatial_gaussian, spatial(1 | site), one record per site.
    # Pre-fix (q1 − C′M⁻¹C): lσ_e = −12: 1.3e-6; −16: 1.1e-2; −20: nll = −500.0
    # against a true +40.95 — a 541-nat fake hole. Post-fix (penalised RSS at
    # the conditional mode, whitened prior term): ≤ 4e-15.
    let G = 40
        coords = rand(rng, G, 2) .* 10
        site = collect(1:G); x = randn(rng, G)
        Dd = [sqrt(sum(abs2, coords[k, :] .- coords[l, :])) for k in 1:G, l in 1:G]
        C0 = exp.(-Dd ./ 2.0)
        y = 0.2 .+ 0.35 .* x .+ 0.7 .* (cholesky(Symmetric(C0 + 1e-8I)).L * randn(rng, G)) .+ 0.2 .* randn(rng, G)
        Xμ = hcat(ones(G), x); Xσ = ones(G, 1)
        fit = _CS._fit_spatial_gaussian(Gaussian(), y, Xμ, Xσ, site, G, coords,
                                        ["(Intercept)", "x"], ["(Intercept)"], :site, 1e-8)
        for lse in (-2.0, -16.0, -20.0)
            θ = [0.2, 0.35, lse, log(0.7), log(2.0)]
            K = exp.(-Dd ./ 2.0) + 1e-8I
            ref = _cs_dense_nll(y, Xμ * θ[1:2], fill(exp(2lse), G), [(site, G, K, 0.49)])
            @test _cs_rel(fit.nll(θ), ref) < (lse == -2.0 ? 1e-12 : 1e-8)
        end
    end

    # (D) _fit_two_structured_gaussian_sparse_spec (relmat + relmat), one record
    # per level of the first component. Pre-fix (rᵀWr − bᵀâ, sparse Cholesky
    # logdet H): lσ_e = −12: 1.6e-6; −16: 3.9e-3; with the quadratic alone fixed
    # the Cholesky logdet still left −0.21 at −18 (a fake hole). Post-fix
    # (penalised RSS; sparse QR of [W^{1/2}Z; B/σ] once κ > 1e5): ≤ 1e-10.
    let G = 50
        M1 = randn(rng, G, G); K1 = M1 * M1' / G + I; d1 = sqrt.(diag(K1)); K1 = K1 ./ (d1 * d1')
        M2 = randn(rng, 10, 10); K2 = M2 * M2' / 10 + I; d2 = sqrt.(diag(K2)); K2 = K2 ./ (d2 * d2')
        g1 = collect(1:G); g2 = repeat(1:10, inner = 5)
        y = 0.3 .+ 0.6 .* (cholesky(Symmetric(K1)).L * randn(rng, G)) .+
            0.5 .* (cholesky(Symmetric(K2)).L * randn(rng, 10))[g2] .+ 0.2 .* randn(rng, G)
        Xμ = ones(G, 1)
        fit = _CS._fit_two_structured_gaussian_sparse(Gaussian(), y, Xμ, g1, G, K1, g2, 10, K2,
                                                      ["(Intercept)"], :g1, :g2, 1e-8)
        for lse in (-2.0, -12.0, -16.0, -18.0)
            θ = [0.3, lse, log(0.6), log(0.5)]
            ref = _cs_dense_nll(y, fill(0.3, G), fill(exp(2lse), G),
                                [(g1, G, K1, 0.36), (g2, 10, K2, 0.25)])
            @test _cs_rel(fit.nll(θ), ref) < (lse == -2.0 ? 1e-12 : 1e-8)
        end
    end
end
