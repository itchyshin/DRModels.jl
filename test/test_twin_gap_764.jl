# #764: animal(1|id) Gaussian on a ONE-record-per-individual pedigree (an
# animal model's usual shape) diverged: logLik ~ 1e253, σ_e → 0, `converged =
# true`, while a dense reassembly of the SAME likelihood at the same point (and
# a 1-D profile) is bounded and interior. Root cause: the homoscedastic
# Woodbury quadratic `q1 - dot(C, Mfac \ C)` in `_fit_structured_gaussian`
# (src/gaussian_structured.jl) is a difference of two O(1/σ_e²) terms that
# cancel to an O(1) residual as σ_e → 0 — with one observation per structured
# level that residual is the WHOLE quadratic form, so it is lost to rounding.
# This is the #548/#835/#837/#838 cancellation class, not genuine
# unboundedness: a fit must never report `converged = true` with a
# non-finite/absurd logLik.
using DRModels
using Test, Random, LinearAlgebra, Statistics

@testset "#764: animal(1|id) one-record-per-individual — no cancellation blowup" begin
    Random.seed!(1)   # this seed reliably diverges under the pre-fix Woodbury quadratic
    G = 300   # individuals, ONE record each — the animal-model shape in #764
    Mraw = randn(G, G); A0 = Mraw * Mraw' / G + I
    d = sqrt.(diag(A0)); A = A0 ./ (d * d')   # additive-relatedness-like matrix
    x = randn(G)                              # a fixed covariate (like Cohort)
    σ = 0.6; σs = 0.5
    u = σs .* (cholesky(Symmetric(A)).L * randn(G))
    y = 0.3 .+ 0.5 .* x .+ u .+ σ .* randn(G)
    id = collect(1:G)
    data = (; y, x, id)

    # Independent stable reference: dense assembly of the SAME marginal
    # V = σ_e² I + σs² A, no Woodbury — this is what the true likelihood is.
    Xμ = hcat(ones(G), x)
    Kfac0 = cholesky(Symmetric(A))
    function nll_dense_ref(βμ, lresid, lσs)
        ημ = Xμ * βμ
        σ2 = exp(2 * lresid); σs2 = exp(2 * lσs)
        V = σ2 .* Matrix{Float64}(I, G, G) .+ σs2 .* A
        Vfac = cholesky(Symmetric(V); check = false)
        r = y .- ημ
        0.5 * (logdet(Vfac) + dot(r, Vfac \ r)) + 0.5 * G * log(2π)
    end

    fit = drm(bf(@formula(y ~ x + animal(1 | id)), @formula(sigma ~ 1)), Gaussian();
              data = data, A = A)

    # 1) The fit must never claim convergence with a non-finite/absurd logLik.
    @test isfinite(loglik(fit))
    @test fit.converged
    @test loglik(fit) < 0    # a sane Gaussian logLik on this scale of data/n

    # 2) The fitted logLik must agree with the independent dense reference at
    # the SAME parameter point (catches silent cancellation even if the
    # optimizer still lands somewhere plausible-looking).
    βμ̂ = coef(fit, :mu)
    lresid̂ = coef(fit, :sigma)[1]
    lσŝ = log(re_sd(fit)[:id])
    ref = nll_dense_ref(βμ̂, lresid̂, lσŝ)
    @test -loglik(fit) ≈ ref atol = 1e-6 rtol = 1e-8

    # 3) Recovery: with G = 300 informative pairs this should land in the
    # right ballpark (loose tolerances — this is a numerics regression test,
    # not a recovery/power test).
    @test exp(lresid̂) ≈ σ atol = 0.25
    @test re_sd(fit)[:id] ≈ σs atol = 0.3
end
