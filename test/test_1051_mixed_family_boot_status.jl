using DRModels
using Test
using StableRNGs

# #1051: the cross-family bootstrap dropped failed refits silently. It must
# report a status (and warn) on the same rules as `_finish_bootstrap_status`.

@testset "#1051 fit_mixed_family bootstrap status" begin
    good(n) = [(; rho_latent = 0.40 + 0.01 * i, converged = true, loglik = -100.0) for i in 1:n]

    @testset "clean run: status bootstrap, no warning" begin
        fits = Any[good(12);]
        st = @test_logs DRModels._mf_boot_status(12, 12)
        @test st == "bootstrap"
    end

    @testset "any dropped refit: bootstrap_incomplete + warning, CI still returned" begin
        bad = [(; rho_latent = 0.9, converged = false, loglik = -100.0) for _ in 1:3]
        fits = Any[good(12); bad; nothing]
        ci, kept = DRModels._mf_boot_ci(fits, 16, 0.95)
        @test kept == 12 && all(isfinite, ci)
        st = @test_logs (:warn, r"dropping 4 of 16 refits") DRModels._mf_boot_status(16, kept)
        @test st == "bootstrap_incomplete"
    end

    @testset "too few draws for a CI: bootstrap_unavailable" begin
        @test DRModels._mf_boot_status(1, 1) == "bootstrap_unavailable"
        # existing floor (max(10, B÷2)) leaves the CI NaN -> not a clean interval
        @test DRModels._mf_boot_status(5, 5) == "bootstrap_unavailable"
    end

    @testset "end to end: fields are reported" begin
        rng = StableRNG(7)
        n = 100
        X = hcat(ones(n), randn(rng, n)); u = randn(rng, n)
        y1 = X * [0.5, 0.8] .+ 0.7 .* u .+ 0.5 .* randn(rng, n)
        y2 = Float64[rand(rng) < 0.5 ? 1.0 : 0.0 for _ in 1:n] .+ 0.0
        y2 = Float64.(round.(exp.(0.3 .+ 0.6 .* u) .* 0.8))
        f = DRModels.fit_mixed_family(y1 = y1, X1 = X, fam1 = Gaussian(), y2 = y2, X2 = X,
                                      fam2 = DRModels.Poisson(), confint = true, B = 12,
                                      rng = StableRNG(1))
        @test f.n_boot_kept + f.n_boot_failed == 12
        @test f.boot_status in ("bootstrap", "bootstrap_incomplete", "bootstrap_unavailable")
        f0 = DRModels.fit_mixed_family(y1 = y1, X1 = X, fam1 = Gaussian(), y2 = y2, X2 = X,
                                       fam2 = DRModels.Poisson(), confint = false)
        @test f0.boot_status == "none" && f0.n_boot_failed == 0
    end
end
