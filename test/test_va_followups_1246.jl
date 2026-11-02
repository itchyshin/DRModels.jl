# DRModels.jl #1246 (items 4 and 6 only): `_va_scored_sigma` honours the
# `:drm_quiet_boundary` guard like `_va_inner_converged`, and `_logfactorial` is O(1).
using DRModels
using Test, Logging

const DVI = DRModels

@testset "#1246 item 6: _logfactorial is O(1) and agrees with the sum" begin
    for k in 0:200
        @test DVI._logfactorial(k) ≈ sum(log, 2:k; init = 0.0) rtol = 1e-12
    end
    @test DVI._logfactorial(0) == 0.0
    @test DVI._logfactorial(1) == 0.0
    t = @elapsed v = DVI._logfactorial(10^9)
    @test isfinite(v) && v > 0
    @test t < 1.0
end

@testset "#1246 item 4: _va_scored_sigma clamp warning is quietable" begin
    b = 20.0
    Xσ = ones(5, 1)
    βσ = [b + 5.0]                       # η_σ past the clamp
    @test_logs (:warn,) DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    loud = with_logger(NullLogger()) do
        DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    end
    quiet = task_local_storage(:drm_quiet_boundary, true) do
        @test_logs DVI._va_scored_sigma(Xσ, βσ, b, "_fit_nb2_ranef_va")
    end
    @test quiet == loud == fill(exp(b), 5)
    # inside the band: silent either way, same value
    @test_logs DVI._va_scored_sigma(Xσ, [1.0], b, "_fit_nb2_ranef_va") == fill(exp(1.0), 5)
end
