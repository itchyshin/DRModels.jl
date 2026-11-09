using DRModels, Test, LinearAlgebra
import DelimitedFiles, Logging

# End-to-end test of the post-fit BLUP guard `_blup_vinv_r` (wave 8): a real `drm` fit that
# reaches the guard with a genuinely non-positive-definite marginal V must RETURN (no throw)
# with NaN BLUPs, one warning (`maxlog = 1`) and `fit.scales[:cond_blup_unavailable] == [1.0]`.
# The helper alone is unit-tested in test_cholesky_guards.jl / test_robust_entry_failed_fits.jl.
#
# Route: two structured Gaussian intercepts, `relmat(1 | g1) + animal(1 | g2)`
# (`_fit_two_structured_gaussian`). That route checks only the SYMMETRY of K and A, because a
# singular-but-PSD relatedness matrix is legitimate; V-level positive definiteness is enforced by
# `cholesky(check = false)` in the objective, which returns the 1e18 failed-objective sentinel.
# With an INDEFINITE K the default start already gives a non-PD V, so the objective is the flat
# sentinel everywhere the optimiser looks, L-BFGS stops at the start, and the post-fit BLUP step
# sees a non-PD V at theta-hat. The fit is therefore also reported as failed (`converged = false`,
# logLik = -1e18, a separate warning); that is the honest reachable case, not a converged fit
# with a late singular V (the objective rejects any non-PD V before a converged theta-hat).
# Committed data: test/fixtures/blup_nan_e2e/ (generator gen_blup_nan_e2e.jl).

const _BNE_DIR = joinpath(@__DIR__, "fixtures", "blup_nan_e2e")
const _BNE_RE = r"BLUPs\) are NaN"

function _bne_data()
    m, h = DelimitedFiles.readdlm(joinpath(_BNE_DIR, "data.csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x = Float64.(col("x")),
            g1 = Int.(col("g1")), g2 = Int.(col("g2")))
end
_bne_K(name) = Matrix{Float64}(DelimitedFiles.readdlm(joinpath(_BNE_DIR, name), ','))

const _BNE_F = bf(@formula(y ~ x + relmat(1 | g1) + animal(1 | g2)), @formula(sigma ~ 1))

# Fit under a test logger; returns (fit-or-exception, all log records).
function _bne_fit(K, A)
    d = _bne_data()
    logs, res = Test.collect_test_logs(; min_level = Logging.Warn) do
        try
            drm(_BNE_F, Gaussian(); data = d, K = K, A = A)
        catch e
            e
        end
    end
    return res, logs
end
_bne_blup_logs(logs) = filter(l -> l.level == Logging.Warn && occursin(_BNE_RE, string(l.message)), logs)

# Marginal V at theta-hat, rebuilt independently of the library's BLUP code
# (theta = [beta (2); log sigma; log sd(g1); log sd(g2)]).
function _bne_V(fit, K, A)
    d = _bne_data(); n = length(d.y); G = 6
    Z(g) = [g[i] == j ? 1.0 : 0.0 for i in 1:n, j in 1:G]
    Z1, Z2 = Z(d.g1), Z(d.g2)
    th = fit.theta
    return exp(2th[3]) * I(n) + exp(2th[4]) * (Z1 * K * Z1') + exp(2th[5]) * (Z2 * A * Z2')
end

@testset "BLUP guard end to end: indefinite K reaches _blup_vinv_r with non-PD V" begin
    Kpd, Kind = _bne_K("K_pd.csv"), _bne_K("K_indef.csv")
    @test issymmetric(Kind) && eigmin(Symmetric(Kind)) < -1          # genuinely indefinite input
    @test isposdef(Symmetric(Kpd))

    fit, logs = _bne_fit(Kind, Kpd)
    @test !(fit isa Exception)                                       # the fit returns, no throw
    @test !(fit isa Exception) && begin
        # V at the returned theta-hat is genuinely not positive definite
        V = _bne_V(fit, Kind, Kpd)
        @test !issuccess(cholesky(Symmetric(V); check = false))
        @test eigmin(Symmetric(V)) < 0
        # every BLUP is NaN, for both structured components
        @test sort(collect(keys(fit.ranef))) == [:g1, :g2]
        @test all(v -> length(v) == 6 && all(isnan, v), values(fit.ranef))
        # the flag survives on the fit, after the one-time warning has fired
        @test haskey(fit.scales, :cond_blup_unavailable)
        @test fit.scales[:cond_blup_unavailable] == [1.0]
        # honest status: this fit is the failed-objective sentinel, not a converged optimum
        @test fit.converged == false
        @test fit.loglik == -1.0e18
        true
    end
    # exactly one BLUP warning, carrying maxlog = 1
    bl = _bne_blup_logs(logs)
    @test length(bl) == 1
    @test length(bl) == 1 && Dict(bl[1].kwargs)[:maxlog] == 1

    # Control: a PD K on the same route and data. No BLUP warning, finite BLUPs, no flag.
    fitpd, logspd = _bne_fit(Kpd, Kpd)
    @test !(fitpd isa Exception)
    @test isempty(_bne_blup_logs(logspd))
    @test !(fitpd isa Exception) && begin
        @test all(v -> length(v) == 6 && all(isfinite, v), values(fitpd.ranef))
        @test !haskey(fitpd.scales, :cond_blup_unavailable)
        @test fitpd.converged == true
        @test isfinite(fitpd.loglik) && fitpd.loglik > -1e6
        @test isposdef(Symmetric(_bne_V(fitpd, Kpd, Kpd)))
        true
    end
end
