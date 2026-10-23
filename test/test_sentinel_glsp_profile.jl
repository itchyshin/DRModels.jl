# test_sentinel_glsp_profile.jl — `_glsp_profile_ci` must not turn a failed solve into an endpoint.
#
# Two opposite failures used to fabricate an endpoint:
#   * ML path: the route NLL returns the 1e18 sentinel on failure; 1e18 - nll_min - thr > 0
#     read as "crossed", so the bisection landed on the edge of the failed region.
#   * REML path: a failed joint mode is Inf (or a sub-fit throws); Inf read as "not
#     crossed", so the arm was reported as the boundary (SD 0 below / Inf above).
# A failed evaluation is now a third outcome. It backs off toward the last good point and,
# if the crossing cannot be bracketed, the arm is reported UNRESOLVED as NaN (the same
# "no interval" value the profile route already uses for the correlation bound).

using DRModels
using Test, Random, LinearAlgebra
import Distributions

# One free-parameter-free profile (θ̂ has length 1 ⇒ the profile IS the objective), so the
# injected objective fully controls what each evaluation returns. nll(v) = ½ (v/s)² has its
# χ²₁ crossing at v = ±1.959964·s.
const _z95 = sqrt(Distributions.quantile(Distributions.Chisq(1), 0.95))
_ci(nll) = DRModels._glsp_profile_ci(nll, v -> zeros(length(v)), [0.0], 1)

@testset "_glsp_profile_ci: clean objective unchanged" begin
    ci = _ci(v -> 0.5 * (v[1] / 2)^2)
    @test ci.sd_lo ≈ exp(-2 * _z95) atol = 1e-5
    @test ci.sd_hi ≈ exp(2 * _z95)  atol = 1e-4
end

@testset "_glsp_profile_ci: genuine boundary is still a boundary" begin
    ci = _ci(v -> 0.01 * v[1]^2)            # never reaches the threshold within ±8
    @test ci.sd_lo == 0.0
    @test ci.sd_hi == Inf
end

@testset "_glsp_profile_ci: 1e18 sentinel is not a crossing" begin
    # True crossing at v = 4·1.96 = 7.84, but the solve is a sentinel for v > 3.
    ci = _ci(v -> v[1] > 3 ? 1e18 : 0.5 * (v[1] / 4)^2)
    @test isnan(ci.sd_hi)                    # unresolved, NOT exp(3) (the cliff edge)
    @test ci.sd_lo ≈ exp(-4 * _z95) rtol = 1e-4   # the clean arm is untouched
end

@testset "_glsp_profile_ci: sentinel beyond a real crossing still resolves it" begin
    # Crossing at 1.96, sentinel only past 5: the cap fails but the backed-off probe crosses.
    ci = _ci(v -> v[1] > 5 ? 1e18 : 0.5 * v[1]^2)
    @test ci.sd_hi ≈ exp(_z95) rtol = 1e-4
    @test ci.sd_lo ≈ exp(-_z95) atol = 1e-5
end

@testset "_glsp_profile_ci: Inf (failed mode) is not a boundary" begin
    ci = _ci(v -> v[1] > 3 ? Inf : 0.5 * (v[1] / 4)^2)
    @test isnan(ci.sd_hi)                    # NOT Inf (a fabricated boundary)
    @test ci.sd_lo ≈ exp(-4 * _z95) rtol = 1e-4
end

@testset "_glsp_profile_ci: a throwing solve is not a boundary" begin
    ci = _ci(v -> v[1] < -3 ? error("ill-conditioned") : 0.5 * (v[1] / 4)^2)
    @test isnan(ci.sd_lo)                    # NOT 0.0
    @test ci.sd_hi ≈ exp(4 * _z95) rtol = 1e-4
end

@testset "_glsp_profile_ci: failure inside the bisection is flagged" begin
    # The cap crosses (v = 8) but every point in (2, 7) fails: no honest endpoint exists.
    ci = _ci(v -> 2 < v[1] < 7 ? 1e18 : 0.5 * (v[1] / 3)^2)
    @test isnan(ci.sd_hi)                    # crossing (v≈5.9) lies inside the failed band
end

# Real fixtures (values recorded from the pre-fix code; profile CIs must not move).
@testset "σ-phylo profile CIs on real fixtures are unchanged" begin
    Random.seed!(202606121)
    p = 64; m = 4; n = p * m
    phy = random_balanced_tree(p; branch_length = 0.30)
    C   = sigma_phy_dense(phy; σ²_phy = 1.0)
    LC  = cholesky(Symmetric(C)).L
    u_mu    = 0.70 .* (LC * randn(p))
    u_sigma = 0.60 .* (LC * randn(p))
    species = repeat(1:p, inner = m)
    x = randn(n); βμ = [0.5, 0.3]; βψ = [0.2]
    y  = [βμ[1] + βμ[2]*x[i] + u_mu[species[i]] +
          exp(βψ[1] + u_sigma[species[i]]) * randn() for i in 1:n]
    fit = drm(bf(@formula(y ~ x + phylo(1 | species)),
                 @formula(sigma ~ phylo(1 | species))),
              Gaussian(); data = (; y, x, species), tree = phy, profile_ci = true)
    @test fit.scales[:profile_ci_sd_sigma] ≈ [0.5354971505254494, 0.9373804004182825] atol = 1e-8
    @test fit.scales[:profile_ci_sd_mu]    ≈ [0.5181310093146736, 0.9679140064350094] atol = 1e-8

    # Absent signal: the genuine boundary [0, Inf] must survive.
    Random.seed!(202606122)
    phy = random_balanced_tree(p; branch_length = 0.30)
    x = randn(n)
    y = [0.4 + 0.2*x[i] + exp(0.1) * randn() for i in 1:n]
    fit = drm(bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | species))),
              Gaussian(); data = (; y, x, species), tree = phy, profile_ci = true)
    @test fit.scales[:profile_ci_sd_sigma] == [0.0, Inf]
end
