# DJ1: hetero and fixed Laplace derivative blocks in src/sparse_laplace_glmm.jl
# that no test calls. Oracles are Distributions densities. The crossed-fit
# drivers that open drafts rewrite (`_fit_crossed_mean_laplace` and the AGHQ
# twins) are not called; only the component-count refusal in front of them is.
using DRModels, Test, LinearAlgebra
import Distributions
import ForwardDiff
import SpecialFunctions: loggamma

const _ATOL = 1e-7
const _RTOL = 1e-7

_d1(f, x) = ForwardDiff.derivative(f, x)
_d2(f, x) = ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), x)
_d3(f, x) = ForwardDiff.derivative(t -> _d2(f, t), x)

function _nb2_nll(y, η, logr)
    r = exp(logr)
    μ = exp(clamp(η, -30.0, 30.0))
    p = r / (r + μ)
    return -Distributions.logpdf(Distributions.NegativeBinomial(r, p), Int(round(y)))
end

function _gamma_nll(y, η, ψ)
    α = exp(-2 * ψ)
    μ = exp(clamp(η, -30.0, 30.0))
    return -Distributions.logpdf(Distributions.Gamma(α, μ / α), y)
end

function _beta_nll(y, η, logφ)
    φ = exp(logφ)
    μ = 1 / (1 + exp(-clamp(η, -30.0, 30.0)))
    return -Distributions.logpdf(Distributions.Beta(μ * φ, (1 - μ) * φ), y)
end

@testset "NB2 fixed third derivative and dispersion score" begin
    y = 3.0
    η = 0.35
    r = 2.4
    lconst = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
    aux = (y = [y], size = r, lconst = [lconst])
    v, d1, d2, d3 = DRModels._laplace_v123(Val(:nb2_fixed), aux, 1, η)
    @test v ≈ _nb2_nll(y, η, log(r)) atol = _ATOL rtol = _RTOL
    @test d1 ≈ _d1(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    @test d2 ≈ _d2(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    @test d3 ≈ _d3(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    @test DRModels._laplace_d3(Val(:nb2_fixed), aux, 1, η) ≈ d3 atol = _ATOL rtol = _RTOL
    nv = DRModels._laplace_nuisance_value(Val(:nb2_fixed), aux, 1, η)
    nd1 = DRModels._laplace_nuisance_d1(Val(:nb2_fixed), aux, 1, η)
    nd2 = DRModels._laplace_nuisance_d2(Val(:nb2_fixed), aux, 1, η)
    # These three are ∂/∂log(r), not ∂/∂ψ. The hetero kernel below carries the −2 chain.
    # nuisance_value is ∂nll/∂log(r). nuisance_d1 and nuisance_d2 are the
    # first and second derivatives of that score with respect to η.
    score_logr(e) = _d1(logr -> _nb2_nll(y, e, logr), log(r))
    @test nv ≈ score_logr(η) atol = _ATOL rtol = _RTOL
    @test nd1 ≈ _d1(score_logr, η) atol = _ATOL rtol = _RTOL
    @test nd2 ≈ _d2(score_logr, η) atol = _ATOL rtol = _RTOL
end

@testset "NB2 and Gamma hetero v123 match the density in both axes" begin
    y = 4.0
    η = -0.15
    ψ = -0.2
    r = exp(-2 * ψ)
    lconst = loggamma(y + r) - loggamma(r) - loggamma(y + 1)
    aux = (y = [y], size = [r], lconst = [lconst])
    v, d1, d2, d3, nv, nd1, nd2 = DRModels._laplace_v123_nuisance(Val(:nb2_hetero), aux, 1, η)
    @test v ≈ _nb2_nll(y, η, log(r)) atol = _ATOL rtol = _RTOL
    @test d1 ≈ _d1(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    @test d2 ≈ _d2(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    @test d3 ≈ _d3(e -> _nb2_nll(y, e, log(r)), η) atol = _ATOL rtol = _RTOL
    # ψ = log σ and r = exp(−2ψ), so the hetero score is the fixed score
    # with the chain factor, then differentiated in η.
    score_ψ(e) = _d1(p -> _nb2_nll(y, e, -2 * p), ψ)
    @test nv ≈ score_ψ(η) atol = _ATOL rtol = _RTOL
    @test nd1 ≈ _d1(score_ψ, η) atol = _ATOL rtol = _RTOL
    @test nd2 ≈ _d2(score_ψ, η) atol = _ATOL rtol = _RTOL
    vv, dd1, dd2, dd3 = DRModels._laplace_v123(Val(:nb2_hetero), aux, 1, η)
    @test (vv, dd1, dd2, dd3) == (v, d1, d2, d3)

    yg = 1.6
    α = exp(-2 * ψ)
    μg = exp(η)
    lc = α * log(α) - loggamma(α) + (α - 1) * log(yg)
    auxg = (y = [yg], shape = [α], lconst = [lc])
    gv, gd1, gd2, gd3, gnv, gnd1, gnd2 = DRModels._laplace_v123_nuisance(Val(:gamma_hetero), auxg, 1, η)
    @test gv ≈ _gamma_nll(yg, η, ψ) atol = _ATOL rtol = _RTOL
    @test gd1 ≈ _d1(e -> _gamma_nll(yg, e, ψ), η) atol = _ATOL rtol = _RTOL
    @test gd2 ≈ _d2(e -> _gamma_nll(yg, e, ψ), η) atol = _ATOL rtol = _RTOL
    @test gd3 ≈ _d3(e -> _gamma_nll(yg, e, ψ), η) atol = _ATOL rtol = _RTOL
    score_γ(e) = _d1(p -> _gamma_nll(yg, e, p), ψ)
    @test gnv ≈ score_γ(η) atol = _ATOL rtol = _RTOL
    @test gnd1 ≈ _d1(score_γ, η) atol = _ATOL rtol = _RTOL
    @test gnd2 ≈ _d2(score_γ, η) atol = _ATOL rtol = _RTOL
end

@testset "Beta fixed v123 matches the Beta density" begin
    y = 0.3
    η = 0.4
    φ = 6.0
    μ = 1 / (1 + exp(-η))
    a = μ * φ
    b = (1 - μ) * φ
    aux = (y = [y], precision = φ, ylogit = [log(y) - log1p(-y)],
           lgammaφ = loggamma(φ))
    v, d1, d2, d3 = DRModels._laplace_v123(Val(:beta_fixed), aux, 1, η)
    @test v ≈ _beta_nll(y, η, log(φ)) atol = _ATOL rtol = _RTOL
    @test d1 ≈ _d1(e -> _beta_nll(y, e, log(φ)), η) atol = _ATOL rtol = _RTOL
    @test d2 ≈ _d2(e -> _beta_nll(y, e, log(φ)), η) atol = _ATOL rtol = _RTOL
    @test d3 ≈ _d3(e -> _beta_nll(y, e, log(φ)), η) atol = 1e-6 rtol = 1e-6
    @test DRModels._laplace_value(Val(:beta_fixed), aux, 1, η) ≈ v atol = _ATOL
    @test a + b ≈ φ
end

@testset "fixed crossed Laplace refuses a component count other than two" begin
    y = [1.0, 2.0]
    X = ones(2, 1)
    one = [(ones(2, 1), [1, 1], 2, "g")]
    @test_throws ErrorException DRModels._fit_nb2_fixed_crossed_laplace(
        DRModels.NegBinomial2(), y, 1.5, X, one, ["(Intercept)"], 1e-4)
    @test_throws ErrorException DRModels._fit_gamma_fixed_crossed_laplace(
        DRModels.Gamma(), y, 1.5, X, one, ["(Intercept)"], 1e-4)
    @test_throws ErrorException DRModels._fit_beta_fixed_crossed_laplace(
        DRModels.Beta(), [0.2, 0.4], 4.0, X, one, ["(Intercept)"], 1e-4)
end
