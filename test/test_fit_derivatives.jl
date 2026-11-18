# Unit tests for src/fit_derivatives.jl: `gradient(fit)`, `hessian(fit)` and the
# internals `_fit_objective`, `_fd_gradient`, `_fd_jacobian`.
# test_lane_n3_drmcontrol.jl (#971) already smoke-tests both accessors across
# the fixed-effect routes (shape, symmetry, a loose stationarity bar, the bare
# fit refusal). This file checks the numbers: agreement with independent
# central differences, the route with an exact-gradient callback, the
# non-dual-number fallback, the REML / weighted objectives, and InterruptException.
using DRModels, Test, Random, LinearAlgebra
import Distributions

const _FD = DRModels

function _fdt_data(seed = 20261020)
    rng = MersenneTwister(seed)
    n = 150
    x = randn(rng, n)
    y = 1 .+ 0.5 .* x .+ exp.(-0.3 .+ 0.2 .* x) .* randn(rng, n)
    ypois = Float64.([rand(rng, Distributions.Poisson(exp(0.3 + 0.4xi))) for xi in x])
    w = 0.5 .+ rand(rng, n)
    return (; x, y, ypois, w)
end

const _FDT = _fdt_data()
_fdt_theta(fit) = Vector{Float64}(fit.theta)
# Gradient size is judged against the objective's scale.
_fdt_small(g, fit) = norm(g) < 1e-4 * max(1.0, abs(fit.nll(_fdt_theta(fit))))

# Gaussian fixed-effect fit on a fresh copy of the data (location-scale and
# location-only), a Poisson fit, and the weighted / REML variants.
_fdt_gauss(; kw...) = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + x)), Gaussian();
                          data = (y = _FDT.y, x = _FDT.x), kw...)
_fdt_gauss_const(; kw...) = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian();
                                data = (y = _FDT.y, x = _FDT.x), kw...)
_fdt_pois(; kw...) = drm(bf(@formula(y ~ 1 + x)), Poisson(); data = (y = _FDT.ypois, x = _FDT.x), kw...)

# A real route that stores an exact gradient callback and whose objective does
# not accept dual numbers (Gamma + phylo(1 | species) Laplace).
function _fdt_phylo_fit()
    Random.seed!(20261020)
    p = 20; m = 6
    phy = random_balanced_tree(p; branch_length = 0.2)
    species = repeat(1:p, inner = m)
    n = length(species)
    x = randn(n)
    u = cholesky(Symmetric(sigma_phy_dense(phy; σ²_phy = 0.2))).L * randn(p)
    μ = exp.(0.2 .+ 0.3 .* x .+ u[species])
    y = Float64.([rand(Distributions.Gamma(4.0, μ[i] / 4.0)) for i in 1:n])
    return drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)), DRModels.Gamma();
               data = (; y, x, species), tree = phy, se = false)
end

struct _FdtInterrupter end
(::_FdtInterrupter)(θ) = throw(InterruptException())

# The callback convention of `fit.nllgrad`: fills `g`, may return nothing.
_fdt_grad_cb(nll) = function (g, θ)
    g .= _FD.ForwardDiff.gradient(nll, θ)
    return nothing
end

@testset "fit_derivatives: gradient / hessian / finite-difference helpers" begin

    @testset "_fd_gradient and _fd_jacobian on known functions" begin
        f(θ) = 3θ[1]^2 + θ[1] * θ[2] + 2θ[2]^2 - 5θ[2]
        θ = [0.7, -1.3]
        @test _FD._fd_gradient(f, θ) ≈ [6θ[1] + θ[2], θ[1] + 4θ[2] - 5] atol = 1e-7
        # a large coordinate scales its step (h = cbrt(eps) * max(|θ_j|, 1))
        θb = [1e4, -2e4]
        @test _FD._fd_gradient(f, θb) ≈ [6θb[1] + θb[2], θb[1] + 4θb[2] - 5] rtol = 1e-6
        gfun(θ) = [6θ[1] + θ[2], θ[1] + 4θ[2] - 5]
        J = _FD._fd_jacobian(gfun, θ)
        @test J ≈ [6.0 1.0; 1.0 4.0] atol = 1e-7
        @test size(J) == (2, 2)
        # non-symmetric Jacobian keeps its orientation: J[i, j] = ∂g_i / ∂θ_j
        hfun(θ) = [θ[1]^2 * θ[2], 4θ[2]]
        Jh = _FD._fd_jacobian(hfun, [2.0, 3.0])
        @test Jh ≈ [2 * 2.0 * 3.0 2.0^2; 0.0 4.0] atol = 1e-6
        # the input is not mutated
        θc = copy(θ); _FD._fd_gradient(f, θc); _FD._fd_jacobian(gfun, θc)
        @test θc == θ
    end

    @testset "gradient: Gaussian and Poisson fixed-effect fits" begin
        for (lab, fit) in (("Gaussian location-scale", _fdt_gauss()),
                           ("Gaussian location-only sigma", _fdt_gauss_const()),
                           ("Poisson", _fdt_pois()))
            @testset "$lab" begin
                g = gradient(fit)
                θ = _fdt_theta(fit)
                @test g isa Vector{Float64}
                @test length(g) == length(coef(fit))
                @test all(isfinite, g)
                @test _fdt_small(g, fit)
                @test g ≈ _FD._fd_gradient(fit.nll, θ) atol = 1e-5
                # these routes store no gradient callback, so this is the AD branch
                @test fit.nllgrad === nothing
                # gradient does not touch the fit
                @test _fdt_theta(fit) == θ
            end
        end
    end

    @testset "gradient: exact-callback route and the AD / finite-difference routes" begin
        # (a) route with fit.nllgrad (Gamma phylo Laplace): the callback value is returned
        phy = _fdt_phylo_fit()
        @test phy.nllgrad !== nothing
        θp = _fdt_theta(phy)
        gcb = zeros(length(θp)); phy.nllgrad(gcb, θp)
        gp = gradient(phy)
        @test gp == gcb
        @test gp ≈ _FD._fd_gradient(phy.nll, θp) atol = 1e-4
        @test _fdt_small(gp, phy)
        # (b) the same objective with the callback dropped: dual numbers are not
        # accepted by the Laplace objective, so it falls back to central differences
        bare = _FD._withnll(phy, phy.nll)
        @test bare.nllgrad === nothing
        @test_throws MethodError _FD.ForwardDiff.gradient(bare.nll, θp)
        @test gradient(bare) ≈ _FD._fd_gradient(bare.nll, θp) atol = 1e-5   # Laplace inner mode is warm-started, not bit-repeatable
        @test gradient(bare) ≈ gp atol = 1e-4
        # (c) forward-mode AD on a fixed-effect objective
        gfit = _fdt_gauss()
        θg = _fdt_theta(gfit)
        @test gradient(gfit) ≈ _FD.ForwardDiff.gradient(gfit.nll, θg)
        @test gradient(gfit) ≈ _FD._fd_gradient(gfit.nll, θg) atol = 1e-5
        # (d) a callback that fills g and returns nothing, and one that returns the vector
        cb = _FD._withnll(gfit, gfit.nll, _fdt_grad_cb(gfit.nll))
        @test gradient(cb) ≈ _FD._fd_gradient(gfit.nll, θg) atol = 1e-5
        cbret = _FD._withnll(gfit, gfit.nll, (g, θ) -> (g .= 0; ret = _FD.ForwardDiff.gradient(gfit.nll, θ); ret))
        @test gradient(cbret) ≈ gradient(cb)
        # a callback is trusted over the objective: a marked gradient comes back as is
        marked = _FD._withnll(gfit, gfit.nll, (g, θ) -> (g .= 1:length(θ); nothing))
        @test gradient(marked) == collect(1.0:length(θg))
    end

    @testset "hessian: symmetric, matches vcov and the Jacobian of the gradient" begin
        for (lab, fit) in (("Gaussian location-scale", _fdt_gauss()),
                           ("Gaussian location-only sigma", _fdt_gauss_const()),
                           ("Poisson", _fdt_pois()))
            @testset "$lab" begin
                H = hessian(fit)
                p = length(coef(fit))
                @test H isa Matrix{Float64}
                @test size(H) == (p, p)
                @test H == H'
                @test isposdef(Symmetric(H))
                @test inv(H) ≈ vcov(fit) rtol = 1e-6
                Jg = _FD._fd_jacobian(θ -> _FD.ForwardDiff.gradient(fit.nll, θ), _fdt_theta(fit))
                @test H ≈ (Jg + Jg') / 2 rtol = 1e-5
                @test H ≈ _FD._fd_jacobian(θ -> _FD._fd_gradient(fit.nll, θ), _fdt_theta(fit)) rtol = 1e-3
                # `control = DRMControl(se = false)` leaves vcov as NaN; hessian still works
                nose = lab == "Poisson" ? _fdt_pois(control = DRMControl(se = false)) :
                       lab == "Gaussian location-only sigma" ? _fdt_gauss_const(control = DRMControl(se = false)) :
                       _fdt_gauss(control = DRMControl(se = false))
                @test all(isnan, vcov(nose))
                @test hessian(nose) ≈ H rtol = 1e-6
            end
        end
        # dual-number-free route with a callback: finite differences of the callback
        phy = _fdt_phylo_fit()
        θp = _fdt_theta(phy)
        gfun = x -> (g = zeros(length(x)); phy.nllgrad(g, x); g)
        Hp = hessian(phy)
        Jp = _FD._fd_jacobian(gfun, θp)
        @test Hp ≈ (Jp + Jp') / 2 rtol = 1e-4
        @test issymmetric(Hp)
        # same objective, no callback: finite differences of finite-difference gradients
        Hb = hessian(_FD._withnll(phy, phy.nll))
        @test issymmetric(Hb)
        @test all(isfinite, Hb)
    end

    @testset "objective on weighted fits" begin
        wfit = _fdt_gauss_const(weights = _FDT.w)
        θ = _fdt_theta(wfit)
        X = hcat(ones(length(_FDT.x)), _FDT.x)
        μ = X * θ[1:2]
        σ = exp(θ[3])
        by_hand = -sum(_FDT.w .* Distributions.logpdf.(Distributions.Normal.(μ, σ), _FDT.y))
        @test wfit.nll(θ) ≈ by_hand rtol = 1e-10
        @test wfit.nll(θ) ≈ -loglik(wfit)
        # it differs from the unweighted objective at the same θ, so gradient is not the unweighted one
        unw = -sum(Distributions.logpdf.(Distributions.Normal.(μ, σ), _FDT.y))
        @test !isapprox(by_hand, unw; rtol = 1e-3)
        g = gradient(wfit)
        @test _fdt_small(g, wfit)
        @test g ≈ _FD._fd_gradient(θ -> -sum(_FDT.w .* Distributions.logpdf.(
                  Distributions.Normal.(X * θ[1:2], exp(θ[3])), _FDT.y)), θ) atol = 1e-5
        @test inv(hessian(wfit)) ≈ vcov(wfit) rtol = 1e-6
        # weighted Poisson
        wp = _fdt_pois(weights = _FDT.w)
        θp = _fdt_theta(wp)
        pois_by_hand = -sum(_FDT.w .* Distributions.logpdf.(Distributions.Poisson.(exp.(X * θp)), _FDT.ypois))
        @test wp.nll(θp) ≈ pois_by_hand rtol = 1e-10
        @test _fdt_small(gradient(wp), wp)
    end

    @testset "objective on REML fits (docs: the negative REML criterion)" begin
        rfit = _fdt_gauss_const(method = :REML)
        @test rfit.estim_method === :REML
        θ = _fdt_theta(rfit)
        X = hcat(ones(length(_FDT.x)), _FDT.x)
        σ2 = exp(2θ[3])
        # REML criterion by hand: ML nll at θ plus half log|X'X / σ²| minus (p/2) log 2π
        ml_by_hand = -sum(Distributions.logpdf.(Distributions.Normal.(X * θ[1:2], sqrt(σ2)), _FDT.y))
        reml_by_hand = ml_by_hand + 0.5 * logdet(X' * X / σ2) - log(2π)
        @test reml_by_hand ≈ -rfit.reml_loglik rtol = 1e-8       # the hand formula matches the stored REML value
        # What the objective returns on a REML fit. The docstring says the negative REML
        # criterion; today `fit.nll` stored on this route is the ML negative log-likelihood.
        @test rfit.nll(θ) ≈ ml_by_hand rtol = 1e-10
        @test_broken rfit.nll(θ) ≈ -rfit.reml_loglik
        # So gradient(fit) is the ML gradient at the REML estimate, not ~0.
        g = gradient(rfit)
        @test g ≈ _FD._fd_gradient(rfit.nll, θ) atol = 1e-5      # consistent with its own objective
        @test_broken _fdt_small(g, rfit)                          # stationarity of the REML criterion
        @test inv(hessian(rfit)) ≈ vcov(rfit) rtol = 1e-6         # vcov is the inverse of that same ML-objective Hessian
    end

    @testset "_fit_objective refuses a fit with no stored objective" begin
        bare = DrmFit(Gaussian(), [:mu => 1:1], [:mu => ["(Intercept)"]], [0.0], fill(1.0, 1, 1), 0.0, 1,
                      true, Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Vector{Float64}}())
        @test bare.nll === nothing
        for what in ("gradient", "hessian")
            err = try _FD._fit_objective(bare, what); nothing catch e; e end
            @test err isa ArgumentError
            msg = sprint(showerror, err)
            @test occursin("$what(fit)", msg)
            @test occursin("fit.nll", msg)
            @test occursin("did not store its objective", msg)
            @test occursin("$what cannot be evaluated", msg)
        end
        for (fn, what) in ((gradient, "gradient"), (hessian, "hessian"))
            err = try fn(bare); nothing catch e; e end
            @test err isa ArgumentError
            @test occursin(what, sprint(showerror, err))
        end
        # with an objective stored, _fit_objective returns it unchanged
        fit = _fdt_gauss_const()
        @test _FD._fit_objective(fit, "gradient") === fit.nll
    end

    @testset "InterruptException is rethrown, not swallowed" begin
        fit = _fdt_gauss_const()
        boom(θ) = throw(InterruptException())
        ifit = _FD._withnll(fit, boom)
        @test_throws InterruptException gradient(ifit)
        @test_throws InterruptException hessian(ifit)
        # a callable struct, as a user objective would be
        cfit = _FD._withnll(fit, _FdtInterrupter())
        @test_throws InterruptException gradient(cfit)
        @test_throws InterruptException hessian(cfit)
        # also when the interrupt arrives from the fallback evaluation (non-dual objective)
        late(θ) = eltype(θ) === Float64 ? throw(InterruptException()) : throw(MethodError(late, (θ,)))
        lfit = _FD._withnll(fit, late)
        @test_throws InterruptException gradient(lfit)
        @test_throws InterruptException hessian(lfit)
        # a callback route: Hessian differences the gradient callback, and an interrupt there propagates
        gcb = _FD._withnll(fit, boom, (g, θ) -> throw(InterruptException()))
        @test_throws InterruptException gradient(gcb)
        @test_throws InterruptException hessian(gcb)
    end
end
