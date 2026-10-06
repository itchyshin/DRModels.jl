# #714 / Class-1 twin parity: `marginal = :Laplace` on an ordinary `(1 | g)` for
# Student() (ordinary_laplace.jl, `_fit_student_ordinary_laplace`).
#
# In-process relationship tests (D-277), no machine-pinned constants:
#   (1) the fitted log-likelihood IS the Laplace approximation: an independent
#       per-group scalar Newton + curvature computation here (Distributions TDist,
#       ForwardDiff derivatives) reproduces −nll at θ̂ and at a perturbed θ, and
#       θ̂ is a stationary point of it;
#   (2) one-point adaptive quadrature IS the Laplace approximation: the default
#       route's per-group AGHQ objective at K = 1 equals the Laplace objective at
#       the same θ, to 1e-8;
#   (3) guards (D-273): the default route is still tagged :LA, explicit
#       `marginal = :LA` equals it bit for bit, and its optimum is NOT the Laplace
#       objective's;
#   (4) every out-of-scope model is refused with this route's message;
#   (5) the vcov is finite when se = true; se = false gives NaN.
# The same-target numbers against native drmTMB live in
# docs/dev-log/evidence/class1-laplace-parity/.
using DRModels
using Test, Random, LinearAlgebra
import Distributions as Dist
import ForwardDiff

const SOL = DRModels

function _sol_sim(; seed, G = 20, m = 8, sd_g = 0.6, σ = 0.5, df = 5)
    rng = Random.Xoshiro(seed)
    g = repeat(1:G, inner = m)
    x = randn(rng, G * m)
    b = sd_g .* randn(rng, G)
    y = 0.3 .+ 0.4 .* x .+ b[g] .+ σ .* rand(rng, Dist.TDist(df), G * m)
    return (y = y, x = x, g = ["g$(lpad(k, 2, '0'))" for k in g])
end

# θ = [β0, β1, log σ, η_ν, log σ_b], ν = 2 + exp(η_ν).
function _sol_reference_nll(d, θ)
    β0, β1, lσ, ην, lsb = θ
    ν = 2 + exp(ην); σb = exp(lsb)
    total = zero(eltype(θ))
    for lev in unique(d.g)
        idx = findall(==(lev), d.g)
        h(b) = sum(Dist.logpdf(Dist.TDist(ν), (d.y[i] - (β0 + β1 * d.x[i] + b)) / exp(lσ)) - lσ
                   for i in idx) + Dist.logpdf(Dist.Normal(0, σb), b)
        b = 0.0
        for _ in 1:200
            d1 = ForwardDiff.derivative(h, b)
            d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), b)
            step = d1 / d2
            b -= step
            abs(step) < 1e-13 && break
        end
        d2 = ForwardDiff.derivative(t -> ForwardDiff.derivative(h, t), b)
        total += h(b) + 0.5 * log(2π) - 0.5 * log(-d2)
    end
    return -total
end

const _SOL_F = bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1))

@testset "Student ordinary (1 | g) marginal = :Laplace (#714)" begin
    d = _sol_sim(seed = 20275905)
    fit = drm(_SOL_F, Student(); data = d, marginal = :Laplace, se = false)

    @testset "logLik is the Laplace approximation, θ̂ its optimum" begin
        @test fit.marginal === :Laplace
        @test fit.converged
        θ̂ = copy(fit.theta)
        @test length(θ̂) == 5 && dof(fit) == 5
        @test loglik(fit) ≈ -_sol_reference_nll(d, θ̂) atol = 1e-7
        θp = θ̂ .+ [0.05, -0.03, 0.04, 0.2, 0.1]
        @test fit.nll(θp) ≈ _sol_reference_nll(d, θp) atol = 1e-7
        @test maximum(abs, ForwardDiff.gradient(θ -> _sol_reference_nll(d, θ), θ̂)) < 1e-3
    end

    @testset "one-point adaptive quadrature equals the Laplace objective" begin
        rhs = Dict(_SOL_F.forms)
        fixed_mu, re, _, _ = SOL._split_ranef(rhs[:mu])
        gidx, G = SOL._group_index(d.g)
        y, Xμ, nmμ = SOL._design(_SOL_F.response, fixed_mu, d)
        _, Xσ, nmσ = SOL._design(_SOL_F.response, rhs[:sigma], d)
        _, Xν, nmν = SOL._design(_SOL_F.response, SOL.ConstantTerm(1), d)
        k1 = SOL._fit_student_ranef(Student(), y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, :g, 1e-8; K = 1)
        for θ in (copy(fit.theta), fit.theta .+ [0.05, -0.03, 0.04, 0.2, 0.1])
            @test k1.nll(θ) ≈ fit.nll(θ) atol = 1e-8
        end
    end

    @testset "default route is unchanged (D-273)" begin
        fd = drm(_SOL_F, Student(); data = d, se = false)
        fla = drm(_SOL_F, Student(); data = d, se = false, marginal = :LA)
        @test fd.marginal === :LA
        @test fd.theta == fla.theta && loglik(fd) == loglik(fla)
        @test abs(loglik(fd) - loglik(fit)) > 1e-4            # a different integrator, not the same objective
    end

    @testset "se" begin
        f1 = drm(_SOL_F, Student(); data = d, marginal = :Laplace)
        @test all(isfinite, f1.vcov) && all(diag(f1.vcov) .> 0)
        @test f1.theta ≈ fit.theta atol = 1e-6
        @test all(isnan, fit.vcov)
    end

    @testset "out-of-scope models are refused" begin
        dd = merge(d, (h = repeat(["a", "b", "c", "d"], 40),))
        fx = bf(@formula(y ~ x), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(fx, Student(); data = d, marginal = :Laplace)
        fs = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(fs, Student(); data = d, marginal = :Laplace)
        fc = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1))
        @test_throws ArgumentError drm(fc, Student(); data = dd, marginal = :Laplace)
        @test_throws ArgumentError drm(_SOL_F, Student(); data = d, marginal = :VA)
        err = try drm(fx, Student(); data = d, marginal = :Laplace); nothing catch e e end
        @test occursin("marginal = :Laplace is not available for Student()", sprint(showerror, err))
    end
end
