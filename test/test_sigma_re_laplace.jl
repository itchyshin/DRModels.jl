# `marginal = :Laplace` on the Gaussian random intercept on sigma,
# bf(y ~ x, sigma ~ 1 + (1 | g)). The default `:LA` integrates each group's
# effect by non-adaptive 32-node Gauss-Hermite quadrature; `:Laplace` fits the
# Laplace approximation that native drmTMB (TMB) computes for this model.
#
# The only fixed numbers here are native drmTMB references
# (test/fixtures/sigma_re_laplace/native.tsv, drmTMB engine = "tmb"); every
# other check compares two computations made in this process.
using DRModels
using Test, Random, LinearAlgebra
using Distributions: Normal, logpdf
using ForwardDiff, QuadGK

const _SRL = DRModels
const _SRL_DIR = joinpath(@__DIR__, "fixtures", "sigma_re_laplace")

function _srl_readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    col(n) = cols[findfirst(==(n), hdr)]
    return (y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = col("g"))
end

_srl_native() = (ls = readlines(joinpath(_SRL_DIR, "native.tsv")); h = split(ls[1], '\t');
                 [Dict(zip(h, split(l, '\t'))) for l in ls[2:end]])

_srl_formula(cell) = startswith(cell, "c6") ?
    bf(@formula(y ~ x), @formula(sigma ~ 1 + x + (1 | g))) :
    bf(@formula(y ~ x), @formula(sigma ~ 1 + (1 | g)))

# Independent per-group Laplace reference, written from the full joint density
# with Distributions and ForwardDiff (no algebraic collapse, all constants kept):
# log ∫ Π N(yᵢ; μᵢ, e^{η0ᵢ+b}) N(b; 0, s²) db ≈ h(b̂) + ½ log 2π − ½ log(−h''(b̂)).
function _srl_ref_group(yg, μg, η0g, s)
    h(b) = sum(logpdf.(Normal.(μg, exp.(η0g .+ b)), yg)) + logpdf(Normal(0, s), b)
    dh(b) = ForwardDiff.derivative(h, b)
    d2h(b) = ForwardDiff.derivative(dh, b)
    b = 0.0
    for _ in 1:100
        step = dh(b) / d2h(b)
        b -= step
        abs(step) < 1e-15 && break
    end
    return h(b), h, b, d2h(b)
end

function _srl_ref_loglik(d, θ, Xμ, Xσ)
    pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    μ = Xμ * θ[1:pμ]; η0 = Xσ * θ[pμ+1:pμ+pσ]; s = exp(θ[end])
    ll = 0.0
    for lev in unique(d.g)
        idx = findall(==(lev), d.g)
        hb, _, _, h2 = _srl_ref_group(d.y[idx], μ[idx], η0[idx], s)
        ll += hb + 0.5 * log(2π) - 0.5 * log(-h2)
    end
    return ll
end

_srl_design(d, sigx) = (hcat(ones(length(d.y)), d.x),
                        sigx ? hcat(ones(length(d.y)), d.x) : ones(length(d.y), 1))

@testset "sigma-RE marginal = :Laplace" begin
    native = _srl_native()

    @testset "matches native drmTMB (TMB Laplace) on committed fixtures" begin
        for cell in unique(r["cell"] for r in native)
            nr = filter(r -> r["cell"] == cell, native)
            d = _srl_readfix(joinpath(_SRL_DIR, cell * ".csv"))
            fit = drm(_srl_formula(cell), Gaussian(); data = d, marginal = :Laplace)
            nat = parse.(Float64, [r["estimate"] for r in nr])
            natse = parse.(Float64, [r["se"] for r in nr])
            @test fit.marginal === :Laplace
            @test fit.converged
            @test dof(fit) == parse(Int, nr[1]["df"])
            @test abs(loglik(fit) - parse(Float64, nr[1]["logLik"])) <= 1e-6
            @test maximum(abs.(fit.theta .- nat) ./ abs.(nat)) <= 1e-5
            @test maximum(abs.(stderror(fit) .- natse) ./ natse) <= 1e-4
            # Same axis and name as the default route: the SD lives on log sigma.
            @test collect(keys(re_sd(fit))) == [:g_logsigma]
            @test re_sd(fit)[:g_logsigma] ≈ exp(fit.theta[end])
        end
    end

    cell = "c7_unequal_G20_sdb050"           # unequal group sizes, 3 to 120 rows
    d = _srl_readfix(joinpath(_SRL_DIR, cell * ".csv"))
    f = _srl_formula(cell)
    Xμ, Xσ = _srl_design(d, false)
    fitL = drm(f, Gaussian(); data = d, marginal = :Laplace)
    fitD = drm(f, Gaussian(); data = d)

    @testset "objective equals an independent Laplace reference" begin
        @test loglik(fitL) ≈ _srl_ref_loglik(d, fitL.theta, Xμ, Xσ) atol = 1e-9
        θp = fitL.theta .+ [0.05, -0.03, 0.1, -0.2]
        @test -fitL.nll(θp) ≈ _srl_ref_loglik(d, θp, Xμ, Xσ) atol = 1e-9
    end

    @testset "AD gradient and Hessian through the inner mode" begin
        θp = fitL.theta .+ [0.05, -0.03, 0.1, -0.2]
        gad = ForwardDiff.gradient(fitL.nll, θp)
        h = 1e-6
        gfd = [(fitL.nll(θp .+ h .* (1:4 .== k)) - fitL.nll(θp .- h .* (1:4 .== k))) / 2h for k in 1:4]
        @test gad ≈ gfd rtol = 1e-6
        Had = ForwardDiff.hessian(fitL.nll, θp)
        Hfd = reduce(hcat, [(ForwardDiff.gradient(fitL.nll, θp .+ h .* (1:4 .== k)) .-
                             ForwardDiff.gradient(fitL.nll, θp .- h .* (1:4 .== k))) ./ 2h for k in 1:4])
        @test Had ≈ Hfd rtol = 1e-5
        @test norm(ForwardDiff.gradient(fitL.nll, fitL.theta), Inf) < 1e-5
    end

    @testset "default :LA is unchanged GHQ-32 (D-273)" begin
        @test fitD.marginal === :LA
        for m in (:LA, :la)
            fitE = drm(f, Gaussian(); data = d, marginal = m)
            @test fitE.theta == fitD.theta
            @test loglik(fitE) === loglik(fitD)
            @test fitE.vcov == fitD.vcov
            @test fitE.marginal === :LA
        end
        # The default objective is still the 32-node Gauss-Hermite sum: an
        # independent transcription at the default optimum reproduces it.
        z, w = _SRL._gauss_hermite(32)
        θ = fitD.theta
        μ = Xμ * θ[1:2]; η0 = Xσ * θ[3:3]; s = exp(θ[4])
        ghq = 0.0
        for lev in unique(d.g)
            idx = findall(==(lev), d.g)
            t = [log(w[k] / sqrt(π)) + sum(logpdf.(Normal.(μ[idx], exp.(η0[idx] .+ sqrt(2) * s * z[k])), d.y[idx]))
                 for k in eachindex(z)]
            mx = maximum(t)
            ghq += mx + log(sum(exp.(t .- mx)))
        end
        @test loglik(fitD) ≈ ghq atol = 1e-8
        # ...and the two integrators are genuinely different objectives here.
        @test abs(loglik(fitD) - loglik(fitL)) > 1.0
        @test abs(fitD.nll(θ) - fitL.nll(θ)) > 0.1
    end

    @testset "Laplace vs a dense 1-D integral: error shrinks with group size" begin
        # One group, mode-centred adaptive quadrature of the exact integrand.
        # Laplace's relative error is O(1/m): small at m = 400, larger at m = 25.
        rng = MersenneTwister(20260925)
        s = 0.4
        errs = Float64[]
        for m in (25, 400)
            b = 0.3
            μg = zeros(m); η0g = fill(-0.3, m)
            yg = exp(-0.3 + b) .* randn(rng, m)
            hb, h, b̂, h2 = _srl_ref_group(yg, μg, η0g, s)
            exact = hb + log(quadgk(u -> exp(h(u) - hb), b̂ - 12 / sqrt(-h2), b̂ + 12 / sqrt(-h2);
                                     rtol = 1e-13)[1])
            B = sum(yg .^ 2 .* exp.(-2 .* η0g))
            lap = -0.5 * m * log(2π) - sum(η0g) + _SRL._sigre_laplace_group(m, B, s^2)
            @test lap ≈ hb + 0.5 * log(2π) - 0.5 * log(-h2) atol = 1e-10
            push!(errs, exact - lap)
        end
        @test abs(errs[2]) < 2e-3
        @test abs(errs[1]) > 5 * abs(errs[2])
    end

    @testset "inner mode solver" begin
        for (m, B, s2) in ((10, 10.0, 0.16), (10, 1e-8, 0.16), (10, 1e8, 0.16),
                           (400, 50.0, 1e-6), (3, 7.0, 25.0), (5, 0.0, 0.2))
            b = _SRL._sigre_mode(m, B, s2)
            @test abs(-m + B * exp(-2b) - b / s2) <= 1e-9 * (m + B * exp(-2b) + abs(b) / s2)
        end
    end

    @testset "refusals: :Laplace only where implemented" begin
        dm = (y = d.y, x = d.x, g = d.g)
        msg = r"marginal = :Laplace is not available for Gaussian\(\) with"
        @test_throws msg drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian();
                             data = dm, marginal = :Laplace)
        @test_throws r"no random effect on `sigma`" drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + x)),
                                                        Gaussian(); data = dm, marginal = :Laplace)
        @test_throws r"random slope on `sigma`" drm(bf(@formula(y ~ x), @formula(sigma ~ 1 + (0 + x | g))),
                                                    Gaussian(); data = dm, marginal = :Laplace)
        @test_throws r"method = :REML" drm(f, Gaussian(); data = dm, marginal = :Laplace, method = :REML)
        @test_throws r"algorithm = :sparse" drm(f, Gaussian(); data = dm, marginal = :Laplace,
                                                algorithm = :sparse)
        @test_throws r"`marginal = :VA` is not available for Gaussian\(\)" drm(f, Gaussian(); data = dm,
                                                                               marginal = :VA)
        @test_throws r"`marginal = :AGHQ` is not available for Gaussian\(\)" drm(f, Gaussian(); data = dm,
                                                                                 marginal = :AGHQ)
        # One naming rule in every message: `:LA` is the route's default
        # integrator (GHQ-32 here), `:Laplace` forces Laplace. No message may
        # send a user to `:LA` as "Laplace".
        @test_throws r"Gauss–Hermite quadrature, not Laplace" drm(f, Gaussian(); data = dm, marginal = :VA)
        dp = (y = round.(Int, abs.(d.y)), x = d.x, g = d.g)
        @test_throws r"implemented only for a Gaussian random intercept on `sigma`" drm(
            bf(@formula(y ~ x + (1 | g))), Poisson(); data = dp, marginal = :Laplace)
    end

    @testset "bootstrap refits with the seed fit's integrator" begin
        # Rebuild replicate 1 exactly as `_bootstrap_result` does, then refit it
        # both ways: the bootstrap draw must be the `:Laplace` refit, not GHQ-32.
        r = bootstrap_result(fitL; data = d, B = 1, rng = MersenneTwister(3))
        sim = _SRL._marginal_simulator(fitL, d)
        rr = MersenneTwister(r.seeds[1])
        ys = sim === nothing ? simulate(fitL; rng = rr) : sim(rr)
        db = _SRL._bootstrap_data(fitL.formula, d, ys)
        draw = [row.lower for row in r.summary]
        @test draw == coef(drm(f, Gaussian(); data = db, marginal = :Laplace))
        @test maximum(abs.(draw .- coef(drm(f, Gaussian(); data = db)))) > 1e-4
    end

    @testset "through drm_bridge" begin
        dm = Dict("y" => d.y, "x" => d.x, "g" => d.g)
        rL = drm_bridge(formula = "y ~ x; sigma ~ 1 + (1 | g)", family = "gaussian", data = dm,
                        options = Dict{String,Any}("marginal" => "Laplace"))
        @test rL["marginal"] == "Laplace"
        @test rL["loglik"] == loglik(fitL)
        rD = drm_bridge(formula = "y ~ x; sigma ~ 1 + (1 | g)", family = "gaussian", data = dm)
        @test rD["marginal"] == "LA"
        @test rD["loglik"] == loglik(fitD)
        @test_throws r"marginal = :Laplace is not available" drm_bridge(
            formula = "y ~ x + (1 | g); sigma ~ 1", family = "gaussian", data = dm,
            options = Dict{String,Any}("marginal" => "Laplace"))
        @test_throws r"option `marginal` is not available for Student\(\)" drm_bridge(
            formula = "y ~ x", family = "student", data = dm,
            options = Dict{String,Any}("marginal" => "Laplace"))
    end
end
