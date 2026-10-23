# marginal = :AGHQ for crossed random intercepts (1 | g) + (1 | h) (#761; supersedes
# the rejected #850). Laplace is biased low for Bernoulli data with a large σ_g and
# few observations per g-level; the nested adaptive quadrature corrects it.
#
# Fixtures: the #850 reviewer's two Bernoulli designs (G = 300, H = 4, n = 1600,
# σ_g = 2.5, β = (0.2, 0.45); σ_h = 0.001 and σ_h = 0.3), stored as CSV so the draw
# does not depend on the Julia/Distributions version. REFERENCE: the reviewer's
# independent integrator (40-node AGHQ per g-level, product AGHQ over the 4 h-effects
# with its own BFGS mode and FD Hessian; Ko = 3 and 5 agree to 4 decimals),
# evaluated at this route's optimum — see the PR body for the script.
using Test, DelimitedFiles, LinearAlgebra, Random
using DRModels

const _CAGHQ_DIR = joinpath(@__DIR__, "fixtures", "crossed_aghq")

function _caghq_data(file)
    M = readdlm(joinpath(_CAGHQ_DIR, file), ','; skipstart = 1)
    s = Float64.(M[:, 1])
    return (; s, fail = 1 .- s, x = Float64.(M[:, 2]),
            g = Symbol.(String.(M[:, 3])), h = Symbol.(String.(M[:, 4])))
end

const _CAGHQ_FM = bf(@formula(cbind(s, fail) ~ x + (1 | g) + (1 | h)))

# Reviewer's reference log-likelihood at this route's optimum (Ko = 3 and Ko = 5 outer
# agree to 4 decimals). Route: −854.7753 / −860.2474, i.e. within 0.002 nat. For scale,
# the same reference at the crossed-LAPLACE optimum is −855.0982 / −860.5694 and the
# Laplace logLik itself is −864.4213 / −869.9158.
const _CAGHQ_REF = Dict("bernoulli_sh0.csv" => -854.7745, "bernoulli_sh0.3.csv" => -860.2458)

@testset "crossed AGHQ: K = 1 is the crossed Laplace objective" begin
    d = _caghq_data("bernoulli_sh0.3.csv")
    n = length(d.s)
    X = hcat(ones(n), d.x)
    gidx, G = DRModels._group_index(d.g); hidx, H = DRModels._group_index(d.h)
    comps = [(ones(n), gidx, G, "g"), (ones(n), hidx, H, "h")]
    lap = DRModels._fit_binomial_crossed_laplace(DRModels.Binomial(), d.s, ones(n), X, comps,
                                                 ["(Intercept)", "x"], 1e-8)
    aux = (s = round.(Int, d.s), ntr = ones(Int, n), logchoose = zeros(n))
    ll = (i, η) -> -DRModels._laplace_value(Val(:binomial), aux, i, η)
    gm = [Int[] for _ in 1:G]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(G), uh = zeros(H), Zre = ones(n, 1))
    f(θ, Kg, Kh) = DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, H, X * θ[1:2], exp(θ[3]),
                                                 exp(θ[4]), DRModels._AGHQRule(1, Kg),
                                                 DRModels._AGHQRule(H, Kh), cache)
    for θ in (lap.theta, lap.theta .+ [0.1, -0.05, 0.2, 0.3])
        # the Laplace route stops its mode search at 1e-8 relative, which moves its
        # ½ log|H| by ~1e-7 at the optimum; off the optimum the two agree to ~1e-10
        @test abs(f(θ, 1, 1) + lap.nll(θ)) < 1e-6
    end
    # the Laplace bias is large here and the quadrature removes it
    @test f(lap.theta, 15, 3) - lap.loglik > 9.0
end

@testset "crossed AGHQ: Poisson K = 1 ≡ crossed Laplace, and the front end" begin
    rng = MersenneTwister(761)
    G, H, n = 40, 3, 400
    x = randn(rng, n); gi = rand(rng, 1:G, n); hi = rand(rng, 1:H, n)
    bg = 0.8 .* randn(rng, G); bh = 0.4 .* randn(rng, H)
    y = [Float64(rand(rng, DRModels.Distributions.Poisson(exp(0.3 + 0.5x[i] + bg[gi[i]] + bh[hi[i]])))) for i in 1:n]
    data = (; y, x, g = Symbol.("g", gi), h = Symbol.("h", hi))
    fm = bf(@formula(y ~ x + (1 | g) + (1 | h)))
    fl = drm(fm, Poisson(); data = data, se = false)
    X = hcat(ones(n), x)
    gidx, Gc = DRModels._group_index(data.g); hidx, Hc = DRModels._group_index(data.h)
    lf = [DRModels._logfactorial(round(Int, v)) for v in y]
    ll = (i, η) -> (ηc = clamp(η, -30.0, 30.0); y[i] * ηc - exp(ηc) - lf[i])
    gm = [Int[] for _ in 1:Gc]; for i in 1:n; push!(gm[gidx[i]], i); end
    cache = (ug = zeros(Gc), uh = zeros(Hc), Zre = ones(n, 1))
    θ = fl.theta
    v1 = DRModels._crossed_aghq_loglik(ll, gm, gidx, hidx, Hc, X * θ[1:2], exp(θ[3]), exp(θ[4]),
                                       DRModels._AGHQRule(1, 1), DRModels._AGHQRule(Hc, 1), cache)
    @test abs(v1 + fl.nll(θ)) < 1e-6
    fa = drm(fm, Poisson(); data = data, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fa.converged
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    # Poisson with ~10 counts per g-level: Laplace is already close
    @test abs(fa.loglik - fl.loglik) < 0.5
end

@testset "crossed AGHQ: accuracy vs the reviewer's reference ($file)" for file in
        ("bernoulli_sh0.csv", "bernoulli_sh0.3.csv")
    d = _caghq_data(file)
    fl = drm(_CAGHQ_FM, DRModels.Binomial(); data = d, se = false)
    fa = drm(_CAGHQ_FM, DRModels.Binomial(); data = d, se = false, marginal = :AGHQ)
    @test fa.marginal === :AGHQ
    @test fl.marginal !== :AGHQ
    @test fa.converged
    # fit.loglik and fit.nll are ONE objective
    @test fa.loglik ≈ -fa.nll(fa.theta) atol = 1e-8
    @test abs(fa.loglik - _CAGHQ_REF[file]) < 0.05
    @test fa.loglik - fl.loglik > 3.0                 # Laplace bias ≈ 9 nat on these designs
    @test exp(fa.theta[3]) > exp(fl.theta[3])        # Laplace shrinks σ_g
    if file == "bernoulli_sh0.3.csv"
        σh = exp(fa.theta[4])
        @test 0.1 < σh < 1.0                          # interior, not at the boundary
    end
    # lrtest refuses to mix integrators (AGHQ vs Laplace crossed; AGHQ vs GHQ nested)
    @test_throws ArgumentError lrtest(fl, fa)
    fn = drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), DRModels.Binomial(); data = d, se = false)
    @test_throws ArgumentError lrtest(fn, fa)
end

@testset "crossed AGHQ: refusals" begin
    rng = MersenneTwister(1)
    n = 200
    d = (; s = Float64.(rand(rng, 0:1, n)), x = randn(rng, n),
         g = Symbol.("g", rand(rng, 1:20, n)), h = Symbol.("h", rand(rng, 1:12, n)))
    d = merge(d, (; fail = 1 .- d.s))
    # both groupings have more than 8 levels
    @test_throws ArgumentError drm(_CAGHQ_FM, DRModels.Binomial(); data = d, marginal = :AGHQ)
    # Binomial :AGHQ is the crossed route only
    @test_throws ArgumentError drm(bf(@formula(cbind(s, fail) ~ x + (1 | g))), DRModels.Binomial();
                                   data = d, marginal = :AGHQ)
end
