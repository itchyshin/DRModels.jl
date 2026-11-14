# P22d: the refusal text for `marginal = :Laplace` on a route without it must list every
# family that does accept it for an ordinary `(1 | g)` on the mean, and only those.
# The test fits `(1 | g)` with `marginal = :Laplace` for every exported family that has
# a `(1 | g)` route, takes the families that were accepted, and checks that set against the
# list in both texts. A family added to (or dropped from) the accepted set without a text
# change fails here.
using DRModels
using Test, Random
import Distributions

const _P22D_LIST = "Poisson/Binomial/NegBinomial2/Gamma/Beta/Tweedie/Student/LogNormal/ZeroOneBeta"

@testset "marginal = :Laplace refusal text lists exactly the accepting families (P22d)" begin
    Random.seed!(1122)
    G = 20; m = 8; n = G * m
    g = repeat(1:G, inner = m); x = randn(n); u = 0.4 .* randn(G)
    η = 0.3 .+ 0.4 .* x .+ u[g]
    p = 1 ./ (1 .+ exp.(-η))
    ntr = fill(10, n)
    s = Float64.([rand(Distributions.Binomial(ntr[i], p[i])) for i in 1:n])
    ycnt = Float64.([rand(Distributions.Poisson(exp(e))) for e in η])
    ycnt1 = ycnt .+ 1
    ytw = map(1:n) do i
        k = rand(Distributions.Poisson(exp(η[i])))
        k == 0 ? 0.0 : sum(rand(Distributions.Gamma(1.5, 0.8), k))
    end
    ypos = exp.(η .+ 0.3 .* randn(n))
    yg = η .+ 0.5 .* randn(n)
    ybeta = clamp.(p .+ 0.05 .* randn(n), 0.02, 0.98)
    yzob = copy(ybeta); yzob[1:5] .= 0.0; yzob[6:10] .= 1.0
    yord = Float64.(1 .+ (η .> -0.2) .+ (η .> 0.6) .+ (η .> 1.2))
    d = (; x, g, yg, ypos, ycnt, ycnt1, s, fail = ntr .- s, ybeta, yzob, yord, ytw)

    # (family, mean-only or sigma ~ 1 formula that the family's `(1 | g)` route takes)
    cases = [
        (Gaussian(), bf(@formula(yg ~ x + (1 | g)), @formula(sigma ~ 1))),
        (Student(), bf(@formula(yg ~ x + (1 | g)))),
        (SkewNormal(), bf(@formula(yg ~ x + (1 | g)))),
        (LogNormal(), bf(@formula(ypos ~ x + (1 | g)))),
        (Poisson(), bf(@formula(ycnt ~ x + (1 | g)))),
        (NegBinomial2(), bf(@formula(ycnt ~ x + (1 | g)), @formula(sigma ~ 1))),
        (TruncatedPoisson(), bf(@formula(ycnt1 ~ x + (1 | g)))),
        (TruncatedNegBinomial2(), bf(@formula(ycnt1 ~ x + (1 | g)), @formula(sigma ~ 1))),
        (Binomial(), bf(@formula(cbind(s, fail) ~ x + (1 | g)))),
        (BetaBinomial(), bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sigma ~ 1))),
        (Beta(), bf(@formula(ybeta ~ x + (1 | g)), @formula(sigma ~ 1))),
        (ZeroOneBeta(), bf(@formula(yzob ~ x + (1 | g)))),
        (Gamma(), bf(@formula(ypos ~ x + (1 | g)), @formula(sigma ~ 1))),
        (Tweedie(), bf(@formula(ytw ~ x + (1 | g)))),
        (CumulativeLogit(), bf(@formula(yord ~ x + (1 | g)))),
    ]
    accepted = String[]
    refusal = nothing
    for (fam, f) in cases
        nm = string(nameof(typeof(fam)))
        res = try
            fit = drm(f, fam; data = d, marginal = :Laplace, control = DRMControl(se = false))
            @test isfinite(loglik(fit))
            push!(accepted, nm)
            nothing
        catch e
            e isa ArgumentError || rethrow()
            e
        end
        # Keep the text of a refusal that comes from `_marginal_method` (BetaBinomial).
        nm == "BetaBinomial" && (refusal = res)
    end

    # The list in the text is exactly the accepted set.
    listed = split(_P22D_LIST, "/")
    @test Set(accepted) == Set(listed)

    @test refusal isa ArgumentError
    @test occursin(_P22D_LIST, sprint(showerror, refusal))

    # The unknown-method text carries the same list.
    err2 = try
        drm(bf(@formula(ycnt ~ x + (1 | g))), Poisson(); data = d, marginal = :nonsense)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin(_P22D_LIST, sprint(showerror, err2))
end
