# P22d: the refusal text for `marginal = :Laplace` on a route without it must list every
# family that does accept it, Tweedie included. Tweedie `(1 | g)` with `marginal = :Laplace`
# is fitted (not refused), so the list in the text would otherwise be wrong.
using DRModels
using Test, Random
import Distributions

@testset "marginal = :Laplace refusal text lists Tweedie (P22d)" begin
    Random.seed!(1122)
    G = 30; m = 12; n = G * m
    g = repeat(1:G, inner = m); x = randn(n)
    ntr = fill(10, n)
    p = 1 ./ (1 .+ exp.(-(0.2 .+ 0.5 .* x .+ 0.4 .* randn(G)[g])))
    s = [rand(Distributions.Binomial(ntr[i], p[i])) for i in 1:n]
    data = (; s = Float64.(s), fail = Float64.(ntr .- s), x, g)
    # BetaBinomial has no `:Laplace` option, so it reaches the refusal text.
    err = try
        drm(bf(@formula(cbind(s, fail) ~ x + (1 | g)), @formula(sigma ~ 1)), BetaBinomial();
            data = data, marginal = :Laplace)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test occursin("Poisson/Binomial/NegBinomial2/Gamma/Beta/Tweedie", msg)

    # The listed claim is true: Tweedie `(1 | g)` accepts `marginal = :Laplace`.
    Random.seed!(1123)
    ug = 0.3 .* randn(G)
    y = map(1:n) do i                       # compound Poisson-gamma draw
        k = rand(Distributions.Poisson(exp(0.2 + 0.4 * x[i] + ug[g[i]])))
        k == 0 ? 0.0 : sum(rand(Distributions.Gamma(1.5, 0.8), k))
    end
    dt = (; y, x, g)
    fit = drm(bf(@formula(y ~ x + (1 | g))), Tweedie(); data = dt, marginal = :Laplace,
                 control = DRMControl(se = false))
    @test isfinite(loglik(fit))

    # The unknown-method text carries the same list.
    err2 = try
        drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = (; y = round.(abs.(y)), x, g),
            marginal = :nonsense)
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("Poisson/Binomial/NegBinomial2/Gamma/Beta/Tweedie", sprint(showerror, err2))
end
