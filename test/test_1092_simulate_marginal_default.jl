# #1092 (owner decision, 25 Oct 2026): `simulate()` on a `(1 | g)` fit draws
# FRESH random effects by default (the marginal simulator the parametric
# bootstrap uses, #462), and the old b = 0 draw stays available as an option.
#
# Before the change `simulate` drew at b = 0 only: for the Poisson `(1 | g)`
# example in the issue the variance of the 24 group means was 0.607 in the
# draws against 2.277 in the data. These tests check the draws against the
# MARGINAL moments of the fitted model (random effects integrated over their
# fitted N(0, sd^2) distribution), and the `re = :zero` draws against the
# b = 0 moments.
using DRModels
using Test, Random, Statistics
import Distributions

# Average over groups of the across-draw variance of each group's mean.
function _t1092_groupmean_var(Y, gidx, G)
    v = zeros(G)
    for g in 1:G
        rows = findall(==(g), gidx)
        v[g] = var(vec(mean(Y[rows, :]; dims = 1)))
    end
    return v
end

@testset "#1092 simulate draws fresh random effects by default" begin
    G, m = 24, 20
    gidx = repeat(1:G, inner = m)
    n = G * m
    labels = string.("g", gidx)

    @testset "Gaussian (1 | g): marginal moments by default" begin
        rng = MersenneTwister(1092)
        x = randn(rng, n)
        u = 0.8 .* randn(rng, G)
        y = 1.0 .+ 0.5 .* x .+ u[gidx] .+ 0.6 .* randn(rng, n)
        dat = (; y, x, g = labels)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = dat)
        s = re_sd(fit)[:g]
        σ = mean(DRModels._scale_vector(fit, :sigma))   # constant: sigma ~ 1
        S = 2000

        Y = simulate(fit; nsim = S, data = dat, rng = MersenneTwister(1))
        @test size(Y) == (n, S)
        # Var(ȳ_g) = s^2 + σ^2 / m under the fitted marginal model.
        target = s^2 + σ^2 / m
        @test mean(_t1092_groupmean_var(Y, gidx, G)) ≈ target rtol = 0.08
        # The grand mean of the draws is the fixed-effect mean.
        @test mean(Y) ≈ mean(predict(fit, dat)) atol = 4 * s / sqrt(G * S)

        # nsim = 1 still returns a Vector.
        @test simulate(fit; data = dat, rng = MersenneTwister(2)) isa AbstractVector

        # b = 0 option: only residual noise within each group.
        Z = simulate(fit; nsim = S, re = :zero, rng = MersenneTwister(3))
        @test mean(_t1092_groupmean_var(Z, gidx, G)) ≈ σ^2 / m rtol = 0.08
        # The b = 0 draw is the old behaviour, bit for bit.
        @test simulate(fit; re = :zero, rng = MersenneTwister(4)) ==
              DRModels._simulate_once(fit, MersenneTwister(4))
    end

    @testset "Poisson (1 | g): marginal moments by default" begin
        rng = MersenneTwister(10920)
        x = randn(rng, n)
        u = 0.5 .* randn(rng, G)
        y = Float64[rand(rng, Distributions.Poisson(exp(0.8 + 0.3 * x[i] + u[gidx[i]])))
                    for i in 1:n]
        dat = (; y, x, g = labels)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = dat)
        s = re_sd(fit)[:g]
        η = predict(fit, dat; type = :link)
        S = 2000

        Y = simulate(fit; nsim = S, data = dat, rng = MersenneTwister(5))
        @test size(Y) == (n, S)
        @test all(v -> v >= 0 && v == round(v), Y)
        # E[y_i] = exp(η_i + s^2 / 2).
        @test mean(Y) ≈ mean(exp.(η .+ s^2 / 2)) rtol = 0.03
        # Var(ȳ_g) = (Σ e^η / m)^2 (e^{2s^2} - e^{s^2}) + Σ e^{η + s^2/2} / m^2.
        target = mean(1:G) do g
            e = exp.(η[gidx .== g])
            (sum(e) / m)^2 * (exp(2s^2) - exp(s^2)) + sum(e .* exp(s^2 / 2)) / m^2
        end
        @test mean(_t1092_groupmean_var(Y, gidx, G)) ≈ target rtol = 0.10

        # b = 0 option: mean exp(η) and Poisson noise only.
        Z = simulate(fit; nsim = S, re = :zero, rng = MersenneTwister(6))
        @test mean(Z) ≈ mean(exp.(η)) rtol = 0.03
        target0 = mean(g -> sum(exp.(η[gidx .== g])) / m^2, 1:G)
        @test mean(_t1092_groupmean_var(Z, gidx, G)) ≈ target0 rtol = 0.10
    end

    @testset "refusals" begin
        rng = MersenneTwister(10921)
        x = randn(rng, n)
        y = 1.0 .+ 0.5 .* x .+ (0.8 .* randn(rng, G))[gidx] .+ 0.6 .* randn(rng, n)
        dat = (; y, x, g = labels)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = dat)
        # The default needs the fitting data to place the fresh effects; it
        # refuses rather than fall back to b = 0 silently.
        err = try
            simulate(fit; rng = MersenneTwister(1)); nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("re = :zero", sprint(showerror, err))
        @test_throws ArgumentError simulate(fit; data = dat, re = :conditional)

        # A fit without random effects needs no data and ignores `re`.
        fit0 = drm(bf(@formula(y ~ x)), Gaussian(); data = dat)
        @test simulate(fit0; rng = MersenneTwister(7)) ==
              simulate(fit0; re = :zero, rng = MersenneTwister(7))
    end
end
