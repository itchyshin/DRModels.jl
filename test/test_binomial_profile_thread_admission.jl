using DRModels, Test, Random
using StableRNGs
import Distributions

@testset "Binomial SD profile threading admits independent quadrature routes" begin
    rng = StableRNG(20261022)
    G, m = 32, 14
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = repeat(z_group, inner = m)
    x = repeat(range(-1.0, 1.0; length = m), outer = G)
    sd_group = exp.(-0.6 .+ 0.35 .* z_group)
    random_intercept = sd_group .* randn(rng, G)
    eta = -0.2 .+ 0.3 .* x .+ random_intercept[group]
    probability = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    data = (; y, x, z, group)
    formula = bf(@formula(y ~ x + (1 | group)), @formula(sd(group) ~ z))
    fit = drm(formula, Binomial(); data, se = false)

    @test fit.converged
    profile = profile_result(fit; parm = :sd => "z", threads = true)
    @test profile.attempted == 1
    @test profile.threaded == (Threads.nthreads() > 1)
end

@testset "Binomial correlated SD profile threading uses ordinary quadrature" begin
    rng = StableRNG(20261023)
    G, m = 64, 18
    group = repeat(1:G, inner = m)
    z_group = repeat([-1.0, 1.0], inner = G ÷ 2)
    z = repeat(z_group, inner = m)
    x = repeat(range(-1.0, 1.0; length = m), outer = G)
    tau0 = exp.(-0.65 .+ 0.55 .* z_group)
    rho, tau1 = 0.4, 0.6
    u = randn(rng, 2, G)
    b0 = tau0 .* u[1, :]
    b1 = tau1 .* (rho .* u[1, :] .+ sqrt(1 - rho^2) .* u[2, :])
    eta = -0.1 .+ 0.25 .* x .+ b0[group] .+ x .* b1[group]
    probability = 1 ./ (1 .+ exp.(-eta))
    y = Float64.(rand.(Ref(rng), Distributions.Bernoulli.(probability)))
    data = (; y, x, z, group)
    formula = bf(
        @formula(y ~ x + (1 + x | group)),
        @formula(sd(group, coef(intercept)) ~ z),
    )
    fit = drm(formula, Binomial(); data, se = false)

    @test fit.converged
    target = only(filter(profile_targets(fit)) do row
        row.parm == "z" && startswith(String(row.param), "sdtarget_")
    end)
    @test target.profile_ready
    profile = profile_result(fit; parm = target.param => target.parm, threads = true)
    @test profile.attempted == 1
    @test profile.threaded == (Threads.nthreads() > 1)
end
