# test_summary.jl — readable DrmFit printout (Base.show MIME"text/plain") and the
# StatsModels-backed `coeftable`. Runs standalone: `using DRModels, Test, Random`.

using DRModels, Test, Random
import Distributions          # qualified — DRModels exports its own `Poisson` family

@testset "DrmFit summary (show + coeftable)" begin
    @testset "Gaussian location–scale (sigma ~ x)" begin
        Random.seed!(1)
        n = 200
        x = randn(n)
        y = 1.0 .+ 0.5 .* x .+ exp.(0.2 .+ 0.3 .* x) .* randn(n)
        fit = drm(bf(@formula(y ~ x), @formula(sigma ~ x)), Gaussian(); data = (; y, x))

        s = sprint(show, MIME("text/plain"), fit)
        @test s isa String && !isempty(s)
        @test occursin("Gaussian", s)
        @test occursin("logLik", s)
        @test occursin("(Intercept)", s)   # at least one coefficient name
        @test occursin("x", s)
        @test occursin("Pr(>|z|)", s)      # the table header

        ct = coeftable(fit)
        @test size(ct.cols[1], 1) == length(coef(fit))   # one row per coefficient
    end

    @testset "Poisson (y ~ x)" begin
        Random.seed!(2)
        n = 150
        x = randn(n)
        λ = exp.(0.3 .+ 0.4 .* x)
        y = Float64.([rand(Distributions.Poisson(λi)) for λi in λ])
        fit = drm(bf(@formula(y ~ x)), Poisson(); data = (; y, x))

        s = sprint(show, MIME("text/plain"), fit)
        @test s isa String && !isempty(s)
        @test occursin("Poisson", s)
        @test occursin("logLik", s)
        @test occursin("(Intercept)", s)

        ct = coeftable(fit)
        @test size(ct.cols[1], 1) == length(coef(fit))
    end

    @testset "Random intercept (y ~ x + (1 | g))" begin
        Random.seed!(3)
        G = 25
        nper = 8
        n = G * nper
        g = repeat(1:G, inner = nper)
        x = randn(n)
        b = 0.7 .* randn(G)
        y = 1.0 .+ 0.5 .* x .+ b[g] .+ 0.4 .* randn(n)
        fit = drm(bf(@formula(y ~ x + (1 | g))), Gaussian(); data = (; y, x, g))

        s = sprint(show, MIME("text/plain"), fit)
        @test s isa String && !isempty(s)
        @test occursin("Gaussian", s)
        @test occursin("logLik", s)
        @test occursin("(Intercept)", s)
        # random-effect SD section should render its grouping factor name
        @test occursin("g", s)

        ct = coeftable(fit)
        @test size(ct.cols[1], 1) == length(coef(fit))
    end

    # #495: the q4 Sigma_a Wald intervals were measured miscalibrated (off-diagonals
    # under-cover, scale diagonals over-cover), so the printed :phylocov block heading
    # must say they are uncalibrated. Pins the label only; estimates are untouched.
    @testset "q4 phylo fit labels the :phylocov block uncalibrated (#495)" begin
        Random.seed!(42)
        phy = random_balanced_tree(6; branch_length = 0.2)
        species = repeat(phy.leaf_names, inner = 3)
        n = length(species)
        x = randn(n)
        u = Dict(name => 0.15 .* randn(4) for name in phy.leaf_names)
        y1 = [1 + 0.4*x[i] + u[species[i]][1] +
              exp(-0.4 + u[species[i]][3]) * randn() for i in 1:n]
        y2 = [-0.2 + 0.3*x[i] + u[species[i]][2] +
              exp(-0.5 + u[species[i]][4]) * randn() for i in 1:n]
        fit = drm(
            bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
               mu2 = @formula(y2 ~ x + phylo(1 | species)),
               sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
               sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
               rho12 = @formula(rho12 ~ 1)),
            Gaussian();
            data = (; y1, y2, x, species), tree = phy, q4_vcov = false,
        )
        s = sprint(show, MIME("text/plain"), fit)
        rows = filter(l -> occursin("Group-level covariance", l), split(s, '\n'))
        @test length(rows) == 1
        @test occursin("uncalibrated", only(rows))
    end
end
