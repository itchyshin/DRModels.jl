# Beta location-scale `yield ~ temp + batch`, `sigma ~ batch` on betareg::GasolineYield
# (n = 32, 10 batches of 2-4 rows) (#1127).
#
# LBFGS used up its iterations 0.0168 nats short of the optimum with the gradient
# still near 50, flagged the fit not converged, and `coeftable` printed 21 finite SEs
# anyway. Reference logLik: drmTMB ca3ddef 92.0596218477018, glmmTMB 1.1.15.2
# 92.0596218474146 (issue #1127). Data frozen in test/fixtures/beta_locscale_1127
# (batch labels prefixed "g" as in the issue).

using DRModels
using Test
using DelimitedFiles
using LinearAlgebra
using Random
using Statistics

const _G1127_REF = 92.0596218475

function _gasoline_1127()
    m, h = readdlm(joinpath(@__DIR__, "fixtures", "beta_locscale_1127", "GasolineYield.csv"), ',';
                   header = true)
    nm = replace.(string.(vec(h)), "\"" => "")
    col(n) = m[:, findfirst(==(n), nm)]
    return (yield = Float64.(col("yield")), temp = Float64.(col("temp")),
            batch = String.(col("batch")))
end

const _G1127_F = bf(@formula(yield ~ 1 + temp + batch), @formula(sigma ~ 1 + batch))

@testset "#1127 Beta location-scale on GasolineYield" begin
    d = _gasoline_1127()
    @test length(d.yield) == 32

    fit = drm(_G1127_F, Beta(); data = d)
    @test abs(loglik(fit) - _G1127_REF) < 1e-4
    @test is_converged(fit)
    # the stop point is a stationary point, not a plateau
    g = DRModels.ForwardDiff.gradient(fit.nll, fit.theta)
    @test maximum(abs, g) < 1e-5
    @test all(isfinite, stderror(fit))

    ct = coeftable(fit)
    @test count(isfinite, ct.cols[2]) == length(fit.theta)

    @testset "a not-converged fit prints no usable SEs" begin
        # A user-capped optimiser is not polished, so this stops short and unconverged.
        capped = drm(_G1127_F, Beta(); data = d, control = DRMControl(maxit = 5))
        @test !is_converged(capped)
        @test all(isnan, coeftable(capped).cols[2])
        @test all(isnan, coeftable(capped).cols[4])
        @test occursin("Wald SEs withheld", sprint(show, MIME"text/plain"(), capped))
        @test size(vcov(capped)) == (21, 21)      # vcov() semantics are unchanged
    end

    @testset "panel of seeded Beta location-scale fits does not get worse" begin
        for seed in 1:10
            rng = MersenneTwister(1127 + seed)
            G = 8; per = rand(rng, 2:4, G)
            batch = String[]; x = Float64[]
            for gi in 1:G, _ in 1:per[gi]
                push!(batch, "b$gi"); push!(x, 200 + 250 * rand(rng))
            end
            σg = exp.(-1 .+ 0.4 .* randn(rng, G))
            y = map(eachindex(x)) do i
                μ = 1 / (1 + exp(-(-3.5 + 0.01 * x[i])))
                φ = 1 / σg[parse(Int, batch[i][2:end])]^2
                clamp(DRModels.Distributions.rand(rng, DRModels.Distributions.Beta(μ * φ, (1 - μ) * φ)), 1e-4, 1 - 1e-4)
            end
            dd = (y = y, x = x, batch = batch)
            f = bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + batch))
            # reference: the same fit with the optimiser taken over by the user (no polish)
            base = drm(f, Beta(); data = dd, control = DRMControl(maxit = 1000))
            new = drm(f, Beta(); data = dd)
            @info "panel seed $seed" gain = loglik(new) - loglik(base) conv_base = is_converged(base) conv_new = is_converged(new)
            @test loglik(new) >= loglik(base) - 1e-9
            @test isfinite(loglik(new))
        end
    end
end
