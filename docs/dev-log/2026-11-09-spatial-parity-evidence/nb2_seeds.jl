# usage: julia --project=<worktree> nb2_seeds.jl <hd dir> <out tsv> <seed lo> <seed hi>
using DRModels, DelimitedFiles, Printf, Random, LinearAlgebra, Statistics
HD, OUT, lo, hi = ARGS[1], ARGS[2], parse(Int, ARGS[3]), parse(Int, ARGS[4])
cm = readdlm(joinpath(HD, "coords.csv"), ',', Any; skipstart = 1)
ids = String.(cm[:, 1]); xy = Float64.(cm[:, 2:3]); G = length(ids)
coords = Dict(ids[i] => xy[i, :] for i in 1:G)
D = [norm(xy[k, :] - xy[l, :]) for k in 1:G, l in 1:G]
pos = [D[k, l] for k in 1:G for l in (k+1):G if D[k, l] > 0]
C = exp.(-D ./ median(pos)) + 1e-6I; L = cholesky(Symmetric(C)).L
f = bf(@formula(y ~ x1 + x2 + x3 + spatial(1 + x1 | p | g)), @formula(sigma ~ 1),
       @formula(sd(g, spatial, coef(intercept)) ~ z), @formula(sd(g, spatial, coef(x1)) ~ z))
open(OUT, "w") do io
    println(io, "seed\tlogLik\tconv\tsecs")
    for seed in lo:hi
        rng = MersenneTwister(seed)
        nper = 10; n = G * nper
        g = repeat(ids, inner = nper)
        gz = rand(rng, G) .* 2 .- 1
        x1, x2, x3 = randn(rng, n), randn(rng, n), randn(rng, n)
        rho = 0.3
        e0 = L * randn(rng, G); e1 = L * randn(rng, G)
        b0 = e0; b1 = rho .* e0 .+ sqrt(1 - rho^2) .* e1
        sd0 = exp.(-0.8 .+ 0.3 .* gz); sd1 = exp.(-0.7 .+ 0.3 .* gz)
        gi = repeat(1:G, inner = nper)
        eta = 0.3 .+ 0.5 .* x1 .+ 0.5 .* x2 .- 0.3 .* x3 .+ sd0[gi] .* b0[gi] .+ x1 .* sd1[gi] .* b1[gi]
        mu = exp.(clamp.(eta, -8, 8)); size_ = 2.0
        y = Float64[]
        for m in mu
            p = size_ / (size_ + m)
            # Gamma-Poisson mixture draw
            lam = m * (-log(rand(rng))) ; lam = 0.0
            k = 0; # negative binomial via sum of geometric: use inverse transform instead
            u = rand(rng); pr = p^size_; cdf = pr; k = 0
            while u > cdf && k < 10_000
                k += 1; pr *= (k - 1 + size_) / k * (1 - p); cdf += pr
            end
            push!(y, Float64(k))
        end
        d = (; y, x1, x2, x3, z = gz[gi], g)
        t = @elapsed r = try
            fit = drm(f, NegBinomial2(); data = d, coords, se = false)
            (loglikelihood(fit), fit.converged)
        catch err
            (NaN, false)
        end
        @printf(io, "%d\t%.6f\t%s\t%.1f\n", seed, r[1], r[2], t); flush(io)
    end
end
