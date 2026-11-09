# usage: julia --project=<wt> cross_probe.jl <hd> <out> <family nb2|pois> <kind spatial|relmat> <lab 0|1> <seed lo> <seed hi>
using DRModels, DelimitedFiles, Printf, Random, LinearAlgebra, Statistics
using DRModels: Optim
HD, OUT, fam, kind, lab, lo, hi = ARGS[1], ARGS[2], ARGS[3], ARGS[4], parse(Int, ARGS[5]), parse(Int, ARGS[6]), parse(Int, ARGS[7])
cm = readdlm(joinpath(HD, "coords.csv"), ',', Any; skipstart = 1)
ids = String.(cm[:, 1]); xy = Float64.(cm[:, 2:3]); G = length(ids)
coords = Dict(ids[i] => xy[i, :] for i in 1:G)
D = [norm(xy[k, :] - xy[l, :]) for k in 1:G, l in 1:G]
pos = [D[k, l] for k in 1:G for l in (k+1):G if D[k, l] > 0]
C = exp.(-D ./ median(pos)) + 1e-6I; L = cholesky(Symmetric(C)).L
labtxt = lab == 1 ? " | p | " : " | "
mu = "y ~ x1 + x2 + x3 + $(kind)(1 + x1$(labtxt)g)"
sds = ["sd(g, $(kind), coef(intercept)) ~ z", "sd(g, $(kind), coef(x1)) ~ z"]
mkf(s) = eval(Meta.parse("@formula(" * s * ")"))
parts = Any[mkf(mu)]; fam == "nb2" && push!(parts, mkf("sigma ~ 1"))
f = bf(parts..., mkf.(sds)...)
kw = kind == "spatial" ? (; coords) : (; K = Matrix(C))
family = fam == "nb2" ? NegBinomial2() : Poisson()
function gradfd!(nll)
    (g, t) -> (for j in eachindex(t); h = 2e-4*(1+abs(t[j])); e = zeros(length(t)); e[j] = h; g[j] = (nll(t .+ e) - nll(t .- e))/(2h); end; g)
end
open(OUT, "w") do io
    println(io, "seed\tlogLik\tconv\tbest_probe_gain\tsecs")
    for seed in lo:hi
        rng = MersenneTwister(seed); nper = 10; n = G * nper
        g = repeat(ids, inner = nper); gz = rand(rng, G) .* 2 .- 1
        x1, x2, x3 = randn(rng, n), randn(rng, n), randn(rng, n)
        rho = lab == 1 ? 0.3 : 0.0
        e0 = L * randn(rng, G); e1 = L * randn(rng, G); b0 = e0; b1 = rho .* e0 .+ sqrt(1 - rho^2) .* e1
        sd0 = exp.(-0.8 .+ 0.3 .* gz); sd1 = exp.(-0.7 .+ 0.3 .* gz); gi = repeat(1:G, inner = nper)
        eta = 0.3 .+ 0.5 .* x1 .+ 0.5 .* x2 .- 0.3 .* x3 .+ sd0[gi] .* b0[gi] .+ x1 .* sd1[gi] .* b1[gi]
        mu_ = exp.(clamp.(eta, -8, 8)); y = Float64[]
        for m in mu_
            if fam == "pois"
                u = rand(rng); k = 0; pr = exp(-m); cdf = pr
                while u > cdf && k < 10_000; k += 1; pr *= m / k; cdf += pr; end
                push!(y, Float64(k))
            else
                sz = 2.0; p = sz / (sz + m); u = rand(rng); pr = p^sz; cdf = pr; k = 0
                while u > cdf && k < 10_000; k += 1; pr *= (k - 1 + sz) / k * (1 - p); cdf += pr; end
                push!(y, Float64(k))
            end
        end
        d = (; y, x1, x2, x3, z = gz[gi], g)
        t = @elapsed res = try
            fit = drm(f, family; data = d, se = false, kw...)
            ll0 = loglikelihood(fit); best = 0.0
            sdr = [r for (nm, r) in fit.blocks if startswith(String(nm), "sd") || occursin("sd", String(nm))]
            for h1 in (-1.0, 1.0), h0 in (-1.0, 1.0)
                s = copy(fit.theta)
                length(sdr) >= 2 && length(sdr[1]) > 1 && (s[sdr[1][2]] = h0 * 2 / maximum(abs, gz))
                length(sdr) >= 2 && length(sdr[2]) > 1 && (s[sdr[2][2]] = h1 * 2 / maximum(abs, gz))
                od = Optim.OnceDifferentiable(fit.nll, gradfd!(fit.nll), s)
                r = try Optim.optimize(od, s, Optim.LBFGS(linesearch = Optim.LineSearches.BackTracking()), Optim.Options(g_tol = 1e-5, iterations = 250)) catch; nothing end
                r === nothing || (best = max(best, -Optim.minimum(r) - ll0))
            end
            (ll0, fit.converged, best)
        catch err
            println(stderr, "ERR ", sprint(showerror, err)[1:min(end,300)])
            (NaN, false, NaN)
        end
        @printf(io, "%d\t%.6f\t%s\t%.5f\t%.1f\n", seed, res[1], res[2], res[3], t); flush(io)
    end
end
