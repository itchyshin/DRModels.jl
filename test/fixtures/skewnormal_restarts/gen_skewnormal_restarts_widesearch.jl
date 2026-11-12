# gen_skewnormal_restarts_widesearch.jl -- best known log-likelihood by a wide search (40 starts).
#
#   julia --project=test test/fixtures/skewnormal_restarts/gen_skewnormal_restarts_widesearch.jl
#
# For each data set: L-BFGS (g_tol 1e-8) from 40 fixed starts (the package default start with the
# slant intercept on a grid from -8 to 8, the fitted optimum jittered, and seeded random starts)
# on the package's own objective (`fit.nll`). Prints the best log-likelihood among starts that
# end with a positive definite Hessian, and how many starts reach it. Used for cell_y_g, where
# drmTMB stops at a worse optimum (-243.9303305893) and the best known value is -243.8767839692.
using DRModels, DelimitedFiles, LinearAlgebra, Random, StableRNGs, Printf
import ForwardDiff
const Optim = DRModels.Optim
const DIR = @__DIR__
function load(name)
    m, h = readdlm(joinpath(DIR, name), ',', Any; header = true)
    h = Symbol.(vec(h)); NamedTuple{Tuple(h)}(Tuple(Float64.(m[:, j]) for j in eachindex(h)))
end
cases = [("sim159.csv", bf(@formula(y ~ x), @formula(sigma ~ x), @formula(nu ~ x))),
         ("cell_y_sk.csv", bf(@formula(y_sk ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1))),
         ("cell_y_g.csv", bf(@formula(y_g ~ x1), @formula(sigma ~ x2), @formula(nu ~ x1)))]
for (file, f) in cases
    d = load(file)
    fit = drm(f, SkewNormal(); data = d, control = DRMControl(maxit = 1000))   # default start alone
    nll = fit.nll; th = fit.theta; p = length(th)
    iα = p - length(coef(fit, :nu)) + 1
    rng = StableRNG(12345)
    starts = Vector{Float64}[]
    for a in range(-8, 8; length = 17)
        t = copy(th); t[iα:end] .= 0.0; t[iα] = a; push!(starts, t)
    end
    for k in 1:3; push!(starts, th .+ 0.5 .* max.(abs.(th), 1) .* randn(rng, p)); end
    while length(starts) < 40
        t = copy(th); t[iα] = 16rand(rng) - 8; t[iα+1:end] .= 1.5 .* randn(rng, p - iα)
        t[1:iα-1] .+= 0.3 .* randn(rng, iα - 1); push!(starts, t)
    end
    lls = Float64[]
    for t in starts
        r = try Optim.optimize(nll, t, Optim.LBFGS(), Optim.Options(g_tol = 1e-8); autodiff = :forward) catch; nothing end
        r === nothing && continue
        H = ForwardDiff.hessian(nll, Optim.minimizer(r))
        ok = Optim.converged(r) && Optim.g_converged(r) && all(isfinite, H) && isposdef(Symmetric(H))
        ok && push!(lls, -Optim.minimum(r))
    end
    best = maximum(lls)
    @printf("%-15s starts %d  usable %d  best %.10f  reaching best (1e-6): %d\n", file, length(starts), length(lls), best, count(>=(best - 1e-6), lls))
end
