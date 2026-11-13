# How `reject_points.tsv` was made (Julia; run from the repository root after `using Pkg; Pkg.activate("test")`).
#
# The six rejection returns of the ordinary bivariate route (`gaussian_bivariate_re.jl`) each return the
# one finite value 1e18. Three of them were not reached by any test: ML negative sum of squares
# (`ml217`), ML non-finite value from finite pieces (`ml221`), and REML negative sum of squares
# (`reml236`; the names are the source lines of the time). This script regenerates 20000 crafted
# trial points for the r7080 data (log sigmas in [-14, 1], |rho| linear predictor in [3, 40], the group
# Cholesky entries perturbed, and for every fourth point the fixed effects scaled by 1e100 to 1e200),
# evaluates the route's own objective at each, and prints the chosen ones. The chosen points are those
# where the unchanged code returns exactly 1e18 and the same code with that single return changed
# to `Inf` returns `Inf` (checked by editing each site in a scratch copy). Selected indices:
const SELECTED = Dict("ml217" => [4, 8, 16, 40, 347, 387, 438, 941, 989, 1147, 1482, 1517],
                      "ml221" => [12, 28, 36, 84],
                      "reml236" => [11340, 15284, 16887])
using DRModels, DelimitedFiles, Logging, Random
cap = Ref{Any}(nothing); capx = Ref{Any}(nothing)
# capture the objective the route hands to its start selection, without running a fit
DRModels._biv_re_best_start(run, obj, x0::AbstractVector) = (cap[] = obj; capx[] = copy(x0); error("captured"))
f = bf(mu1 = @formula(y1 ~ x1 + (1 | p | g)), mu2 = @formula(y2 ~ x2 + (1 | p | g)),
       sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
m, h = readdlm(joinpath(@__DIR__, "r7080.csv"), ',', Any; header = true)
cols = Pair{Symbol,Any}[]
for (j, n) in enumerate(String.(vec(h)))
    c = m[:, j]; push!(cols, Symbol(n) => n == "g" ? String.(c) : Float64.(c))
end
for (method, keys) in ((:ML, ("ml217", "ml221")), (:REML, ("reml236",)))
    try with_logger(NullLogger()) do; drm(f, Gaussian(); data = (; cols...), method = method); end catch e; end
    obj = cap[]; x0 = capx[]; L = length(x0); pβ = method === :ML ? L - 6 : 0
    rng = Xoshiro(20261113)
    wanted = Dict(i => k for k in keys for i in SELECTED[k])
    for i in 1:20000
        x = copy(x0); mode = mod1(i, 4)
        x[pβ+1] = -15 * rand(rng) + 1; x[pβ+2] = -15 * rand(rng) + 1
        x[pβ+3] = (rand(rng) < 0.5 ? -1 : 1) * (3 + 37 * rand(rng))
        x[(pβ+4):L] .+= randn(rng, 3) .* (mode == 1 ? 0.1 : mode == 2 ? 1.0 : 3.0)
        if method === :ML
            x[1:pβ] .*= 1 .+ 0.2 .* randn(rng, pβ)
            mode == 4 && (x[1:pβ] .*= 10.0^(100 + 100 * rand(rng)))
        end
        haskey(wanted, i) && println(wanted[i], "\t", i, "\t", join(repr.(x), ","))
    end
end
