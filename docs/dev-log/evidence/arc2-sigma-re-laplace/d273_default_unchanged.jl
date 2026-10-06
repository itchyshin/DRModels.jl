# D-273 guard for the default path: fits the seven fixtures with the default
# `marginal` (GHQ-32) and prints logLik, θ (repr), converged and hash(vcov).
# Run once under this branch and once under `git archive origin/main`; the two
# outputs must be byte-identical.
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=<tree> \
#     docs/dev-log/evidence/arc2-sigma-re-laplace/d273_default_unchanged.jl <fixture dir>
using DRModels
dir = ARGS[1]
function readfix(path)
    lines = readlines(path)
    hdr = replace.(split(lines[1], ","), "\"" => "")
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    col(n) = cols[findfirst(==(n), hdr)]
    return (y = parse.(Float64, col("y")), x = parse.(Float64, col("x")), g = col("g"))
end
for file in sort(readdir(dir))
    endswith(file, ".csv") || continue
    cell = replace(file, ".csv" => "")
    d = readfix(joinpath(dir, file))
    sig = startswith(cell, "c6") ? @formula(sigma ~ 1 + x + (1 | g)) : @formula(sigma ~ 1 + (1 | g))
    fit = drm(bf(@formula(y ~ x), sig), Gaussian(); data = d)
    println(cell, "\t", repr(loglik(fit)), "\t", repr(fit.theta), "\t", fit.converged, "\t",
            hash(fit.vcov), "\t", fit.marginal)
end
