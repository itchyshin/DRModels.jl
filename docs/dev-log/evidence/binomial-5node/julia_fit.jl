# Binomial `(1 | g)` default 3 -> 5 nodes (#908): before / after / 61-node reference.
# Reuses the class-1 parity fixtures (MASS::bacteria, a binary cell, a cbind cell).
#   OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/binomial-5node/julia_fit.jl
using DRModels
using Printf
const D = DRModels
const FIX = joinpath("docs", "dev-log", "evidence", "class1-laplace-parity", "fixtures")

function readfix(path)
    lines = readlines(path)
    hdr = Symbol.(replace.(split(lines[1], ","), "\"" => ""))
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    vals = map(c -> all(v -> tryparse(Float64, v) !== nothing, c) ? parse.(Float64, c) : c, cols)
    return NamedTuple{Tuple(hdr)}(Tuple(vals))
end

const CELLS = [
    ("binary_713",   bf(@formula(y ~ 1 + x + (1 | g)))),
    ("bacteria_713", bf(@formula(y ~ 1 + week + (1 | g)))),
    ("cbind_713",    bf(@formula(cbind(succ, fail) ~ 1 + x + (1 | g)))),
]

function kfit(f, data, K; g_tol = 1e-8)
    rhs = Dict(f.forms)
    fixed_mu, re, _, _ = D._split_ranef(rhs[:mu])
    grp = re[1][2]; gidx, G = D._group_index(getproperty(data, grp))
    _, Xμ, nmμ = D._design(f.response, fixed_mu, data)
    s, ntr = D._binomial_response(f, data)
    return D._fit_binomial_ranef(D.Binomial(), s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; nq = K)
end

println("_BINOMIAL_RANEF_AGHQ_K = ", D._BINOMIAL_RANEF_AGHQ_K)
println("cell\tK\tsd_mu\tlogLik\tdlogLik_ref61\tsd_rel_ref61\tmax_rel_beta_ref61\tsecs")
for (cell, f) in CELLS
    d = readfix(joinpath(FIX, cell * ".csv"))
    ref = kfit(f, d, 61)
    for K in (1, 3, 5, 7, 9, 61)
        t = @elapsed ft = K == 61 ? ref : kfit(f, d, K)
        sd = exp(ft.theta[end]); sdr = exp(ref.theta[end])
        mb = maximum(abs.(ft.theta[1:end-1] .- ref.theta[1:end-1]) ./ abs.(ref.theta[1:end-1]))
        @printf("%s\t%d\t%.5f\t%.6f\t%.2e\t%.2e\t%.2e\t%.2f\n", cell, K, sd, loglik(ft),
                loglik(ft) - loglik(ref), (sd - sdr) / sdr, mb, t)
    end
    dflt = drm(f, D.Binomial(); data = d, se = false)
    @printf("%s\tdrm-default\t%.5f\t%.6f\tdefault==K5 theta: %s\n", cell, exp(dflt.theta[end]), loglik(dflt),
            dflt.theta == kfit(f, d, D._BINOMIAL_RANEF_AGHQ_K).theta)
end
