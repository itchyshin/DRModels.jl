# The default GHQ-32 fit evaluated at its OWN optimum, against native drmTMB and
# the exact marginal. exact_integral.jl evaluates every integrator at native
# drmTMB's estimates, where GHQ-32 is always low for large groups; at the
# default fit's own optimum the reported logLik can instead lie ABOVE both the
# exact marginal and drmTMB's, and its random-effect SD can be far from
# drmTMB's. Fixtures (fixtures_own_optimum/, kept apart from fixtures/ so the
# D-273 script's output is unchanged) and native rows (native_own_optimum.tsv,
# columns in drmTMB order beta_mu, beta_sigma, log_sd_sigma) come from a
# review check with drmTMB 4902180c0, engine = "tmb"; see receipt.md.
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/arc2-sigma-re-laplace/own_optimum.jl
using DRModels, QuadGK, ForwardDiff, Printf

const DIR = joinpath("docs", "dev-log", "evidence", "arc2-sigma-re-laplace")

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
readtsv(path) = (ls = readlines(path); h = split(ls[1], '\t');
                 [Dict(zip(h, split(l, '\t'))) for l in ls[2:end]])

# Same exact marginal as exact_integral.jl (per-group adaptive quadrature).
function exact_loglik(d, θ, sigx)
    n = length(d.y)
    μ = θ[1] .+ θ[2] .* d.x
    η0 = sigx ? θ[3] .+ θ[4] .* d.x : fill(θ[3], n)
    s = exp(θ[end])
    ll = 0.0
    for lev in unique(d.g)
        idx = findall(==(lev), d.g)
        r = d.y[idx] .- μ[idx]; e0 = η0[idx]
        h(b) = sum(-0.5 * log(2π) .- (e0 .+ b) .- 0.5 .* r .^ 2 .* exp.(-2 .* (e0 .+ b))) -
               0.5 * log(2π) - log(s) - 0.5 * b^2 / s^2
        b = 0.0
        for _ in 1:200
            st = ForwardDiff.derivative(h, b) /
                 ForwardDiff.derivative(u -> ForwardDiff.derivative(h, u), b)
            b -= st
            abs(st) < 1e-14 && break
        end
        sd = 1 / sqrt(-ForwardDiff.derivative(u -> ForwardDiff.derivative(h, u), b))
        hb = h(b)
        ll += hb + log(quadgk(u -> exp(h(u) - hb), b - 12sd, b + 12sd; rtol = 1e-12)[1])
    end
    return ll
end

native = readtsv(joinpath(DIR, "native_own_optimum.tsv"))
rows = String["cell\tlogLik_native\tlog_sd_native\tlogLik_Laplace\tlog_sd_Laplace\t" *
              "logLik_default\tlog_sd_default\texact_at_default\tdefault_minus_exact\tdefault_minus_native"]
for cell in unique(r["cell"] for r in native)
    nr = filter(r -> r["cell"] == cell, native)
    d = readfix(joinpath(DIR, "fixtures_own_optimum", cell * ".csv"))
    sigx = startswith(cell, "X")
    f = bf(@formula(y ~ x), sigx ? @formula(sigma ~ 1 + x + (1 | g)) : @formula(sigma ~ 1 + (1 | g)))
    fL = drm(f, Gaussian(); data = d, marginal = :Laplace)
    fD = drm(f, Gaussian(); data = d)
    llN = parse(Float64, nr[1]["logLik"]); sdN = parse(Float64, nr[end]["est"])
    exD = exact_loglik(d, fD.theta, sigx)
    push!(rows, @sprintf("%s\t%.6f\t%.4f\t%.6f\t%.4f\t%.6f\t%.4f\t%.6f\t%+.4f\t%+.4f", cell,
        llN, sdN, loglik(fL), fL.theta[end], loglik(fD), fD.theta[end], exD,
        loglik(fD) - exD, loglik(fD) - llN))
    println(rows[end])
end
write(joinpath(DIR, "own_optimum.tsv"), join(rows, "\n") * "\n")
