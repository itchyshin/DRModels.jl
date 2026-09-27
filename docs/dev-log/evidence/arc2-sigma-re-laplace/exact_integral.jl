# Which integrator is closer to the exact marginal? At native drmTMB's θ̂ for
# each fixture, evaluates the marginal log-likelihood three ways: exact (per
# group, adaptive quadrature of the integrand over ±12 posterior SDs around the
# mode, rtol 1e-12), the `:Laplace` objective, and the default GHQ-32 objective.
# Writes exact_integral.tsv. Context only: the parity target is TMB's Laplace
# value, not the exact integral.
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/arc2-sigma-re-laplace/exact_integral.jl
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

function exact_loglik(d, θ, sigx)
    n = length(d.y)
    μ = θ[1] .+ θ[2] .* d.x
    η0 = sigx ? θ[3] .+ θ[4] .* d.x : fill(θ[3], n)
    s = exp(θ[end])
    ll = 0.0
    for lev in unique(d.g)
        idx = findall(==(lev), d.g)
        r = d.y[idx] .- μ[idx]; e0 = η0[idx]; m = length(idx)
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

native = readtsv(joinpath(DIR, "native.tsv"))
rows = String["cell\tlogLik_exact\tlogLik_Laplace\tlogLik_GHQ32\tLaplace_minus_exact\tGHQ32_minus_exact"]
for cell in unique(r["cell"] for r in native)
    nr = filter(r -> r["cell"] == cell, native)
    θ = parse.(Float64, [r["estimate"] for r in nr])
    d = readfix(joinpath(DIR, "fixtures", cell * ".csv"))
    sigx = startswith(cell, "c6")
    sig = sigx ? @formula(sigma ~ 1 + x + (1 | g)) : @formula(sigma ~ 1 + (1 | g))
    f = bf(@formula(y ~ x), sig)
    lap = -drm(f, Gaussian(); data = d, marginal = :Laplace).nll(θ)
    ghq = -drm(f, Gaussian(); data = d).nll(θ)
    ex = exact_loglik(d, θ, sigx)
    push!(rows, @sprintf("%s\t%.8f\t%.8f\t%.8f\t%.4f\t%.4f", cell, ex, lap, ghq, lap - ex, ghq - ex))
    println(rows[end])
end
write(joinpath(DIR, "exact_integral.tsv"), join(rows, "\n") * "\n")
