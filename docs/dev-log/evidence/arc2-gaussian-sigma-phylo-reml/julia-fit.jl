# Arc 2 receipt: DRModels.jl side of the Gaussian location-scale phylo()-on-sigma
# REML/ML comparison (plus the mean-only phylo neighbour). Reads the fixtures native-fit.R wrote (data CSV + unit-height
# Newick), fits the SAME formulas through the public `drm()` exactly as the drmTMB
# bridge calls it (sigma-only: the asymmetric route; mu + sigma: `phylo_coupled =
# true`, the block native fits, with the mean-scale phylo correlation), and writes
# julia.tsv plus compare.tsv (native vs julia, per cell).
#
# Run from the DRModels.jl root:
#   JULIA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/arc2-gaussian-sigma-phylo-reml/julia-fit.jl
# ARC2_TAG=<tag> writes julia-<tag>.tsv instead and skips compare.tsv; used with
# `--project=<pristine base checkout>` to bank the before-change values
# (julia-base.tsv), so the unchanged ML and mean-only rows are shown identical.
using DRModels, Printf
const HERE = @__DIR__
const TAG = get(ENV, "ARC2_TAG", "")
const OUT = TAG == "" ? "julia.tsv" : "julia-$TAG.tsv"

function read_fixture(fx)
    lines = readlines(joinpath(HERE, "fixture-$fx.csv"))
    rows = [split(l, ",") for l in lines[2:end]]
    y  = [parse(Float64, r[1]) for r in rows]
    x  = [parse(Float64, r[2]) for r in rows]
    sp = [String(strip(r[3], '"')) for r in rows]
    return (; y, x, sp), String(strip(read(joinpath(HERE, "fixture-$fx.nwk"), String)))
end

function read_tsv(path)
    lines = readlines(path)
    hdr = split(lines[1], '\t')
    [Dict(zip(hdr, split(l, '\t'))) for l in lines[2:end]]
end

shapes = [
    ("mu_only",    bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ 1)), false),
    ("sigma_only", bf(@formula(y ~ x), @formula(sigma ~ phylo(1 | sp))), false),
    ("mu_sigma",   bf(@formula(y ~ x + phylo(1 | sp)), @formula(sigma ~ phylo(1 | sp))), true),
]
fmt(v) = isnan(v) ? "NA" : @sprintf("%.8f", v)
rows = Vector{Vector{String}}()
for fx in ("F1", "F2")
    data, nwk = read_fixture(fx)
    for (shape, form, coupled) in shapes, method in (:ML, :REML)
        fit = nothing
        t0 = time()
        try
            fit = drm(form, Gaussian(); data = data, tree = nwk, method = method,
                      phylo_coupled = coupled, g_tol = 1e-8)
        catch err
            msg = replace(first(sprint(showerror, err), 120), r"\s+" => " ")
            push!(rows, vcat([fx, shape, String(method), "julia", string(length(data.y)), "NA"],
                             fill("NA", 10), ["ERROR: " * msg, "NA"]))
            println(join(rows[end], "\t"))
            continue
        end
        t = time() - t0
        cμ = coef(fit, :mu); cσ = coef(fit, :sigma)
        sd_mu, sd_sig, cor = if coupled
            fit.scales[:lambda_sd_mu][1], fit.scales[:lambda_sd_sigma][1], fit.scales[:lambda_cor][1]
        elseif shape == "mu_only"
            re_sd(fit)[:sp], NaN, NaN
        else
            NaN, exp(coef(fit, :resd_sigma)[1]), NaN
        end
        push!(rows, [fx, shape, String(method), "julia", string(length(data.y)), string(dof(fit)),
                     fmt(loglik(fit)), fmt(cμ[1]), fmt(cμ[2]), fmt(cσ[1]),
                     fmt(sd_mu), fmt(sd_sig), fmt(cor),
                     fmt(stderror(fit)[1]), fmt(stderror(fit)[2]), fmt(stderror(fit)[3]),
                     string(is_converged(fit)), @sprintf("%.2f", t)])
        println(join(rows[end], "\t"))
    end
end
hdr = ["fixture", "shape", "estimator", "engine", "n", "df", "logLik", "mu_intercept", "mu_x",
       "sigma_intercept", "sd_mu", "sd_sigma", "cor", "se_mu_intercept", "se_mu_x",
       "se_sigma_intercept", "converged", "seconds"]
open(joinpath(HERE, OUT), "w") do io
    println(io, join(hdr, '\t'))
    foreach(r -> println(io, join(r, '\t')), rows)
end

TAG == "" || exit(0)

# Native vs Julia, per cell: |ΔlogLik| and the max relative difference over the
# estimates (relative; the correlation, and an SD at the zero boundary on both
# engines, absolutely). SAME requires df equal, |ΔlogLik| <= 1e-6 and every
# estimate within 1e-5.
native = read_tsv(joinpath(HERE, "native.tsv"))
julia  = read_tsv(joinpath(HERE, "julia.tsv"))
key(r) = (r["fixture"], r["shape"], r["estimator"])
jmap = Dict(key(r) => r for r in julia)
pnum(s) = s == "NA" ? NaN : parse(Float64, s)
open(joinpath(HERE, "compare.tsv"), "w") do io
    println(io, join(["fixture", "shape", "estimator", "df_native", "df_julia", "logLik_native",
                      "logLik_julia", "abs_dlogLik", "max_rel_d_estimate", "worst_estimate",
                      "max_rel_d_se_beta", "native_converged", "verdict"], '\t'))
    for nr in native
        jr = jmap[key(nr)]
        dll = abs(pnum(nr["logLik"]) - pnum(jr["logLik"]))
        worst = 0.0; wname = ""
        for c in ("mu_intercept", "mu_x", "sigma_intercept", "sd_mu", "sd_sigma", "cor")
            a = pnum(nr[c]); b = pnum(jr[c])
            (isnan(a) && isnan(b)) && continue
            # Relative, except the correlation (absolute) and an SD at the zero
            # boundary on both engines (both < 1e-3: absolute).
            d = (c == "cor" || (startswith(c, "sd_") && abs(a) < 1e-3 && abs(b) < 1e-3)) ?
                abs(a - b) : abs(a - b) / max(abs(a), 1e-8)
            d > worst && (worst = d; wname = c)
        end
        # Wald SEs of the three fixed effects: REPORTED, not part of the verdict.
        dse = maximum(abs(pnum(nr[c]) - pnum(jr[c])) / abs(pnum(nr[c]))
                      for c in ("se_mu_intercept", "se_mu_x", "se_sigma_intercept"))
        isfinite(dse) || (dse = NaN)   # native vcov unavailable (non-PD sdreport)
        same = nr["df"] == jr["df"] && dll <= 1e-6 && worst <= 1e-5
        # A variance at the zero boundary on both engines (flat likelihood): the
        # SD agrees only absolutely; label it rather than hide it.
        boundary = !same && nr["df"] == jr["df"] && dll <= 1e-6 && startswith(wname, "sd_") &&
                   abs(pnum(nr[wname])) < 1e-3 && abs(pnum(jr[wname])) < 1e-3
        println(io, join([nr["fixture"], nr["shape"], nr["estimator"], nr["df"], jr["df"],
                          nr["logLik"], jr["logLik"], @sprintf("%.3e", dll), @sprintf("%.3e", worst),
                          wname, @sprintf("%.3e", dse), nr["converged"], same ? "SAME" : boundary ? "SAME_BOUNDARY_SD" : "DIFFERENT"], '\t'))
    end
end
print(read(joinpath(HERE, "compare.tsv"), String))
