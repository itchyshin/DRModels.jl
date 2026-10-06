# Class-1 twin-parity receipt: DRModels side.
#
# Reads the fixtures written by native_fit.R and fits each cell three ways:
#   Laplace  : drm(...; marginal = :Laplace)   (TMB convention; the twin target)
#   Default  : drm(...)                        (per-group adaptive GHQ, 5 nodes, #719)
#   Ref      : the same default integrator at 61 nodes per group (high-accuracy
#              reference for the TRUE marginal likelihood; internal fitters)
# and writes comparison.tsv: per cell and route, df, logLik, |ΔlogLik| and the
# max relative / absolute estimate gap against native drmTMB (native.tsv), plus
# each route's gap to the 61-node reference.
#
#   JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project=. \
#     docs/dev-log/evidence/class1-laplace-parity/julia_fit.jl
using DRModels
using StatsModels: ConstantTerm
using Printf
const D = DRModels
const DIR = joinpath("docs", "dev-log", "evidence", "class1-laplace-parity")

function readfix(path)
    lines = readlines(path)
    hdr = Symbol.(replace.(split(lines[1], ","), "\"" => ""))
    cols = [String[] for _ in hdr]
    for l in lines[2:end], (j, v) in enumerate(split(l, ","))
        push!(cols[j], replace(v, "\"" => ""))
    end
    vals = map(cols) do c
        all(v -> tryparse(Float64, v) !== nothing, c) ? parse.(Float64, c) : c
    end
    return NamedTuple{Tuple(hdr)}(Tuple(vals))
end
readtsv(path) = (ls = readlines(path); h = split(ls[1], '\t');
                 [Dict(zip(h, split(l, '\t'))) for l in ls[2:end]])

# cell => (family, formula)
const CELLS = [
    ("poisson_709",   Poisson(),        bf(@formula(y ~ 1 + (1 | g)))),
    ("beta_710",      D.Beta(),         bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1))),
    ("binary_713",    D.Binomial(),     bf(@formula(y ~ 1 + x + (1 | g)))),
    ("bacteria_713",  D.Binomial(),     bf(@formula(y ~ 1 + week + (1 | g)))),
    ("cbind_713",     D.Binomial(),     bf(@formula(cbind(succ, fail) ~ 1 + x + (1 | g)))),
    ("student_714",   D.Student(),      bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))),
    ("student_big_a", D.Student(),      bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))),
    ("student_big_b", D.Student(),      bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))),
    ("nbinom2_715",   NegBinomial2(),   bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))),
    ("gamma_716",     D.Gamma(),        bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1))),
]

# High-accuracy reference: the default per-group adaptive quadrature at K nodes.
function reffit(f, fam, data; K = 61, g_tol = 1e-8)
    rhs = Dict(f.forms)
    fixed_mu, re, _, _ = D._split_ranef(rhs[:mu])
    grp = re[1][2]; gidx, G = D._group_index(getproperty(data, grp))
    y, Xμ, nmμ = D._design(f.response, fixed_mu, data)
    sig() = D._design(f.response, get(rhs, :sigma, ConstantTerm(1)), data)
    if fam isa Poisson
        return D._fit_poisson_ranef(fam, y, Xμ, gidx, G, nmμ, grp, g_tol; K = K)
    elseif fam isa D.Binomial
        s, ntr = D._binomial_response(f, data)
        return D._fit_binomial_ranef(fam, s, ntr, Xμ, gidx, G, nmμ, grp, g_tol; nq = K)
    elseif fam isa D.Student
        _, Xσ, nmσ = sig()
        _, Xν, nmν = D._design(f.response, get(rhs, :nu, ConstantTerm(1)), data)
        return D._fit_student_ranef(fam, y, Xμ, Xσ, Xν, gidx, G, nmμ, nmσ, nmν, grp, g_tol; K = K)
    else
        _, Xσ, nmσ = sig()
        fn = fam isa D.Beta ? D._fit_beta_ranef : fam isa D.Gamma ? D._fit_gamma_ranef :
             D._fit_negbin2_ranef
        return fn(fam, y, Xμ, Xσ, gidx, G, nmμ, nmσ, grp, g_tol; K = K)
    end
end

gaps(est, ref) = (maximum(abs.(est .- ref) ./ max.(abs.(ref), 1e-8)), maximum(abs.(est .- ref)))

native = readtsv(joinpath(DIR, "native.tsv"))
rows = String["cell\troute\tdf\tconverged\tlogLik\tabs_dlogLik_native\tmax_rel_dest_native\tmax_abs_dest_native\t" *
              "abs_dlogLik_ref61\tmax_rel_dest_ref61\tsd_mu\tsd_mu_rel_native"]
for (cell, fam, f) in CELLS
    nr = filter(r -> r["cell"] == cell, native)
    isempty(nr) && continue
    d = readfix(joinpath(DIR, "fixtures", cell * ".csv"))
    nat = parse.(Float64, [r["estimate"] for r in nr]); llN = parse(Float64, nr[1]["logLik"])
    t0 = time()
    fits = Dict("Laplace" => drm(f, fam; data = d, marginal = :Laplace, se = false),
                "Default" => drm(f, fam; data = d, se = false),
                "Ref61"   => reffit(f, fam, d))
    ref = fits["Ref61"]
    for route in ("Laplace", "Default", "Ref61")
        ft = fits[route]
        est = ft.theta; @assert length(est) == length(nat) "$cell $route"
        rl, ab = gaps(est, nat)
        rr, _ = gaps(est, ref.theta)
        push!(rows, @sprintf("%s\t%s\t%d\t%s\t%.10f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.8f\t%.3e",
                             cell, route, dof(ft), ft.converged, loglik(ft), abs(loglik(ft) - llN), rl, ab,
                             abs(loglik(ft) - loglik(ref)), rr, exp(est[end]),
                             abs(exp(est[end]) - exp(nat[end])) / exp(nat[end])))
    end
    @printf("%s done in %.1fs  native sd %.6f\n", cell, time() - t0, exp(nat[end]))
end
open(joinpath(DIR, "comparison.tsv"), "w") do io
    foreach(r -> println(io, r), rows)
end
println(join(rows, "\n"))
