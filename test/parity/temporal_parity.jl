# temporal_parity.jl — shared helpers for the temporal AR1 / OU drmTMB parity
# cells (D-310). Included by test/parity/runparity_temporal.jl (gated,
# DRM_PARITY_TESTS=1) and by test/test_parity_temporal.jl (always on).
#
# Each cell lives in test/parity/temporal/<cell>/: `expected.toml` holds
# drmTMB's GENERATED numbers (logLik, β, process SD, φ or decay, the optional
# `(1 | id)` SD or wave-2 phylogenetic stable SD, σ) and `expected.meta.toml` their provenance (drmTMB SHA, R
# version, the R call). The input data are the CSVs in test/fixtures/temporal/,
# named by `[fit].data_file`. No drmTMB source is involved (GPL → MIT boundary);
# the generator is test/parity/gen_temporal_parity.R.

using DRModels
using TOML
using DelimitedFiles: readdlm

const TEMPORAL_PARITY_ROOT = joinpath(@__DIR__, "temporal")
const TEMPORAL_FIXTURE_ROOT = joinpath(@__DIR__, "..", "fixtures", "temporal")

temporal_parity_cells() =
    [joinpath(TEMPORAL_PARITY_ROOT, d) for d in sort(readdir(TEMPORAL_PARITY_ROOT))
     if isfile(joinpath(TEMPORAL_PARITY_ROOT, d, "expected.toml"))]

# Read a fixture CSV into a NamedTuple: the grouping column as String, the AR1
# time column as Int (AR1 needs integer occasions), every other column Float64.
function temporal_parity_data(file::AbstractString; group::AbstractString,
                              time::AbstractString, structure::AbstractString)
    raw, hdr = readdlm(joinpath(TEMPORAL_FIXTURE_ROOT, file), ','; header = true)
    names = String.(strip.(string.(vec(hdr))))
    cols = map(enumerate(names)) do (j, nm)
        v = raw[:, j]
        val = nm == group ? String.(string.(v)) :
              (nm == time && structure == "ar1") ? Int.(Float64.(v)) :
              Float64.(v)
        Symbol(nm) => val
    end
    return NamedTuple(cols)
end

_temporal_time_column(julia_formula) =
    String(match(r"temporal\(1 \| \w+, (\w+), (?:ar1|ou|homtoep)\)", julia_formula).captures[1])

# The tree of a paired phylo() + OU cell (wave 2): `[fit].tree_file` names a
# Newick file in test/fixtures/temporal/; `nothing` for the other cells.
_temporal_tree(f) = haskey(f, "tree_file") ?
    read(joinpath(TEMPORAL_FIXTURE_ROOT, f["tree_file"]), String) : nothing

function _temporal_formula(text::AbstractString)
    ex = Meta.parse(text)
    return Core.eval(@__MODULE__, Expr(:macrocall, Symbol("@formula"), LineNumberNode(0), ex))
end

_reldiff(a, b) = abs(a - b) / max(abs(a), abs(b))

"""
    temporal_parity_check(dir) -> (fit, rows)

Fit cell `dir` with DRModels.jl (ML, as drmTMB) and compare it to the stored
drmTMB numbers. `rows` has one entry per compared quantity:
`(quantity, julia, drmtmb, absdiff, reldiff, pass)`; logLik is judged on the
absolute scale (`atol_loglik`), every parameter on the relative scale
(`rtol_par`).
"""
function temporal_parity_check(dir::AbstractString)
    ex = TOML.parsefile(joinpath(dir, "expected.toml"))
    f = ex["fit"]
    structure = f["structure"]
    jlform = f["julia_formula"]
    data = temporal_parity_data(f["data_file"]; group = f["group"],
                                time = _temporal_time_column(jlform), structure = structure)
    fit = drm(bf(_temporal_formula(jlform), @formula(sigma ~ 1)), Gaussian(); data = data,
              tree = _temporal_tree(f))
    tp = temporal_parameters(fit)

    atol_ll = Float64(ex["tol"]["atol_loglik"])
    rtol = Float64(ex["tol"]["rtol_par"])
    rows = NamedTuple[]
    push_row!(q, j, r, ok) = push!(rows, (quantity = q, julia = j, drmtmb = r,
                                          absdiff = abs(j - r), reldiff = _reldiff(j, r), pass = ok))

    ll_r = Float64(f["loglik"])
    push_row!("logLik", loglik(fit), ll_r, abs(loglik(fit) - ll_r) <= atol_ll)

    names = Dict(p => ns for (p, ns) in fit.coefnames)[:mu]
    β = coef(fit, :mu)
    for (k, v) in ex["coef"]
        nm = replace(k, r"^mu_" => "")
        i = findfirst(==(nm), names)
        i === nothing && error("$(basename(dir)): DRModels.jl has no mean coefficient `$nm` (has $(names))")
        push_row!(k, β[i], Float64(v), _reldiff(β[i], Float64(v)) <= rtol)
    end

    t = ex["temporal"]
    jl = Dict("sd" => tp.sd, "phi" => tp.phi, "decay" => tp.decay,
              "sd_iid" => tp.sd_iid, "sd_phylo" => tp.sd_phylo, "sigma" => tp.sigma)
    # `[tol].boundary`: SDs drmTMB estimated at their zero boundary, where the
    # likelihood is flat and only "both engines put it below 1e-3" is testable.
    boundary = get(ex["tol"], "boundary", String[])
    for k in ("sd", "phi", "decay", "sd_iid", "sd_phylo", "sigma")
        haskey(t, k) || continue
        jv = jl[k]
        jv === nothing && error("$(basename(dir)): DRModels.jl reports no `$k`")
        ok = k in boundary ? max(Float64(jv), Float64(t[k])) < 1e-3 :
                             _reldiff(Float64(jv), Float64(t[k])) <= rtol
        push_row!(k, Float64(jv), Float64(t[k]), ok)
    end
    return fit, rows
end
