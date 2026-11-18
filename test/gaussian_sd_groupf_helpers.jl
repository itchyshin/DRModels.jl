# Shared by test_gaussian_sd_groupf.jl and tools/gaussian_sd_groupf_compare.jl: the models of
# docs/dev-log/2026-11-09-gaussian-sd-groupf.md, a Julia fit of each on a committed dataset, and the
# comparison with the committed drmTMB 0.8.1 reference (test/data/gaussian_sd_groupf/r_reference.tsv).
# The comparison itself (`gsd_compare`) is the wave-9 one, so thresholds cannot drift between the notes.
using DRModels, DelimitedFiles, Statistics
isdefined(@__MODULE__, :gsd_compare) || include("gaussian_sd_gaps_helpers.jl")

const GSF_DIR = joinpath(@__DIR__, "data", "gaussian_sd_groupf")

# id => (response, mean random terms, sigma formula, blocks [(group, members)], independent terms
#        [(group, :intercept | slope)], SD formulas [(group, member | nothing, predictor)])
const GSF_MODELS = Dict(
    "g_blk_slope"        => (:y, "(1 + x1 | g) + (0 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1])], [(:g2, :x2)], []),
    "g_blk_slope_sd"     => (:y, "(1 + x1 | g) + (0 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1])], [(:g2, :x2)], [(:g2, :x2, "z2")]),
    "g_blk_indep"        => (:y, "(1 + x1 | g) + (0 + x3 | g)", "sigma ~ 1",
                             [(:g, [:intercept, :x1])], [(:g, :x3)], []),
    "g_two_blk"          => (:y, "(1 + x1 | g) + (1 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1]), (:g2, [:intercept, :x2])], [], []),
    "g_two_blk_sd"       => (:y, "(1 + x1 | g) + (1 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1]), (:g2, [:intercept, :x2])], [],
                             [(:g, :x1, "z"), (:g2, :x2, "z2")]),
    "g_blk_int_slope"    => (:y, "(1 + x1 | g) + (1 | g2) + (0 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1])], [(:g2, :intercept), (:g2, :x2)], []),
    "g_blk_int_slope_sd" => (:y, "(1 + x1 | g) + (1 | g2) + (0 + x2 | g2)", "sigma ~ 1",
                             [(:g, [:intercept, :x1])], [(:g2, :intercept), (:g2, :x2)],
                             [(:g2, :intercept, "z2"), (:g2, :x2, "z2")]),
    "f_corr"             => (:yf, "(1 | p | g)", "sigma ~ 1 + (1 | p | g)", [], [], [(:g, nothing, "z")]),
    "f_indep"            => (:yf, "(1 | g)", "sigma ~ 1 + (1 | g)", [], [], [(:g, nothing, "z")]),
    "f_cross"            => (:yf, "(1 | g)", "sigma ~ 1 + (1 | g2)", [], [], [(:g, nothing, "z")]),
)
const GSF_ORDER = ["g_blk_slope", "g_blk_slope_sd", "g_blk_indep", "g_two_blk", "g_two_blk_sd",
                   "g_blk_int_slope", "g_blk_int_slope_sd", "f_corr", "f_indep", "f_cross"]
gsf_group(id) = startswith(id, "f_") ? "F mean sd() + sigma RE" : "G block beside slopes"

function gsf_read(ds)
    raw, hdr = readdlm(joinpath(GSF_DIR, ds * ".csv"), ','; header = true)
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(vec(hdr))
        v = raw[:, j]
        cols[Symbol(h)] = eltype(v) <: Number || all(x -> x isa Number, v) ? Float64.(v) : String.(v)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

_gsf_member(m) = m === :intercept ? "intercept" : String(m)
function gsf_build(id; extra_sd = String[])
    (resp, re, sigma, _, _, sds) = GSF_MODELS[id]
    xs = resp === :yf ? "x1" : "x1 + x2 + x3"
    parts = Any[gsd_formula("$(resp) ~ $xs + $re"), gsd_formula(sigma)]
    for (grp, mem, pred) in sds
        lhs = mem === nothing ? "sd($grp)" : "sd($grp, coef($(_gsf_member(mem))))"
        push!(parts, gsd_formula(lhs * " ~ " * pred))
    end
    for e in extra_sd
        push!(parts, gsd_formula(e))
    end
    return bf(parts...)
end

# the R spelling of an SD-formula target, as written in r_reference.tsv
gsf_rtarget(grp, mem) = mem === nothing ? "sd($grp)" :
    "sd($grp, coef = \"$(mem === :intercept ? "(Intercept)" : mem)\")"

# Keyed estimates of a Julia fit, matching the keys of r_reference.tsv after `gsf_norm`.
function gsf_extract(fit, id)
    (resp, _, _, blocks, extras, sds) = GSF_MODELS[id]
    out = Dict{String,Float64}()
    out["ll"] = loglik(fit); out["conv"] = fit.converged ? 1.0 : 0.0
    for (nm, v) in zip(first(c[2] for c in fit.coefnames if c[1] === :mu), coef(fit, :mu))
        out["mu:" * nm] = v
    end
    out["sigma:(Intercept)"] = coef(fit, :sigma)[1]
    withf = Dict((g, m) => p for (g, m, p) in sds)
    keys_ = [p.first for p in fit.blocks if p.first ∉ (:mu, :sigma)]
    names_of(k) = first(cn[2] for cn in fit.coefnames if cn[1] === k)
    if resp === :yf
        mean_key, sigma_key = keys_[1], keys_[2]
        for (nm, v) in zip(names_of(mean_key), coef(fit, mean_key))
            out["sdf:sd(g):" * nm] = v
        end
        out["sd:sigma"] = exp(coef(fit, sigma_key)[1])
        any(==(:cor), keys_) && (out["cor:mu_sigma"] = tanh(coef(fit, :cor)[1]))
        return out
    end
    pos = 0
    for (grp, members) in blocks
        for m in members
            pos += 1
            c = coef(fit, keys_[pos])
            if haskey(withf, (grp, m))
                for (nmi, v) in zip(names_of(keys_[pos]), c)
                    out["sdf:" * gsf_rtarget(grp, m) * ":" * nmi] = v
                end
            else
                out["sd:blk:$grp:$(m === :intercept ? "(Intercept)" : m)"] = exp(c[1])
            end
        end
        pos += 1
        q = length(members)
        Rm = DRModels._q34_corr_from_theta(coef(fit, keys_[pos]), q)
        for i in 2:q, j in 1:(i-1)
            mj = members[j] === :intercept ? "(Intercept)" : String(members[j])
            out["cor:$grp:$mj,$(members[i])"] = Rm[i, j]
        end
    end
    for (grp, m) in extras
        pos += 1
        c = coef(fit, keys_[pos])
        if haskey(withf, (grp, m))
            for (nmi, v) in zip(names_of(keys_[pos]), c)
                out["sdf:" * gsf_rtarget(grp, m) * ":" * nmi] = v
            end
        else
            out["sd:ext:$grp:$(m === :intercept ? "(Intercept)" : m)"] = exp(c[1])
        end
    end
    return out
end

# R keys -> comparable keys.
function gsf_norm(key)
    startswith(key, "sd:") || startswith(key, "cor:") || return key
    m = match(r"^sd:mu:\(1 \+ [^|]*\| (\w+)\):(.*)$", key)
    m === nothing || return "sd:blk:$(m.captures[1]):$(m.captures[2])"
    m = match(r"^sd:mu:\(1 \| (\w+)\)$", key)
    m === nothing || return "sd:ext:$(m.captures[1]):(Intercept)"
    m = match(r"^sd:mu:\(0 \+ (\w+) \| (\w+)\)$", key)
    m === nothing || return "sd:ext:$(m.captures[2]):$(m.captures[1])"
    startswith(key, "sd:sigma:") && return "sd:sigma"
    m = match(r"^cor:mu:cor\((.*) \| (\w+)\)$", key)
    m === nothing || return "cor:$(m.captures[2]):$(m.captures[1])"
    startswith(key, "cor:mu_sigma:") && return "cor:mu_sigma"
    return key
end

function gsf_reference()
    raw = readdlm(joinpath(GSF_DIR, "r_reference.tsv"), '\t'; header = false, skipstart = 1)
    ref = Dict{Tuple{String,String},Dict{String,Tuple{Float64,Float64}}}()
    for r in eachrow(raw)
        d = get!(ref, (String(r[1]), String(r[2])), Dict{String,Tuple{Float64,Float64}}())
        est = r[4] isa Number ? Float64(r[4]) : NaN
        se = r[5] isa Number ? Float64(r[5]) : NaN
        d[gsf_norm(String(r[3]))] = (est, se)
    end
    return ref
end
