# Shared by test_gaussian_sd_gaps.jl and tools/gaussian_sd_gaps_compare.jl: the 15 models of
# docs/dev-log/2026-11-09-gaussian-sd-gaps.md, a Julia fit of each on a committed dataset, and the
# comparison with the committed drmTMB 0.8.1 reference (test/data/gaussian_sd_gaps/r_reference.tsv).
using DRModels, DelimitedFiles, Statistics

const GSD_DIR = joinpath(@__DIR__, "data", "gaussian_sd_gaps")

# id => (model class, random-effect terms, [(coefficient, predictor)] SD formulas, coefficient labels)
# coefficient "g2" is `sd(g2) ~ pred`; others are `sd(g, coef(·)) ~ pred`.
const GSD_MODELS = Dict(
    "ord_q2_lab"   => (:ord_q2, "(1 + x1 | p | g)", [], ["(Intercept)", "x1"]),
    "ord_q3"       => (:q34, "(1 + x1 + x2 | g)", [], ["(Intercept)", "x1", "x2"]),
    "ord_q4_lab"   => (:q34, "(1 + x1 + x2 + x3 | p | g)", [], ["(Intercept)", "x1", "x2", "x3"]),
    "crs_q2"       => (:crs, "(1 + x1 | g) + (1 | g2)", [], ["(Intercept)", "x1"]),
    "crs_q2_lab"   => (:crs, "(1 + x1 | p | g) + (1 | g2)", [], ["(Intercept)", "x1"]),
    "crs_q3"       => (:crs, "(1 + x1 + x2 | g) + (1 | g2)", [], ["(Intercept)", "x1", "x2"]),
    "crs_q4_lab"   => (:crs, "(1 + x1 + x2 + x3 | p | g) + (1 | g2)", [], ["(Intercept)", "x1", "x2", "x3"]),
    "crs_q2_first" => (:crs, "(1 + x1 | g) + (1 | g2)", [("(Intercept)", "z"), ("g2", "z2")], ["(Intercept)", "x1"]),
    "crs_q2_last"  => (:crs, "(1 + x1 | g) + (1 | g2)", [("x1", "z")], ["(Intercept)", "x1"]),
    "crs_q2_z"     => (:crs, "(1 + x1 | g) + (1 | g2)", [("(Intercept)", "z"), ("x1", "z"), ("g2", "z2")], ["(Intercept)", "x1"]),
    "ind_q2_first" => (:indep, "(1 | g) + (0 + x1 | g)", [("(Intercept)", "z")], ["(Intercept)", "x1"]),
    "ind_q2_last"  => (:indep, "(1 | g) + (0 + x1 | g)", [("x1", "z")], ["(Intercept)", "x1"]),
    "ind_q2_z"     => (:indep, "(1 | g) + (0 + x1 | g)", [("(Intercept)", "z"), ("x1", "z")], ["(Intercept)", "x1"]),
    "ind_q2_one"   => (:indep, "(1 | g) + (0 + x1 | g)", [("(Intercept)", "1"), ("x1", "1")], ["(Intercept)", "x1"]),
    "ind_q3_z"     => (:indep, "(1 | g) + (0 + x1 | g) + (0 + x2 | g)", [("(Intercept)", "z"), ("x1", "z"), ("x2", "z")], ["(Intercept)", "x1", "x2"]),
)
const GSD_ORDER = ["ord_q2_lab", "ord_q3", "ord_q4_lab", "crs_q2", "crs_q2_lab", "crs_q3", "crs_q4_lab",
                   "crs_q2_first", "crs_q2_last", "crs_q2_z", "ind_q2_first", "ind_q2_last", "ind_q2_z",
                   "ind_q2_one", "ind_q3_z"]
gsd_group(id) = startswith(id, "ord") ? "C/D ordinary" : startswith(id, "crs") ? (occursin("_q2_", id) && !endswith(id, "_lab") ? "B crossed SD" : "E crossed no-SD") : "A indep SD"

function gsd_read(ds)
    raw, hdr = readdlm(joinpath(GSD_DIR, ds * ".csv"), ','; header = true)
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(vec(hdr))
        v = raw[:, j]
        cols[Symbol(h)] = eltype(v) <: Number || all(x -> x isa Number, v) ? Float64.(v) : String.(v)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

gsd_formula(s) = Core.eval(@__MODULE__, Meta.parse("@formula(" * s * ")"))

function gsd_build(id)
    (_, re, sds, _) = GSD_MODELS[id]
    parts = Any[gsd_formula("y ~ x1 + x2 + x3 + " * re), gsd_formula("sigma ~ 1")]
    for (coef, pred) in sds
        lhs = coef == "g2" ? "sd(g2)" : "sd(g, coef($(coef == "(Intercept)" ? "intercept" : coef)))"
        push!(parts, gsd_formula(lhs * " ~ " * pred))
    end
    return bf(parts...)
end

# the R spelling of a target, as written in r_reference.tsv
gsd_rtarget(coef) = coef == "g2" ? "sd(g2)" : "sd(g, coef = \"$coef\")"

# Keyed estimates of a Julia fit, matching the keys of r_reference.tsv after `gsd_norm`.
function gsd_extract(fit, id)
    (cls, _, sds, members) = GSD_MODELS[id]
    out = Dict{String,Float64}()
    out["ll"] = loglik(fit); out["conv"] = fit.converged ? 1.0 : 0.0
    for (nm, v) in zip(first(c[2] for c in fit.coefnames if c[1] === :mu), coef(fit, :mu))
        out["mu:" * nm] = v
    end
    out["sigma:(Intercept)"] = coef(fit, :sigma)[1]
    withf = Dict(c => p for (c, p) in sds)
    q = length(members)
    if cls === :ord_q2
        L11, L22, L21 = exp(coef(fit, :recov)[1]), exp(coef(fit, :recov)[2]), coef(fit, :recov)[3]
        sd2 = sqrt(L21^2 + L22^2)
        out["sd:blk:(Intercept)"] = L11; out["sd:blk:x1"] = sd2
        out["cor:cor((Intercept),x1 | g)"] = L21 / sd2
    elseif cls in (:q34, :crs)
        keys_ = [p.first for p in fit.blocks if p.first ∉ (:mu, :sigma, :recov)]
        for k in 1:q
            c = coef(fit, keys_[k])
            if haskey(withf, members[k])
                nm = first(cn[2] for cn in fit.coefnames if cn[1] === keys_[k])
                for (nmi, v) in zip(nm, c)
                    out["sdf:" * gsd_rtarget(members[k]) * ":" * nmi] = v
                end
            else
                out["sd:blk:" * members[k]] = exp(c[1])
            end
        end
        θ = coef(fit, :recov)
        Rm = DRModels._q34_corr_from_theta(θ, q)
        for i in 2:q, j in 1:(i-1)
            out["cor:cor($(members[j]),$(members[i]) | g)"] = Rm[i, j]
        end
        if cls === :crs
            k = keys_[q+1]
            c = coef(fit, k)
            if haskey(withf, "g2")
                nm = first(cn[2] for cn in fit.coefnames if cn[1] === k)
                for (nmi, v) in zip(nm, c)
                    out["sdf:sd(g2):" * nmi] = v
                end
            else
                out["sd:g2"] = exp(c[1])
            end
        end
    else  # :indep, one `:sd` block, components in term order
        sdc = coef(fit, :sd); pos = 0
        terms = [c == "(Intercept)" ? "(1 | g)" : "(0 + $c | g)" for c in members]
        for (k, c) in enumerate(members)
            if haskey(withf, c)
                np = withf[c] == "1" ? 1 : 2
                nm = withf[c] == "1" ? ["(Intercept)"] : ["(Intercept)", withf[c]]
                for (i, nmi) in enumerate(nm)
                    out["sdf:" * gsd_rtarget(c) * ":" * nmi] = sdc[pos+i]
                end
                pos += np
            else
                out["sd:" * terms[k]] = exp(sdc[pos+1]); pos += 1
            end
        end
    end
    return out
end

# R keys -> comparable keys.
function gsd_norm(key)
    startswith(key, "sd:") || startswith(key, "sdf:") || startswith(key, "cor:") || return key
    key = replace(key, " | p | " => " | ")
    if startswith(key, "sd:")
        key == "sd:(1 | g2)" && return "sd:g2"
        m = match(r"^sd:\(1 \+ [^|]*\| g\):(.*)$", key)
        m === nothing || return "sd:blk:" * m.captures[1]
    end
    key = replace(key, r"cor:cor\(" => "cor:cor(")
    return key
end

function gsd_reference()
    raw = readdlm(joinpath(GSD_DIR, "r_reference.tsv"), '\t'; header = false, skipstart = 1)
    ref = Dict{Tuple{String,String},Dict{String,Tuple{Float64,Float64}}}()
    for r in eachrow(raw)
        d = get!(ref, (String(r[1]), String(r[2])), Dict{String,Tuple{Float64,Float64}}())
        est = r[4] isa Number ? Float64(r[4]) : NaN
        se = r[5] isa Number ? Float64(r[5]) : NaN
        d[gsd_norm(String(r[3]))] = (est, se)
    end
    return ref
end

# Comparison of one fit against the reference; returns a NamedTuple of the pre-registered quantities.
function gsd_compare(jv, rv)
    dll = abs(jv["ll"] - rv["ll"][1])
    fe = 0.0; sdf = 0.0; sdlog = 0.0; cor = 0.0; unmatched = String[]
    for (k, (est, se)) in rv
        k in ("ll", "conv", "pdhess", "error") && continue
        haskey(jv, k) || (push!(unmatched, "R only: " * k); continue)
        d = abs(jv[k] - est)
        if startswith(k, "mu:") || startswith(k, "sigma:")
            fe = max(fe, d / se)
        elseif startswith(k, "sdf:")
            sdf = max(sdf, d / se)
        elseif startswith(k, "sd:")
            sdlog = max(sdlog, abs(log(jv[k]) - log(est)) / se)
        elseif startswith(k, "cor:")
            cor = max(cor, d)
        end
    end
    for k in keys(jv)
        k in ("ll", "conv") && continue
        haskey(rv, k) || push!(unmatched, "Julia only: " * k)
    end
    flagged = rv["conv"][1] == 0 || rv["pdhess"][1] == 0 || jv["conv"] == 0 ||
              any(k -> startswith(k, "sd:") && (jv[k] < 1e-3 || rv[k][1] < 1e-3), intersect(keys(jv), keys(rv))) ||
              any(k -> startswith(k, "cor:") && (abs(jv[k]) > 0.99 || abs(rv[k][1]) > 0.99), intersect(keys(jv), keys(rv)))
    pass = dll ≤ 1e-4 && fe ≤ 0.01 && sdf ≤ 0.01 && sdlog ≤ 0.05 && cor ≤ 0.01 && isempty(unmatched)
    return (; dll, fe, sdf, sdlog, cor, unmatched, flagged, pass)
end
