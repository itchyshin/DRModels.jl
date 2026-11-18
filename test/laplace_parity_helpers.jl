# Shared by test_laplace_parity.jl and tools/laplace_parity_compare.jl: Julia fits with
# `marginal = :Laplace` of gap-map cells whose default route integrates by adaptive quadrature,
# compared with the committed drmTMB 0.8.1 (ML) references. Agreement rule, written before the
# reference numbers of the single-term cells were read (docs/dev-log/2026-11-09-laplace-parity.md):
#   |dlogLik| <= 1e-4; every mean / sigma / nu coefficient within 1e-3 of R's standard error;
#   every random-effect log SD (or SD-formula coefficient) within 5e-3.
include(joinpath(@__DIR__, "nongauss_indep_helpers.jl"))

const LP_DIR = joinpath(@__DIR__, "fixtures", "laplace_parity")

# cell => ngi reference entry plus `coefs`: block (:mu, :sigma, :nu) => [(estimate, se), ...] in R's order
function lp_reference()
    ref = Dict{String,Any}()
    for (dir, file) in ((LP_DIR, "r_reference.csv"), (LP_DIR, "r_reference_corr.csv"), (NGI_DIR, "r_reference.csv"))
        raw, _ = readdlm(joinpath(dir, file), ','; header = true)
        part = Dict{String,Any}()
        ngi_read_reference!(part, raw)
        coefs = Dict{String,Dict{Symbol,Vector{Tuple{Float64,Float64}}}}()
        for r in eachrow(raw)
            m = match(r"^(mu|sigma|nu):", String(r[2]))
            m === nothing && continue
            blk = get!(() -> Dict{Symbol,Vector{Tuple{Float64,Float64}}}(), coefs, String(r[1]))
            push!(get!(() -> Tuple{Float64,Float64}[], blk, Symbol(m.captures[1])), (Float64(r[3]), r[4] isa Number ? Float64(r[4]) : NaN))
        end
        for (id, e) in part
            haskey(ref, id) || (ref[id] = merge(e, (; coefs = coefs[id])))
        end
    end
    return ref
end

# the formula of a cell; `corr` is one correlated block `(1 + x1 [+ x2 [+ x3]] | g)`, the other kinds are ngi_build's
function lp_build(id)
    fam, kind, q, pat = ngi_parts(id)
    kind == "corr" || return ngi_build(id)
    parts = Any[ngi_formula("y ~ x1 + x2 + x3 + (" * join(["1"; ["x1", "x2", "x3"][1:q-1]], " + ") * " | g)")]
    get(NGI_HAS_SIGMA, fam, true) && push!(parts, ngi_formula("sigma ~ 1"))
    for j in ngi_sd_members(id)
        coef = NGI_COEFS[j]
        push!(parts, ngi_formula("sd(g, coef($(coef == "(Intercept)" ? "intercept" : coef))) ~ " * (pat == "one" ? "1" : "z")))
    end
    return bf(parts...)
end

# largest gaps against one reference entry; fixed effects are scaled by R's standard error.
# A correlated block (`corr`) compares only the SD-formula coefficients: the constant SDs and the
# correlations live in other parameter blocks whose layout differs between the engines.
function lp_compare(id, fit, r)
    dll = abs(loglik(fit) - r.loglik)
    fe = 0.0
    for (blk, rows) in r.coefs
        jv = coef(fit, blk)
        length(jv) == length(rows) || return (dll = dll, fe = Inf, sdlog = Inf, boundary = false)
        fe = max(fe, maximum(abs(jv[k] - rows[k][1]) / (isfinite(rows[k][2]) ? rows[k][2] : 1.0) for k in eachindex(jv)))
    end
    sdblocks = [first(p) for p in fit.blocks if first(p) in (:sd, :resd) || startswith(String(first(p)), "sdtarget_")]
    jsd = reduce(vcat, [coef(fit, b) for b in sdblocks]; init = Float64[])
    corr = ngi_parts(id)[2] == "corr"
    rsd = corr ? reduce(vcat, [r.sdreg[NGI_COEFS[j]] for j in ngi_sd_members(id)]; init = Float64[]) : ngi_ref_sdpar(id, r)
    sdlog = isempty(rsd) ? 0.0 : maximum(abs.(jsd .- rsd))
    boundary = any(<(-6), jsd) || any(<(-6), rsd)
    return (dll = dll, fe = fe, sdlog = sdlog, boundary = boundary)
end
lp_agrees(c) = c.dll <= 1e-4 && c.fe <= 1e-3 && (c.sdlog <= 5e-3 || c.boundary)

lp_fit(id; marginal = :Laplace) =
    drm(lp_build(id), ngi_family(id); data = ngi_data(split(id, '|')[1]), marginal)
