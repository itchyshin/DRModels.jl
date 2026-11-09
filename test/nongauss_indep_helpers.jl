# Shared by test_nongauss_indep_routes.jl and tools/nongauss_indep_compare.jl: Julia fits of the
# gap-map `indep` and `slope` cells on committed data, and their comparison with the committed
# drmTMB 0.8.1 reference (test/fixtures/nongauss_indep/r_reference.csv, made by gen_ref.R there).
# Agreement rule, pre-registered in docs/dev-log/2026-11-09-nongauss-indep-design.md:
#   |dlogLik| <= 1e-4, fixed effects (mu, sigma) within 1e-3, random-effect log SDs within 5e-3.
using DRModels, DelimitedFiles

const NGI_DIR = joinpath(@__DIR__, "fixtures", "nongauss_indep")

function ngi_data(family; crossed::Bool = false)
    raw, hdr = readdlm(joinpath(NGI_DIR, crossed ? "data_$(family)_crossed.csv" : "data_$(family).csv"), ','; header = true)
    cols = Dict{Symbol,Any}()
    for (j, h) in enumerate(vec(hdr))
        v = raw[:, j]
        cols[Symbol(h)] = all(x -> x isa Number, v) ? Float64.(v) : String.(v)
    end
    return NamedTuple{Tuple(keys(cols))}(Tuple(values(cols)))
end

const NGI_FAMILIES = Dict("poisson" => Poisson(), "nb2" => NegBinomial2(), "binomial" => Binomial(),
                          "beta" => Beta(), "lognormal" => LogNormal(), "gamma" => Gamma(),
                          "student" => Student(), "tweedie" => Tweedie())
const NGI_HAS_SIGMA = Dict("poisson" => false, "binomial" => false)

ngi_formula(s) = Core.eval(@__MODULE__, Meta.parse("@formula(" * s * ")"))

# cell id: "family|kind|lab0|q<q>|sd-<pattern>|cm0|ML"; pattern none | one | z | first | last
ngi_parts(id) = (p = split(id, '|'); (p[1], p[2], parse(Int, p[4][2:end]), String(p[5][4:end])))
const NGI_COEFS = ["(Intercept)", "x1", "x2", "x3"]

# which term members carry an SD formula, as indices into the term list
function ngi_sd_members(id)
    fam, kind, q, pat = ngi_parts(id)
    pat == "none" && return Int[]
    kind == "crossed" && return [1, 2]
    pat in ("one", "z") && return collect(1:q)
    pat == "first" && return [1]
    return [q]
end

function ngi_build(id)
    fam, kind, q, pat = ngi_parts(id)
    re = kind == "slope" ? "(0 + x1 | g)" : kind == "crossed" ? "(1 | g) + (1 | g2)" :
         "(1 | g)" * join([" + (0 + $v | g)" for v in ["x1", "x2", "x3"][1:q-1]])
    parts = Any[ngi_formula("y ~ x1 + x2 + x3 + " * re)]
    get(NGI_HAS_SIGMA, fam, true) && push!(parts, ngi_formula("sigma ~ 1"))
    for j in ngi_sd_members(id)
        if kind == "crossed"
            push!(parts, ngi_formula(j == 1 ? "sd(g) ~ " * (pat == "one" ? "1" : "z") :
                                              "sd(g2) ~ " * (pat == "one" ? "1" : "z2")))
        else
            coef = NGI_COEFS[kind == "slope" ? 2 : j]
            push!(parts, ngi_formula("sd(g, coef($(coef == "(Intercept)" ? "intercept" : coef))) ~ " *
                                     (pat == "one" ? "1" : "z")))
        end
    end
    return bf(parts...)
end
ngi_family(id) = NGI_FAMILIES[split(id, '|')[1]]

# reference: cell => (loglik, clean, mu, sigma, sd (constant-SD terms, natural scale),
#                     sdreg (coefficient name => [(Intercept), z] of its `sd(g, coef = ..) ~ ..`))
function ngi_reference()
    ref = Dict{String,Any}()
    for file in ("r_reference.csv", "r_reference_crossed.csv")
        raw, hdr = readdlm(joinpath(NGI_DIR, file), ','; header = true)
        ngi_read_reference!(ref, raw)
    end
    return ref
end

function ngi_read_reference!(ref, raw)
    for r in eachrow(raw)
        id = String(r[1]); term = String(r[2]); est = r[3]
        e = get!(ref, id) do
            (loglik = Float64(r[5]), clean = string(r[7]) == "TRUE" && string(r[8]) == "TRUE" && r[6] == 0,
             mu = Float64[], sigma = Float64[], sd = Float64[], sdreg = Dict{String,Vector{Float64}}())
        end
        startswith(term, "mu:") && push!(e.mu, Float64(est))
        startswith(term, "sigma:") && push!(e.sigma, Float64(est))
        startswith(term, "sd:mu:") && push!(e.sd, Float64(est))
        m = match(r"^sd\(g, coef = \"([^\"]+)\"\):(.+)$", term)
        m === nothing || push!(get!(e.sdreg, String(m.captures[1]), Float64[]), Float64(est))
        m = match(r"^sd\((g2?)\):(.+)$", term)                       # crossed intercepts
        m === nothing || push!(get!(e.sdreg, String(m.captures[1]), Float64[]), Float64(est))
    end
    return ref
end

# the reference SD parameters in term order: the SD-formula coefficients of a term that has one, the
# log of the constant SD otherwise (the layout of the Julia `:sd` / `:resd` coefficients)
function ngi_ref_sdpar(id, r)
    fam, kind, q, pat = ngi_parts(id)
    members = ngi_sd_members(id)
    out = Float64[]; ic = 0
    for j in 1:(kind == "slope" ? 1 : kind == "crossed" ? 2 : q)
        if j in members
            append!(out, r.sdreg[kind == "crossed" ? ["g", "g2"][j] : NGI_COEFS[kind == "slope" ? 2 : j]])
        else
            ic += 1
            push!(out, log(r.sd[ic]))
        end
    end
    return out
end

# largest gaps against one reference entry; `sdpar` is the max gap over the SD parameters
# (log SDs, or SD-formula coefficients) and `boundary` flags a term whose SD is at zero in either fit
function ngi_compare(id, fit, r)
    dll = abs(loglik(fit) - r.loglik)
    fe = maximum(abs.(coef(fit, :mu) .- r.mu))
    isempty(r.sigma) || (fe = max(fe, maximum(abs.(coef(fit, :sigma) .- r.sigma))))
    sdblocks = [first(p) for p in fit.blocks if first(p) in (:sd, :resd) || startswith(String(first(p)), "sdtarget_")]
    jsd = reduce(vcat, [coef(fit, b) for b in sdblocks])      # the crossed-intercept `sd(g)` route names one block per group
    rsd = ngi_ref_sdpar(id, r)
    sdpar = maximum(abs.(jsd .- rsd))
    boundary = any(<(-6), jsd) || any(<(-6), rsd)
    return (dll = dll, fe = fe, sdlog = sdpar, boundary = boundary)
end
ngi_agrees(c) = c.dll <= 1e-4 && c.fe <= 1e-3 && (c.sdlog <= 5e-3 || c.boundary)
