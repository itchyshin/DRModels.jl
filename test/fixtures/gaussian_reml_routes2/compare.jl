# Agreement of DRModels `method = :REML` with R drmTMB 0.8.1 `REML = TRUE` on three more Gaussian designs:
#   B  one correlated block of 3 or 4 coefficients            (1 + x1 + x2 [+ x3] | g)         (also labelled `| p |`)
#   F  a correlated block of 2-4 coefficients beside a crossed intercept  (1 + x1 [+ ..] | g) + (1 | g2)
#   S  relmat / animal / spatial intercept with its own sd() formula     sd(g, <level>) ~ 1  and  ~ z
# three simulated datasets each (n = 200, 40 groups). Data and R reference come from `gen_ref.R` in this folder.
# Thresholds were fixed in docs/dev-log/2026-11-09-gaussian-reml-routes2.md BEFORE any reference was computed.
using DRModels, DelimitedFiles, Statistics

const _FXD2 = @__DIR__
const REML2_THRESH = (loglik = 1e-4, mu_se_units = 0.02, sd_rel = 1e-2, cor_abs = 1e-2, boundary_rel = 1e-4)

_rd2(f) = readdlm(joinpath(_FXD2, f), ',', Any; header = true)
function _cols2(f)
    (m, h) = _rd2(f); h = String.(vec(h))
    Dict(c => m[:, i] for (i, c) in enumerate(h))
end

const _REF2 = let (m, h) = _rd2("r_reml_reference.csv")
    h = String.(vec(h)); ix(c) = findfirst(==(c), h)
    [(shape = String(m[i, ix("shape")]), seed = Int(m[i, ix("seed")]), term = String(m[i, ix("term")]),
      est = m[i, ix("estimate")] isa Number ? Float64(m[i, ix("estimate")]) : NaN,
      se = m[i, ix("se")] isa Number ? Float64(m[i, ix("se")]) : NaN, ll = Float64(m[i, ix("loglik")]),
      conv = Int(m[i, ix("conv")]) == 0 && string(m[i, ix("pdhess")]) in ("TRUE", "true")) for i in 1:size(m, 1)]
end

# shape => (data prefix, response column, mu formula, provider, extra sd formula or "", R term names of the random-effect SDs,
#           number of correlation parameters)
const REML2_SHAPES = Dict(
    "ord3"    => ("ord", "y3", "y ~ x1 + x2 + x3 + (1 + x1 + x2 | g)", :none, "",
                  ["sd:mu:(1 + x1 + x2 | g):(Intercept)", "sd:mu:(1 + x1 + x2 | g):x1", "sd:mu:(1 + x1 + x2 | g):x2"], 3),
    "ord4"    => ("ord", "y4", "y ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g)", :none, "",
                  ["sd:mu:(1 + x1 + x2 + x3 | g):(Intercept)", "sd:mu:(1 + x1 + x2 + x3 | g):x1",
                   "sd:mu:(1 + x1 + x2 + x3 | g):x2", "sd:mu:(1 + x1 + x2 + x3 | g):x3"], 6),
    "ord3lab" => ("ord", "y3", "y ~ x1 + x2 + x3 + (1 + x1 + x2 | p | g)", :none, "",
                  ["sd:mu:(1 + x1 + x2 | p | g):(Intercept)", "sd:mu:(1 + x1 + x2 | p | g):x1",
                   "sd:mu:(1 + x1 + x2 | p | g):x2"], 3),
    "cross2"  => ("crossed", "y2", "y ~ x1 + x2 + x3 + (1 + x1 | g) + (1 | g2)", :none, "",
                  ["sd:mu:(1 + x1 | g):(Intercept)", "sd:mu:(1 + x1 | g):x1", "sd:mu:(1 | g2)"], 1),
    "cross3"  => ("crossed", "y3", "y ~ x1 + x2 + x3 + (1 + x1 + x2 | g) + (1 | g2)", :none, "",
                  ["sd:mu:(1 + x1 + x2 | g):(Intercept)", "sd:mu:(1 + x1 + x2 | g):x1", "sd:mu:(1 + x1 + x2 | g):x2",
                   "sd:mu:(1 | g2)"], 3),
    "cross4"  => ("crossed", "y4", "y ~ x1 + x2 + x3 + (1 + x1 + x2 + x3 | g) + (1 | g2)", :none, "",
                  ["sd:mu:(1 + x1 + x2 + x3 | g):(Intercept)", "sd:mu:(1 + x1 + x2 + x3 | g):x1",
                   "sd:mu:(1 + x1 + x2 + x3 | g):x2", "sd:mu:(1 + x1 + x2 + x3 | g):x3", "sd:mu:(1 | g2)"], 6),
)
for kind in (:animal, :relmat, :spatial)
    term = kind === :animal ? "animal(1 | g)" : kind === :relmat ? "relmat(1 | g)" : "spatial(1 | g)"
    k = String(kind)
    REML2_SHAPES["$(k)_sd1"] = (k, "y_sd1", "y ~ x1 + x2 + x3 + $term", kind, "sd(g, $k) ~ 1",
                                ["sd_$(k)(g):(Intercept)"], 0)
    REML2_SHAPES["$(k)_sdz"] = (k, "y_sdz", "y ~ x1 + x2 + x3 + $term", kind, "sd(g, $k) ~ z",
                                ["sd_$(k)(g):(Intercept)", "sd_$(k)(g):z"], 0)
end
const REML2_GROUPS = ["B block q3-q4" => ["ord3", "ord4", "ord3lab"],
                      "F block beside crossed intercept" => ["cross2", "cross3", "cross4"],
                      "S structured with sd()" => ["animal_sd1", "animal_sdz", "relmat_sd1", "relmat_sdz",
                                                   "spatial_sd1", "spatial_sdz"]]

function reml2_fit(shape::String, seed::Int)
    prefix, ycol, fstr, kwsym, sdstr, _, _ = REML2_SHAPES[shape]
    d = _cols2("$(prefix)_s$(seed).csv")
    data = (; y = Float64.(d[ycol]), x1 = Float64.(d["x1"]), x2 = Float64.(d["x2"]), x3 = Float64.(d["x3"]),
            z = Float64.(d["z"]), g = String.(d["g"]), g2 = String.(d["g2"]))
    mu = eval(Meta.parse("DRModels.@formula($fstr)"))
    parts = Any[mu, DRModels.@formula(sigma ~ 1)]
    isempty(sdstr) || push!(parts, eval(Meta.parse("DRModels.@formula($sdstr)")))
    ids = String.(vec(readdlm(joinpath(_FXD2, "A_ids.csv"), ',', Any; skipstart = 1)))
    Amat = Matrix{Float64}(readdlm(joinpath(_FXD2, "A.csv"), ',', Float64))
    kw = if kwsym === :animal
        (; A = Amat, ids = ids)
    elseif kwsym === :relmat
        (; K = Amat)
    elseif kwsym === :spatial
        m = readdlm(joinpath(_FXD2, "coords.csv"), ',', Any; skipstart = 1)
        (; coords = Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1)))
    else
        (;)
    end
    return drm(bf(parts...), Gaussian(); data = data, method = :REML, kw...), data
end

# One comparison row: ll_diff, max |Δmu|/SE_R, max relative SD difference (block shapes: SDs on the natural scale;
# structured shapes: the sd() coefficients on the log scale, where |Δlog| is the relative SD difference), max
# |Δcorrelation|, boundary flag, converged flags.
const REML2_REF_ALL = _REF2
function reml2_compare(shape::String, seed::Int)
    fit, data = reml2_fit(shape, seed)
    prefix, _, _, kwsym, sdstr, rterms, ncor = REML2_SHAPES[shape]
    ref = filter(r -> r.shape == shape && r.seed == seed, REML2_REF_ALL)
    R(term) = only(filter(r -> r.term == term, ref))
    th = fit.theta; blk = fit.blocks
    mu = th[blk[1][2]]
    mu_units = [abs(mu[j] - R(t).est) / R(t).se for (j, t) in enumerate(["mu:(Intercept)", "mu:x1", "mu:x2", "mu:x3"])]
    sig_j = exp(th[blk[2][2]][1])
    sds_j = Float64[sig_j]; sds_r = Float64[R("sigma").est]; cor_diff = 0.0
    if kwsym === :none
        # blocks: mu, sigma, one SD block per block coefficient, :recov, then each crossed intercept's SD block
        irec = findfirst(b -> first(b) === :recov, blk)
        q = irec - 3
        append!(sds_j, [exp(th[blk[2+k][2]][1]) for k in 1:q])
        append!(sds_j, [exp(th[blk[e][2]][1]) for e in (irec + 1):length(blk)])
        append!(sds_r, [R(t).est for t in rterms])
        Rm = DRModels._q34_corr_from_theta(th[blk[irec][2]], q)
        members = ["(Intercept)", "x1", "x2", "x3"][1:q]
        # R names each correlation `cor:<scope>:cor(a,b | g)`; match by the two coefficient names
        rc = [(m.captures[1], m.captures[2], x.est) for x in ref
              for m in (match(r"^cor:[^:]+:cor\((.+?),(.+?) \|", x.term),) if m !== nothing]
        length(rc) == ncor == q * (q - 1) ÷ 2 || error("correlation count mismatch for $shape")
        cor_diff = maximum(abs(Rm[findfirst(==(a), members), findfirst(==(b), members)] - est) for (a, b, est) in rc)
    else
        sdc = th[blk[3][2]]
        append!(sds_j, exp.(sdc)); append!(sds_r, exp.([R(t).est for t in rterms]))
    end
    boundary = any(x -> x < REML2_THRESH.boundary_rel * std(data.y), vcat(sds_j, sds_r))
    return (shape = shape, seed = seed, ll_diff = abs(fit.loglik - first(ref).ll), mu_se_units = maximum(mu_units),
            sd_rel = maximum(abs.(sds_j .- sds_r) ./ sds_r), cor_abs = cor_diff, boundary = boundary,
            converged = fit.converged, r_converged = first(ref).conv, estim = fit.estim_method)
end

# Evidence table: `julia --project test/fixtures/gaussian_reml_routes2/compare.jl [out.tsv]`
if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] : "gaussian_reml_routes2_agreement.tsv"
    open(out, "w") do io
        println(io, "group\tshape\tseed\tll_diff\tmu_se_units\tsd_rel\tcor_abs\tboundary\tjulia_converged\tr_converged")
        for (grp, shapes) in REML2_GROUPS, shape in shapes, seed in (21, 22, 23)
            r = reml2_compare(shape, seed)
            println(io, join([grp, shape, seed, r.ll_diff, r.mu_se_units, r.sd_rel, r.cor_abs, r.boundary, r.converged, r.r_converged], "\t"))
        end
    end
end
