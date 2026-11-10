# Agreement of DRModels `method = :REML` with R drmTMB 0.8.1 `REML = TRUE` on five Gaussian mean-side
# random-effect designs (slope only, several independent scalar terms, crossed intercepts, relmat/animal
# intercept, phylo intercept + slope), three simulated datasets each. Data and the R reference come from
# `gen_ref.R` in this folder (n = 200 per dataset, 40 groups; fixed 400-row refusal-fuzz covariates halved).
# Thresholds were fixed in docs/dev-log/2026-11-09-gaussian-reml-routes.md BEFORE any reference was computed.
using DRModels, DelimitedFiles, Statistics

const _FXD = @__DIR__
const REML_ROUTE_THRESH = (loglik = 1e-4, mu_se_units = 0.02, sd_rel = 1e-2, boundary_rel = 1e-4)

_rd(f) = readdlm(joinpath(_FXD, f), ',', Any; header = true)
function _cols(f)
    (m, h) = _rd(f); h = String.(vec(h))
    Dict(c => m[:, i] for (i, c) in enumerate(h))
end

const _REF = let (m, h) = _rd("r_reml_reference.csv")
    h = String.(vec(h)); ix(c) = findfirst(==(c), h)
    [(shape = String(m[i, ix("shape")]), seed = Int(m[i, ix("seed")]), term = String(m[i, ix("term")]),
      est = Float64(m[i, ix("estimate")]), se = Float64(m[i, ix("se")]), ll = Float64(m[i, ix("loglik")]),
      conv = Int(m[i, ix("conv")]) == 0 && string(m[i, ix("pdhess")]) in ("TRUE", "true")) for i in 1:size(m, 1)]
end

# shape => (data file prefix, response column, formula string, extra-kwargs symbol, R sd terms in theta[:resd] order)
const REML_ROUTE_SHAPES = Dict(
    "slope"  => ("ord",     "y_slope", "y ~ x1 + x2 + x3 + (0 + x1 | g)", :none,
                 ["sd:mu:(0 + x1 | g)"]),
    "indep2" => ("ord",     "y_i2", "y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)", :none,
                 ["sd:mu:(1 | g)", "sd:mu:(0 + x1 | g)"]),
    "indep3" => ("ord",     "y_i3", "y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)", :none,
                 ["sd:mu:(1 | g)", "sd:mu:(0 + x1 | g)", "sd:mu:(0 + x2 | g)"]),
    "indep4" => ("ord",     "y_i4", "y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g) + (0 + x3 | g)", :none,
                 ["sd:mu:(1 | g)", "sd:mu:(0 + x1 | g)", "sd:mu:(0 + x2 | g)", "sd:mu:(0 + x3 | g)"]),
    "cross1" => ("crossed", "y_cross", "y ~ x1 + x2 + x3 + (1 | g) + (1 | g2)", :none,
                 ["sd:mu:(1 | g)", "sd:mu:(1 | g2)"]),
    "animal" => ("animal",  "y", "y ~ x1 + x2 + x3 + animal(1 | g)", :animal, ["sd:mu:animal(1 | g)"]),
    "relmat" => ("relmat",  "y", "y ~ x1 + x2 + x3 + relmat(1 | g)", :relmat, ["sd:mu:relmat(1 | g)"]),
    "phylo2" => ("phylo",   "y", "y ~ x1 + x2 + x3 + phylo(1 + x1 | g)", :phylo,
                 ["sd:mu:phylo(1 | g)", "sd:mu:phylo(0 + x1 | g)"]),
)
const REML_ROUTE_GROUPS = ["A slope" => ["slope"], "B multi" => ["indep2", "indep3", "indep4", "cross1"],
                           "C dense structured" => ["animal", "relmat"], "D phylo slope" => ["phylo2"]]

function reml_route_fit(shape::String, seed::Int)
    prefix, ycol, fstr, kwsym, _ = REML_ROUTE_SHAPES[shape]
    d = _cols("$(prefix)_s$(seed).csv")
    data = (; y = Float64.(d[ycol]), x1 = Float64.(d["x1"]), x2 = Float64.(d["x2"]), x3 = Float64.(d["x3"]),
            g = String.(d["g"]), g2 = String.(d["g2"]))
    f = eval(Meta.parse("DRModels.@formula($fstr)"))
    kw = if kwsym === :animal
        ids = String.(vec(readdlm(joinpath(_FXD, "A_ids.csv"), ',', Any; skipstart = 1)))
        (; A = Matrix{Float64}(readdlm(joinpath(_FXD, "A.csv"), ',', Float64)), ids = ids)
    elseif kwsym === :relmat
        (; K = Matrix{Float64}(readdlm(joinpath(_FXD, "A.csv"), ',', Float64)))
    elseif kwsym === :phylo
        (; tree = strip(read(joinpath(_FXD, "tree.nwk"), String)))
    else
        (;)
    end
    return drm(bf(f, DRModels.@formula(sigma ~ 1)), Gaussian(); data = data, method = :REML, kw...), data
end

# One comparison row: (shape, seed, ll_diff, max |Δmu|/SE_R, max relative SD diff, boundary, converged, R converged)
function reml_route_compare(shape::String, seed::Int)
    fit, data = reml_route_fit(shape, seed)
    ref = filter(r -> r.shape == shape && r.seed == seed, _REF)
    R(term) = only(filter(r -> r.term == term, ref))
    th = fit.theta
    mu = th[DRModels._block_range(fit, :mu)]
    mu_units = [abs(mu[j] - R(t).est) / R(t).se for (j, t) in enumerate(["mu:(Intercept)", "mu:x1", "mu:x2", "mu:x3"])]
    sds_j = vcat(exp.(th[DRModels._block_range(fit, :sigma)]), exp.(th[DRModels._block_range(fit, :resd)]))
    sds_r = [R("sigma").est; [R(t).est for t in REML_ROUTE_SHAPES[shape][5]]]
    boundary = any(x -> x < REML_ROUTE_THRESH.boundary_rel * std(data.y), vcat(sds_j, sds_r))
    sd_rel = maximum(abs.(sds_j .- sds_r) ./ sds_r)
    return (shape = shape, seed = seed, ll_diff = abs(fit.loglik - first(ref).ll), mu_se_units = maximum(mu_units),
            sd_rel = sd_rel, boundary = boundary, converged = fit.converged, r_converged = first(ref).conv,
            estim = fit.estim_method)
end

# Evidence table: `julia --project test/fixtures/gaussian_reml_routes/compare.jl [out.tsv]`
if abspath(PROGRAM_FILE) == @__FILE__
    out = length(ARGS) >= 1 ? ARGS[1] : "gaussian_reml_routes_agreement.tsv"
    open(out, "w") do io
        println(io, "group\tshape\tseed\tll_diff\tmu_se_units\tsd_rel\tboundary\tjulia_converged\tr_converged")
        for (grp, shapes) in REML_ROUTE_GROUPS, shape in shapes, seed in (11, 12, 13)
            r = reml_route_compare(shape, seed)
            println(io, join([grp, shape, seed, r.ll_diff, r.mu_se_units, r.sd_rel, r.boundary, r.converged, r.r_converged], "\t"))
        end
    end
end
