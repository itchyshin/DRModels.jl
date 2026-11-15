# Fits shared by test/test_fitted_phylo_modes.jl and the pre-change capture script (PR #1324).
# The two phylo groups of the `fitted()` change:
#   group 1: a Gaussian mean with ONE structured intercept, relmat(1 | g), animal(1 | g), dense phylo(1 | g)
#   group 2: phylo(1 | g) Laplace fits for Poisson, NB2, Binomial, BetaBinomial, Gamma and Beta
# Data and tree are the p17_fitted fixtures; K and A are written by p17_fitted_phylo_r.R (rows in the first-seen
# order of g, g01 ... g16). Ids match the cell names of that R script. The `*M` cells put a missing
# response on the rows where `y_gm` is NA (16 of 160 rows).
using DRModels
using DelimitedFiles

const _PM_DIR = @__DIR__
const _PM_FACS = Set(["g", "g2", "h", "f", "f2", "gna", "gu", "y_cat", "y_ordf"])

function _pm_data()
    m, h = readdlm(joinpath(_PM_DIR, "data.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        v = if n in _PM_FACS
            any(x -> x == "NA", c) ? Union{Missing,String}[x == "NA" ? missing : String(x) for x in c] : String.(c)
        elseif any(x -> x == "NA", c)
            Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c]
        elseif all(x -> x isa Integer, c)
            Int.(c)
        else
            Float64.(c)
        end
        push!(cols, Symbol(n) => v)
    end
    return (; cols...)
end

function _pm_mask(d, name)
    na = ismissing.(d.y_gm)
    y = getproperty(d, name)
    return Union{Missing,Float64}[na[i] ? missing : Float64(y[i]) for i in eachindex(y)]
end

# the data with a missing response (the `_m` columns)
_pm_mr(d) = merge(d, (y_g_m = _pm_mask(d, :y_g), y_pois_m = _pm_mask(d, :y_pois), y_nb_m = _pm_mask(d, :y_nb),
                      y_gam_m = _pm_mask(d, :y_gam), y_beta_m = _pm_mask(d, :y_beta),
                      s_m = _pm_mask(d, :s), fl_m = _pm_mask(d, :fl)))

_pm_K() = Matrix{Float64}(readdlm(joinpath(_PM_DIR, "k_relmat.csv"), ','))
_pm_A() = Matrix{Float64}(readdlm(joinpath(_PM_DIR, "a_animal.csv"), ','))
# The engine pairs row i of K / A with the i-th level of g in order of first appearance in the data, so
# the matrix (rows in the label order g01 ... g16) is put in that order for the data at hand. Dropping rows
# with a missing response can change the first-appearance order; the labels always stay matched.
_pm_order(d) = [parse(Int, l[2:end]) for l in unique(d.g)]
_pm_Kfor(d) = (o = _pm_order(d); _pm_K()[o, o])
_pm_Afor(d) = (o = _pm_order(d); _pm_A()[o, o])
const _PM_TREE = strip(read(joinpath(_PM_DIR, "tree16.newick"), String))
_pm_tree() = DRModels.augmented_phy(_PM_TREE)

# (id, fit function of the data and extra keywords)
const _PM_G1 = [
    ("G1|relmat", (d; kw...) -> drm(bf(@formula(y_g ~ x1 + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, K = _pm_Kfor(d), kw...)),
    ("G1|animal", (d; kw...) -> drm(bf(@formula(y_g ~ x1 + animal(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, A = _pm_Afor(d), kw...)),
    ("G1|phylo-dense", (d; kw...) -> drm(bf(@formula(y_g ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, tree = _pm_tree(), algorithm = :gls, kw...)),
    ("G1M|relmat", (d; kw...) -> drm(bf(@formula(y_g_m ~ x1 + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, K = _pm_Kfor(d), kw...)),
    ("G1M|animal", (d; kw...) -> drm(bf(@formula(y_g_m ~ x1 + animal(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, A = _pm_Afor(d), kw...)),
    ("G1M|phylo-dense", (d; kw...) -> drm(bf(@formula(y_g_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, tree = _pm_tree(), algorithm = :gls, kw...)),
]
const _PM_G2 = [
    ("G2|poisson", (d; kw...) -> drm(bf(@formula(y_pois ~ x1 + phylo(1 | g))), Poisson(); data = d, tree = _pm_tree(), kw...)),
    ("G2|nb2", (d; kw...) -> drm(bf(@formula(y_nb ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d, tree = _pm_tree(), kw...)),
    ("G2|binomial", (d; kw...) -> drm(bf(@formula(cbind(s, fl) ~ x1 + phylo(1 | g))), Binomial(); data = d, tree = _pm_tree(), kw...)),
    ("G2|betabinomial", (d; kw...) -> drm(bf(@formula(cbind(s, fl) ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), BetaBinomial(); data = d, tree = _pm_tree(), kw...)),
    ("G2|gamma", (d; kw...) -> drm(bf(@formula(y_gam ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d, tree = _pm_tree(), kw...)),
    ("G2|beta", (d; kw...) -> drm(bf(@formula(y_beta ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), DRModels.Beta(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|poisson", (d; kw...) -> drm(bf(@formula(y_pois_m ~ x1 + phylo(1 | g))), Poisson(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|nb2", (d; kw...) -> drm(bf(@formula(y_nb_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), NegBinomial2(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|binomial", (d; kw...) -> drm(bf(@formula(cbind(s_m, fl_m) ~ x1 + phylo(1 | g))), Binomial(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|betabinomial", (d; kw...) -> drm(bf(@formula(cbind(s_m, fl_m) ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), BetaBinomial(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|gamma", (d; kw...) -> drm(bf(@formula(y_gam_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), DRModels.Gamma(); data = d, tree = _pm_tree(), kw...)),
    ("G2M|beta", (d; kw...) -> drm(bf(@formula(y_beta_m ~ x1 + phylo(1 | g)), @formula(sigma ~ 1)), DRModels.Beta(); data = d, tree = _pm_tree(), kw...)),
]
# Variants reached through the same fitters (a covariate on sigma, REML, a Gaussian sigma ~ x). They have no
# drmTMB value pinned here; the test checks them against an independent recomputation.
const _PM_EXTRA = [
    ("X|relmat|REML", (d; kw...) -> drm(bf(@formula(y_g ~ x1 + relmat(1 | g)), @formula(sigma ~ 1)), Gaussian(); data = d, K = _pm_Kfor(d), method = :REML, kw...)),
    ("X|relmat|sigma=x1", (d; kw...) -> drm(bf(@formula(y_g ~ x1 + relmat(1 | g)), @formula(sigma ~ x1)), Gaussian(); data = d, K = _pm_Kfor(d), kw...)),
    ("X|poisson|REML", (d; kw...) -> drm(bf(@formula(y_pois ~ x1 + phylo(1 | g))), Poisson(); data = d, tree = _pm_tree(), method = :REML, kw...)),
    ("X|nb2|sigma=x1", (d; kw...) -> drm(bf(@formula(y_nb ~ x1 + phylo(1 | g)), @formula(sigma ~ x1)), NegBinomial2(); data = d, tree = _pm_tree(), kw...)),
    ("X|gamma|sigma=x1", (d; kw...) -> drm(bf(@formula(y_gam ~ x1 + phylo(1 | g)), @formula(sigma ~ x1)), DRModels.Gamma(); data = d, tree = _pm_tree(), kw...)),
    ("X|beta|sigma=x1", (d; kw...) -> drm(bf(@formula(y_beta ~ x1 + phylo(1 | g)), @formula(sigma ~ x1)), DRModels.Beta(); data = d, tree = _pm_tree(), kw...)),
]
const _PM_CELLS = vcat(_PM_G1, _PM_G2, _PM_EXTRA)
