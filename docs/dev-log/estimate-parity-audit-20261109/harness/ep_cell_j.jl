using DRModels, DelimitedFiles, Logging, Printf, Statistics, Random, LinearAlgebra
const DD = get(ENV, "FUZZ_DATA", "data")

const cells_file = "cells_all.tsv"; rows = readdlm(cells_file, '\t', Any; header = false)
const CELLS = Dict{String,NamedTuple}()
for r in 2:size(rows, 1)
    CELLS[String(rows[r, 1])] = (; family = String(rows[r, 2]), kind = String(rows[r, 3]), lab = Int(rows[r, 4]),
        q = Int(rows[r, 5]), pat = String(rows[r, 6]), cm = Int(rows[r, 7]), est = String(rows[r, 8]))
end

const NWK = strip(read(joinpath(DD, "tree.nwk"), String))
const COORDS = let m = readdlm(joinpath(DD, "coords.csv"), ',', Any; skipstart = 1)
    Dict(String(m[i, 1]) => [Float64(m[i, 2]), Float64(m[i, 3])] for i in 1:size(m, 1))
end
const IDS = String.(vec(readdlm(joinpath(DD, "A_ids.csv"), ',', Any; skipstart = 1)))
const AMAT = Matrix{Float64}(readdlm(joinpath(DD, "A.csv"), ',', Float64))

fam_of(f) = f == "gaussian" ? Gaussian() : f == "poisson" ? Poisson() : f == "nb2" ? NegBinomial2() :
    f == "gamma" ? DRModels.Gamma() : f == "binomial" ? Binomial(:logit) : f == "beta" ? Beta() :
    f == "lognormal" ? LogNormal() : f == "student" ? Student() : f == "tweedie" ? Tweedie() : error("family $f")
has_sigma(f) = !(f in ("poisson", "binomial"))

const _DATA = Dict{String,Any}()
function load_data(fam, kind)
    key = "$(fam)_$(kind)"
    get!(_DATA, key) do
        m, h = readdlm(joinpath(DD, "data_$(key).csv"), ',', Any; header = true)
        h = String.(vec(h))
        col(n) = m[:, findfirst(==(n), h)]
        (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), x3 = Float64.(col("x3")),
           z = Float64.(col("z")), z2 = Float64.(col("z2")), g = String.(col("g")), g2 = String.(col("g2")))
    end
end

const CNAME = ["intercept", "x1", "x2", "x3"]
const LEVEL = Dict("animal" => "animal", "relmat" => "relmat", "phylo" => "phylogenetic", "spatial" => "spatial")
mkf(s) = eval(Meta.parse("@formula(" * s * ")"))

function build(cl)
    q, kind = cl.q, cl.kind
    bar = join(["1", "x1", "x2", "x3"][1:q], " + ")
    lab = (cl.lab == 1 || cl.cm == 1) ? " | p | " : " | "
    inner = bar * lab * "g"
    re = kind == "slope" ? "(0 + x1 | g)" : kind == "indep" ? "(1 | g)" * join([" + (0 + $v | g)" for v in ["x1", "x2", "x3"][1:q-1]]) :
         kind == "ordinary" ? "($inner)" : kind == "crossed" ? "($inner) + (1 | g2)" :
         kind == "animal" ? "animal($inner)" : kind == "relmat" ? "relmat($inner)" :
         kind == "phylo" ? "phylo($inner)" : "spatial($inner)"
    mu = mkf("y ~ x1 + x2 + x3 + $re")
    sel = cl.pat == "none" ? Int[] : cl.pat in ("one", "z") ? collect(1:q) : cl.pat == "first" ? [1] : [q]
    rhs = cl.pat == "one" ? "1" : "z"
    lv = haskey(LEVEL, kind) ? ", " * LEVEL[kind] : ""
    sds = Any[]
    for j in sel
        cf = kind == "slope" ? ", coef(x1)" : q == 1 ? "" : ", coef($(CNAME[j]))"
        push!(sds, mkf("sd(g$lv$cf) ~ $rhs"))
    end
    if kind == "crossed" && !(cl.pat in ("none", "last"))
        push!(sds, mkf("sd(g2) ~ " * (cl.pat == "one" ? "1" : "z2")))
    end
    parts = Any[mu]
    has_sigma(cl.family) && push!(parts, cl.cm == 1 ? mkf("sigma ~ 1 + (1 | p | g)") : mkf("sigma ~ 1"))
    return bf(parts..., sds...)
end

# Optional follow-up mode: FUZZ_MARGINAL=Laplace re-runs cells with drmTMB's Laplace objective (documented opt-in).
const EXTRA_KW = haskey(ENV, "FUZZ_MARGINAL") ? (; marginal = Symbol(ENV["FUZZ_MARGINAL"])) : (;)
# FUZZ_COORDS=matrix passes `coords` as a G x 2 Matrix (rows in A_ids order) instead of a Dict: some routes
# accept only one of the two formats.
const COORDS_MAT = reduce(vcat, [permutedims(COORDS[i]) for i in IDS])
function kwargs_for(kind)
    kind == "phylo" ? (; tree = NWK) : kind == "spatial" ? (; coords = get(ENV, "FUZZ_COORDS", "dict") == "matrix" ? COORDS_MAT : COORDS) :
    kind == "animal" ? (; A = AMAT, ids = IDS) : kind == "relmat" ? (; K = AMAT) : (;)
end

