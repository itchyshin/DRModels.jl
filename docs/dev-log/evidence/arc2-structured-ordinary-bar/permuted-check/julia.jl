# Julia side of native.R (run from this directory after it).
using DRModels
println("pathof: ", pathof(DRModels))
function read_data(path)
    lines = filter(!isempty, readlines(path)); header = split(lines[1], '\t')
    rows = [split(l, '\t') for l in lines[2:end]]
    cols = Dict{String,Any}()
    for (j, h) in enumerate(header)
        raw = [String(r[j]) for r in rows]; p = tryparse.(Float64, raw)
        cols[String(h)] = any(isnothing, p) ? raw : Float64.(p)
    end
    cols
end
nat = Dict((split(l,'\t')[1], split(l,'\t')[2]) => parse(Float64, split(l,'\t')[3]) for l in readlines("native.tsv"))
d = read_data("phylo.data.tsv"); tree = strip(read("phylo.tree.nwk", String))
nt = (; y = d["y"], x = d["x"], sp = d["sp"], h = d["h"])
f = bf(@formula(y ~ x + phylo(1 | sp) + (1 | h)), @formula(sigma ~ 1))
fit = drm(f, Gaussian(); data = nt, tree = tree)
println("phylo drm:    df ", dof(fit), " logLik ", loglik(fit), " diff ", loglik(fit) - nat[("phylo","logLik")], " re_sd ", re_sd(fit), " phylo_scale ", fit.phylo_scale)
res = drm_bridge(; formula = "y ~ x + phylo(1 | sp) + (1 | h); sigma ~ 1", family = "gaussian", data = d, tree = tree)
println("phylo bridge: df ", res["df"], " logLik ", res["loglik"], " diff ", res["loglik"] - nat[("phylo","logLik")])
d2 = read_data("relmat.data.tsv")
K = reduce(vcat, [permutedims(parse.(Float64, split(l, '\t'))) for l in filter(!isempty, readlines("relmat.K_firstseen.tsv"))])
nt2 = (; y = d2["y"], x = d2["x"], id = d2["id"], h = d2["h"])
fit2 = drm(bf(@formula(y ~ x + relmat(1 | id) + (1 | h)), @formula(sigma ~ 1)), Gaussian(); data = nt2, K = K)
println("relmat drm:   df ", dof(fit2), " logLik ", loglik(fit2), " diff ", loglik(fit2) - nat[("relmat","logLik")])
res2 = drm_bridge(; formula = "y ~ x + relmat(1 | id) + (1 | h); sigma ~ 1", family = "gaussian", data = d2, K = K)
println("relmat bridge: df ", res2["df"], " logLik ", res2["loglik"], " diff ", res2["loglik"] - nat[("relmat","logLik")])
