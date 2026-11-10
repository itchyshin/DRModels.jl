# Julia side of the agreement protocol (docs/dev-log/2026-11-08-tweedie-structured-sd.md).
# julia --project=<repo> agreement_julia.jl <out.tsv>
using DRModels, DelimitedFiles, LinearAlgebra

const DIR = @__DIR__
function load(ds)
    m, h = readdlm(joinpath(DIR, ds * ".csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    (; y = Float64.(col("y")), x = Float64.(col("x")), z = Float64.(col("z")), g = String.(col("g")))
end
open(ARGS[1], "w") do io
    println(io, join(["engine", "dataset", "model", "idx", "name", "est", "se", "conv", "pdhess", "secs"], "\t"))
    for ds in ["phylo_1", "phylo_2", "phylo_3", "relmat_1", "relmat_2"], model in ["sd-none", "sd-z"]
        d = load(ds)
        phylo = startswith(ds, "phylo")
        re = phylo ? "phylo(1 | g)" : "relmat(1 | g)"
        lvl = phylo ? "phylogenetic" : "relmat"
        mu = eval(Meta.parse("@formula(y ~ x + $re)"))
        f = model == "sd-z" ? bf(mu, eval(Meta.parse("@formula(sd(g, $lvl) ~ z)"))) : bf(mu)
        kw = phylo ? (; tree = strip(read(joinpath(DIR, ds * "_tree.nwk"), String))) :
                     (; K = Matrix{Float64}(readdlm(joinpath(DIR, ds * "_K.csv"), ',', Float64; skipstart = 1)))
        secs = @elapsed fit = drm(f, Tweedie(); data = d, kw...)
        est = coef(fit); se = sqrt.(max.(diag(vcov(fit)), 0.0))
        names = vcat([[string(b, ":", n) for n in DRModels._block_coefnames(fit, b)] for (b, _) in fit.blocks]...)
        for i in eachindex(est)
            println(io, join(["Julia", ds, model, i, names[i], est[i], se[i], fit.converged,
                              all(isfinite, se), round(secs, digits = 2)], "\t"))
        end
    end
end
