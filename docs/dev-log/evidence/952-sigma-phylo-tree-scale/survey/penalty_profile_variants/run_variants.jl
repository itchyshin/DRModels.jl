# Sibling survey (#952): Gaussian sigma-phylo route under penalty (MAP) and profile_ci.
# Usage: julia --project=/opt/base|/opt/fix run_variants.jl <label>
using DRModels, DelimitedFiles, Printf

const LABEL = isempty(ARGS) ? "?" : ARGS[1]
const D = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5/"
raw = readdlm(D * "carni70.csv", ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')),
         lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
trees = Dict("raw" => read(D * "carni70.nwk", String),
             "unit" => read(D * "carni70_unit.nwk", String))
h = Dict(k => phylo_tree_height(augmented_phy(v)) for (k, v) in trees)
@printf("# %s  height raw=%.15g unit=%.15g ratio=%.15g\n", LABEL, h["raw"], h["unit"], h["raw"] / h["unit"])
flush(stdout)

const C = 53.8                 # the stated tree height used for the sd_u equivalence
const SHIFT = 0.5 * log(C)

f = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)),
       @formula(sigma ~ 1 + phylo(1 | species)))

# raw-tree sd_u = 1/sqrt(53.8) is the same model as unit-tree sd_u = 1.0
sdu(tree) = tree == "raw" ? 1 / sqrt(C) : 1.0

variants = [
    ("sep_pen",        tree -> (; penalty = drm_phylo_penalty(sd_u = sdu(tree), sd_alpha = 0.05))),
    ("coup_pen",       tree -> (; phylo_coupled = true,
                                  penalty = drm_phylo_penalty(sd_u = sdu(tree), sd_alpha = 0.05))),
    ("coup_pen_cor1",  tree -> (; phylo_coupled = true,
                                  penalty = drm_phylo_penalty(sd_u = sdu(tree), sd_alpha = 0.05, cor_sd = 1.0))),
    ("sep_profile",    tree -> (; profile_ci = true)),
]

fmt(x) = x isa AbstractVector ? "[" * join((@sprintf("%.10g", v) for v in x), ", ") * "]" :
         x isa Real ? @sprintf("%.10g", x) : string(x)

# theta mapped to the unit-height scale: log SDs + 0.5 log c, L21 * sqrt(c)
function unit_theta(fit, tree)
    θ = copy(fit.theta)
    tree == "raw" || return θ
    for (blk, rng) in fit.blocks
        if blk === :resd_mu || blk === :resd_sigma
            θ[rng] .+= SHIFT
        elseif blk === :recov
            i = first(rng); θ[i] += SHIFT; θ[i+1] += SHIFT; θ[i+2] *= sqrt(C)
        end
    end
    θ
end

for (vname, kw) in variants, tree in ("unit", "raw")
    t0 = time()
    try
        fit = drm(f, Gaussian(); data = dat, tree = trees[tree], kw(tree)...)
        dt = time() - t0
        sc = fit.scales
        g(k) = haskey(sc, k) ? fmt(sc[k]) : "NA"
        @printf("FIT|%s|%s|%s|loglik=%.10g|is_converged=%s|fit.converged=%s|time=%.1f|estim=%s|phylo_penalty=%s|blocks=%s|theta=%s|theta_unitscale=%s|sd_mu=%s|sd_sigma=%s|cor=%s|profile_ci_sd_mu=%s|profile_ci_sd_sigma=%s|sd_u=%.10g\n",
                LABEL, vname, tree, loglik(fit), is_converged(fit), fit.converged, dt,
                fit.estim_method, fmt(fit.phylo_penalty),
                join(["$(k)=$(first(r)):$(last(r))" for (k, r) in fit.blocks], ","),
                fmt(fit.theta), fmt(unit_theta(fit, tree)),
                g(:lambda_sd_mu), g(:lambda_sd_sigma), g(:lambda_cor),
                g(:profile_ci_sd_mu), g(:profile_ci_sd_sigma), sdu(tree))
    catch err
        dt = time() - t0
        msg = replace(sprint(showerror, err), '\n' => ' ')
        @printf("FIT|%s|%s|%s|ERROR|time=%.1f|%s\n", LABEL, vname, tree, dt, first(msg, 600))
    end
    flush(stdout)
end
println("# done ", LABEL)
