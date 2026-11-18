# Sibling survey (#952): Gaussian sigma-phylo REML variants on ade4::carni70,
# raw tree (height 53.8) vs unit-height tree. Usage: julia --project=<proj> reml_variants.jl <label>
using DRModels, DelimitedFiles, Printf

const LABEL = isempty(ARGS) ? "?" : ARGS[1]
const D = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
raw = readdlm(joinpath(D, "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')),
         lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
trees = [("raw", read(joinpath(D, "carni70.nwk"), String)),
         ("unit", read(joinpath(D, "carni70_unit.nwk"), String))]
for (tn, nwk) in trees
    println("TREE ", tn, " height = ", @sprintf("%.12g", phylo_tree_height(augmented_phy(nwk))))
end
flush(stdout)

f_both = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1 + phylo(1 | species)))
f_asym = bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1 + phylo(1 | species)))
variants = [
    ("a_separate",   f_both, (; method = :REML)),
    ("b_coupled",    f_both, (; method = :REML, phylo_coupled = true)),
    ("c_asymmetric", f_asym, (; method = :REML)),
]

g10(x) = @sprintf("%.10g", x)
results = Dict{Tuple{String,String},Any}()
for (vn, f, kw) in variants, (tn, nwk) in trees
    t0 = time()
    try
        fit = drm(f, Gaussian(); data = dat, tree = nwk, kw...)
        dt = time() - t0
        sds = try
            s = gaussian_locscale_phylo_sds(fit); "sd_mu=$(g10(s.sd_mu)) sd_sigma=$(g10(s.sd_sigma))"
        catch e
            "sds_err=$(sprint(showerror, e))"
        end
        cor = haskey(fit.scales, :lambda_cor) ? " lambda_cor=$(g10(fit.scales[:lambda_cor][1]))" : ""
        println("FIT ", LABEL, " ", vn, " ", tn,
                " loglik=", g10(loglik(fit)),
                " reml_loglik=", g10(fit.reml_loglik),
                " ml_loglik=", g10(fit.ml_loglik),
                " estim=", fit.estim_method,
                " is_converged=", is_converged(fit),
                " raw_converged=", fit.converged,
                " df=", length(fit.theta),
                " time_s=", @sprintf("%.1f", dt),
                " theta=[", join(g10.(fit.theta), ", "), "]",
                " blocks=", join(["$(k)=>$(r)" for (k, r) in fit.blocks], ","),
                " ", sds, cor)
        results[(vn, tn)] = (loglik(fit), fit.ml_loglik, is_converged(fit))
    catch err
        dt = time() - t0
        msg = replace(sprint(showerror, err), '\n' => ' ')
        println("FIT ", LABEL, " ", vn, " ", tn, " ERROR time_s=", @sprintf("%.1f", dt),
                " err=", first(msg, 400))
        results[(vn, tn)] = ("ERROR", msg)
    end
    flush(stdout)
end

for (vn, _, _) in variants
    r = results[(vn, "raw")]; u = results[(vn, "unit")]
    if r isa Tuple && u isa Tuple && r[1] isa Real && u[1] isa Real
        println("DELTA ", LABEL, " ", vn, " |reml raw-unit|=", @sprintf("%.3e", abs(r[1] - u[1])),
                " (raw-unit=", @sprintf("%.10g", r[1] - u[1]), ")",
                " |ml raw-unit|=", @sprintf("%.3e", abs(r[2] - u[2])),
                " conv raw/unit=", r[3], "/", u[3])
    else
        println("DELTA ", LABEL, " ", vn, " n/a raw=", r, " unit=", u)
    end
end
flush(stdout)
