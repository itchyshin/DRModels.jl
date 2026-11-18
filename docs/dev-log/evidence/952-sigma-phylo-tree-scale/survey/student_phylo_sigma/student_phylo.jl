# Survey: Student() phylo routes on carni70, raw vs unit tree. Usage:
#   julia --project=/opt/base student_phylo.jl base
#   julia --project=/opt/fix  student_phylo.jl fix
using DRModels, DelimitedFiles, Printf
const PROJ = ARGS[1]
const D = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
raw = readdlm(joinpath(D, "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')), lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
trees = ["raw" => read(joinpath(D, "carni70.nwk"), String),
         "unit" => read(joinpath(D, "carni70_unit.nwk"), String)]
println("PROJECT=$PROJ pathof=", pathof(DRModels)); flush(stdout)

variants = [
    "mu_only_phylo" => () -> bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1)),
    "sigma_only_phylo" => () -> bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1 + phylo(1 | species))),
    "mu_and_sigma_phylo" => () -> bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1 + phylo(1 | species))),
    "coupled_1|p|phylo" => () -> bf(@formula(lsize ~ 1 + lrange + (1 | p | phylo(species))), @formula(sigma ~ 1 + (1 | p | phylo(species)))),
    "sigma_only_coupled_1|p|phylo" => () -> bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1 + (1 | p | phylo(species)))),
    "nu_phylo(extra)" => () -> bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1), @formula(nu ~ 1 + phylo(1 | species))),
]

errtext(e) = replace(sprint(showerror, e), '\n' => " | ")

function runfit(label, treelabel, mkf, kw)
    t0 = time()
    try
        f = mkf()
        fit = drm(f, Student(); data = dat, kw...)
        dt = time() - t0
        ll = loglik(fit)
        th = try fit.theta catch; "n/a" end
        sds = try sprint(show, DRModels.ranef_sd(fit)) catch; "n/a" end
        println(@sprintf("FIT proj=%s variant=%s tree=%s status=ok loglik=%.10g converged=%s time_s=%.1f theta=%s sds=%s",
                         PROJ, label, treelabel, ll, is_converged(fit), dt, sprint(show, th), sds))
        flush(stdout)
        return ll
    catch e
        dt = time() - t0
        println(@sprintf("FIT proj=%s variant=%s tree=%s status=ERROR time_s=%.1f error=%s",
                         PROJ, label, treelabel, dt, first(errtext(e), 600)))
        flush(stdout)
        return e
    end
end

# Control: tree-free Student location-scale (no phylo); tree-independent by construction.
runfit("control_no_phylo", "none", () -> bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1)), (;))

for (label, mkf) in variants
    res = Dict{String,Any}()
    for (tl, nwk) in trees
        res[tl] = runfit(label, tl, mkf, (; tree = nwk))
    end
    r, u = res["raw"], res["unit"]
    if r isa Real && u isa Real
        println(@sprintf("DELTA proj=%s variant=%s |raw-unit|=%.3e", PROJ, label, abs(r - u)))
    else
        same = (r isa Exception && u isa Exception) && errtext(r) == errtext(u)
        println("DELTA proj=$PROJ variant=$label |raw-unit|=n/a (errors identical on raw and unit: $same)")
    end
    # Same formula without the tree kwarg: exposes the in-function refusal (if any).
    runfit(label, "no_tree_kwarg", mkf, (;))
    flush(stdout)
end
println("DONE proj=$PROJ")
