# Sibling survey for DRModels.jl #952: Gaussian mu-only phylo route,
# bf(lsize ~ 1 + lrange + phylo(1 | species), sigma ~ 1), raw vs unit-height tree.
# Usage: julia --project=/opt/base|/opt/fix run_mu_only.jl <label>
using DRModels, DelimitedFiles, Printf

label = isempty(ARGS) ? "unknown" : ARGS[1]
D = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
raw = readdlm(joinpath(D, "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')),
         lsize = Float64.(raw[:, 2]), lrange = Float64.(raw[:, 3]))
trees = Dict("raw" => read(joinpath(D, "carni70.nwk"), String),
             "unit" => read(joinpath(D, "carni70_unit.nwk"), String))
for (k, v) in trees
    println("TREE ", k, " height=", phylo_tree_height(augmented_phy(v)))
end
flush(stdout)

form() = bf(@formula(lsize ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1))

variants = [
    ("auto",          (;)),
    ("sparse",        (; algorithm = :sparse)),
    ("sparse_lbfgs",  (; algorithm = :sparse_lbfgs)),
    ("em",            (; algorithm = :em)),
    ("gls",           (; algorithm = :gls)),
    ("sparse_true",   (; sparse = true)),
    ("auto_REML",     (; method = :REML)),
]

# Warm-up (compile) on the unit tree, default algorithm; not reported as a result.
t0 = time()
try
    drm(form(), Gaussian(); data = dat, tree = trees["unit"])
catch e
    println("WARMUP error: ", sprint(showerror, e))
end
@printf("WARMUP done in %.1f s\n", time() - t0); flush(stdout)

results = Dict{Tuple{String,String},Any}()
for (vname, kw) in variants, tname in ("raw", "unit")
    t = time()
    line = try
        fit = drm(form(), Gaussian(); data = dat, tree = trees[tname], kw...)
        el = time() - t
        ll = loglik(fit)
        results[(vname, tname)] = ll
        rs_native = try re_sd(fit) catch e; "ERR:" * sprint(showerror, e) end
        rs_drm = try re_sd(fit; scale = :drmtmb, tree = trees[tname]) catch e; "ERR:" * sprint(showerror, e) end
        reml_ll = try fit.reml_loglik catch; nothing end
        @sprintf("FIT project=%s variant=%s tree=%s loglik=%.10g is_converged=%s converged_flag=%s time_s=%.2f phylo_scale=%s theta=%s re_sd_native=%s re_sd_drmtmb=%s reml_loglik=%s estim_method=%s",
                 label, vname, tname, ll, is_converged(fit), fit.converged, el,
                 string(fit.phylo_scale), string(round.(fit.theta; sigdigits = 10)),
                 string(rs_native), string(rs_drm), string(reml_ll), string(fit.estim_method))
    catch e
        el = time() - t
        results[(vname, tname)] = "ERR"
        msg = replace(sprint(showerror, e), '\n' => ' ')
        @sprintf("FIT project=%s variant=%s tree=%s ERROR time_s=%.2f msg=%s", label, vname, tname, el, first(msg, 400))
    end
    println(line); flush(stdout)
end
for (vname, _) in variants
    a = results[(vname, "raw")]; b = results[(vname, "unit")]
    if a isa Real && b isa Real
        @printf("DELTA project=%s variant=%s raw_minus_unit=%.6e\n", label, vname, a - b)
    else
        println("DELTA project=", label, " variant=", vname, " raw=", a, " unit=", b)
    end
end
flush(stdout)
