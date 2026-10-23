# Sigma-only phylo (asymmetric block) survey. Usage: julia --project=<proj> fit_sigma_only.jl <label>
using DRModels, DelimitedFiles, Printf, LinearAlgebra
label = isempty(ARGS) ? "?" : ARGS[1]
N5 = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"
raw = readdlm(joinpath(N5, "carni70.csv"), ','; header = true)[1]
dat = (; species = String.(strip.(String.(raw[:, 1]), '"')), lsize = Float64.(raw[:, 2]),
         lrange = Float64.(raw[:, 3]))
trees = [("raw", read(joinpath(N5, "carni70.nwk"), String)),
         ("unit", read(joinpath(N5, "carni70_unit.nwk"), String))]
println("project=$label  pathof(DRModels)=", pathof(DRModels)); flush(stdout)

form() = bf(@formula(lsize ~ 1 + lrange), @formula(sigma ~ 1 + phylo(1 | species)))

variants = [("ML", (;)), ("REML", (; method = :REML))]
res = Dict{Tuple{String,String},Any}()
for (vname, kw) in variants, (tname, nwk) in trees
    t0 = time()
    try
        fit = drm(form(), Gaussian(); data = dat, tree = nwk, kw...)
        dt = time() - t0
        sds = try gaussian_locscale_phylo_sds(fit) catch e; sprint(showerror, e) end
        se = try sqrt.(abs.(diag(fit.vcov))) catch; fill(NaN, length(fit.theta)) end
        res[(vname, tname)] = loglik(fit)
        @printf("RESULT project=%s variant=%s tree=%s loglik=%.10g is_converged=%s raw_converged=%s time_s=%.1f\n",
                label, vname, tname, loglik(fit), is_converged(fit), fit.converged, dt)
        println("   theta=", repr(fit.theta))
        println("   se=", repr(se))
        println("   blocks=", fit.blocks, " coefnames=", fit.coefnames)
        println("   phylo_sds=", sds, "  estim_method=", fit.estim_method,
                "  ml_loglik=", fit.ml_loglik, "  reml_loglik=", fit.reml_loglik)
    catch err
        dt = time() - t0
        res[(vname, tname)] = err
        @printf("RESULT project=%s variant=%s tree=%s ERROR time_s=%.1f: %s\n", label, vname, tname, dt,
                first(sprint(showerror, err), 600))
    end
    flush(stdout)
end
for (vname, _) in variants
    a = res[(vname, "raw")]; b = res[(vname, "unit")]
    if a isa Number && b isa Number
        @printf("DELTA project=%s variant=%s raw_minus_unit=%.6e\n", label, vname, a - b)
    else
        println("DELTA project=$label variant=$vname not numeric: raw=$(a isa Number ? a : typeof(a)) unit=$(b isa Number ? b : typeof(b))")
    end
end
flush(stdout)
# Warm (post-compilation) timing of the raw and unit ML fits.
for (tname, nwk) in trees
    t0 = time()
    try
        fit = drm(form(), Gaussian(); data = dat, tree = nwk)
        @printf("WARM project=%s variant=ML tree=%s loglik=%.10g is_converged=%s time_s=%.2f\n",
                label, tname, loglik(fit), is_converged(fit), time() - t0)
    catch err
        @printf("WARM project=%s variant=ML tree=%s ERROR time_s=%.2f: %s\n", label, tname, time() - t0,
                first(sprint(showerror, err), 300))
    end
    flush(stdout)
end
