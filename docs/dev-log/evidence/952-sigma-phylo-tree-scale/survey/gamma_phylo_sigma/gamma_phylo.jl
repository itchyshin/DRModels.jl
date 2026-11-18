# Gamma phylo routes: raw vs unit-height tree. Usage:
#   julia --project=/opt/<base|fix> gamma_phylo.jl <label>
using DRModels, DelimitedFiles, Logging, Printf

const LABEL = isempty(ARGS) ? "unknown" : ARGS[1]
const N5 = "/tmp/claude-0/-home-user-DRModels-jl/bf8c2b27-03aa-5044-87f6-e06cfadb9c02/scratchpad/n5"

raw = readdlm(joinpath(N5, "carni70.csv"), ','; header = true)[1]
species = String.(strip.(String.(raw[:, 1]), '"'))
lsize = Float64.(raw[:, 2]); lrange = Float64.(raw[:, 3])
dat = (; species = species, size = exp.(lsize), lsize = lsize, lrange = lrange)
trees = [("raw", read(joinpath(N5, "carni70.nwk"), String)),
         ("unit", read(joinpath(N5, "carni70_unit.nwk"), String))]

println("# project=$LABEL pkgdir=$(pkgdir(DRModels)) n=$(length(species))")
for (tn, nwk) in trees
    h = try phylo_tree_height(augmented_phy(nwk)) catch e; "ERR $(sprint(showerror, e))" end
    println("# tree=$tn height=$h")
end
flush(stdout)

variants = [
    ("G1_mu_phylo_sigma1",
     bf(@formula(size ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1)), (;)),
    ("G2_mu_phylo_sigmaX",
     bf(@formula(size ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1 + lrange)), (;)),
    ("G3_coupled_phylo_sigma1",
     bf(@formula(size ~ 1 + lrange + (1 | p | phylo(species))),
        @formula(sigma ~ 1 + (1 | p | phylo(species)))), (;)),
    ("G4_coupled_phylo_sigmaX",
     bf(@formula(size ~ 1 + lrange + (1 | p | phylo(species))),
        @formula(sigma ~ 1 + lrange + (1 | p | phylo(species)))), (;)),
    ("R1_separate_phylo_mu_and_sigma",
     bf(@formula(size ~ 1 + lrange + phylo(1 | species)),
        @formula(sigma ~ 1 + phylo(1 | species))), (;)),
    ("R2_sigma_only_phylo",
     bf(@formula(size ~ 1 + lrange), @formula(sigma ~ 1 + phylo(1 | species))), (;)),
    ("R3_coupled_sigma_only",
     bf(@formula(size ~ 1 + lrange), @formula(sigma ~ 1 + (1 | p | phylo(species)))), (;)),
    ("R4_mu_phylo_marginal_Laplace",
     bf(@formula(size ~ 1 + lrange + phylo(1 | species)), @formula(sigma ~ 1)),
     (; marginal = :Laplace)),
]

function phylo_sds(fit)
    b = Dict(fit.blocks)
    out = String[]
    if haskey(b, :resd)
        for i in b[:resd]
            push!(out, @sprintf("sd_phylo_mu=%.10g (log=%.10g)", exp(fit.theta[i]), fit.theta[i]))
        end
    end
    if haskey(b, :recov)
        r = collect(b[:recov])           # [logL11, logL22, L21]
        l11, l22, l21 = fit.theta[r[1]], fit.theta[r[2]], fit.theta[r[3]]
        sm = exp(l11); ss = sqrt(l21^2 + exp(2l22)); cr = l21 / ss
        push!(out, @sprintf("sd_phylo_mu=%.10g sd_phylo_sigma=%.10g cor=%.10g", sm, ss, cr))
    end
    return join(out, " ")
end

results = Dict{Tuple{String,String},Any}()
for (vn, f, kw) in variants, (tn, nwk) in trees
    buf = IOBuffer()
    t0 = time()
    line = try
        fit = with_logger(SimpleLogger(buf, Logging.Warn)) do
            drm(f, DRModels.Gamma(); data = dat, tree = nwk, kw...)
        end
        dt = time() - t0
        results[(vn, tn)] = (loglik(fit), is_converged(fit))
        @sprintf("FIT project=%s variant=%s tree=%s loglik=%.10g loglik_full=%.17g converged=%s fieldconv=%s time_s=%.1f blocks=%s theta=%s %s",
                 LABEL, vn, tn, loglik(fit), loglik(fit), is_converged(fit), fit.converged, dt,
                 string([k => collect(v) for (k, v) in fit.blocks]),
                 string(round.(fit.theta; sigdigits = 10)), phylo_sds(fit))
    catch e
        dt = time() - t0
        msg = replace(sprint(showerror, e), '\n' => ' ')
        results[(vn, tn)] = ("ERR", msg)
        @sprintf("ERR project=%s variant=%s tree=%s time_s=%.1f error=%s",
                 LABEL, vn, tn, dt, first(msg, 600))
    end
    println(line)
    w = replace(String(take!(buf)), '\n' => ' ')
    isempty(strip(w)) || println("   WARN ", first(w, 700))
    flush(stdout)
end
for (vn, _, _) in variants
    a = get(results, (vn, "raw"), nothing); b = get(results, (vn, "unit"), nothing)
    if a !== nothing && b !== nothing && a[1] isa Real && b[1] isa Real
        d = abs(a[1] - b[1])
        @printf("DELTA project=%s variant=%s |raw-unit|=%.3e raw_conv=%s unit_conv=%s invariant=%s\n",
                LABEL, vn, d, a[2], b[2], d <= 1e-6 && a[2] && b[2])
    else
        same = a !== nothing && b !== nothing && a[1] == "ERR" && b[1] == "ERR" && a[2] == b[2]
        println("DELTA project=$LABEL variant=$vn raw=$(a === nothing ? nothing : a[1]) unit=$(b === nothing ? nothing : b[1]) identical_error=$same")
    end
end
println("# done $LABEL")
