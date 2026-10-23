# Diagnostic only: one constrained solve on the existing sparse LSS route.
# Usage: julia --project=. tools/profile-scaling-pilot.jl DEPTH NEW_TOML
using DRModels, Random, LinearAlgebra, Statistics, TOML, SHA, Dates, Sockets
length(ARGS) == 2 || error("DEPTH NEW_TOML required")
depth = parse(Int, ARGS[1]); 4 <= depth <= 8 || error("pilot depth must be 4:8")
output = ARGS[2]; isfile(output) && error("refusing to overwrite evidence")
BLAS.set_num_threads(1)
root = dirname(@__DIR__)
function sources()
    files = sort([joinpath(d,f) for (d,_,fs) in walkdir(joinpath(root,"src")) for f in fs if endswith(f,".jl")])
    Dict(relpath(f,root)=>bytes2hex(sha256(read(f))) for f in files)
end
out = Dict{String,Any}("scope"=>"Synthetic M6q-shaped ML sparse cost diagnostic, not Ayumi reproduction or full CI",
    "source_before"=>sources(), "runner_sha256"=>bytes2hex(sha256(read(@__FILE__))),
    "julia"=>string(VERSION), "threads"=>Threads.nthreads(), "blas"=>BLAS.get_num_threads(),
    "host"=>gethostname(), "started"=>string(now()), "status"=>"STARTED")
save() = open(io->TOML.print(io,out),output,"w")
save()
try
    rng = MersenneTwister(56328)
    function node(prefix,d)
        d == 0 && return "$prefix:$(1/depth)"
        children = "($(node(prefix*"a",d-1)),$(node(prefix*"b",d-1)))"
        d == depth ? children*";" : children*":$(1/depth)"
    end
    tree = augmented_phy(node("s",depth)); G = tree.n_leaves
    # Independent small dense simulator/oracle only; no dense engine substituted.
    K = DRModels.sigma_phy_dense(tree)
    xg = randn(rng,G); zg = randn(rng,G)
    g = repeat(1:G,inner=2); x = xg[g]; z = zg[g]
    X = hcat(ones(length(g)),x,x.^2,z,z.^2)
    beta = [0.3,0.5,-0.1,0.2,0.05]
    delta = [-1.0,0.1,0.02,-0.1,0.01]
    alpha = [-0.4,0.15,-0.03,0.1,0.02]
    Zg = hcat(ones(G),xg,xg.^2,zg,zg.^2)
    u = exp.(Zg*alpha) .* (cholesky(Symmetric(K)).L * randn(rng,G))
    y = X*beta + u[g] + exp.(X*delta).*randn(rng,length(g))
    data = (;y,x,x2=x.^2,z,z2=z.^2,species=tree.leaf_names[g])
    form = bf(@formula(y~x+x2+z+z2+phylo(1|species)),
        @formula(sigma~x+x2+z+z2), @formula(sd(species,phylogenetic)~x+x2+z+z2))
    out["species"] = G;out["observations"] = length(y);out["status"] = "FITTING";save()
    timed = @timed drm(form,Gaussian();data,tree,algorithm=:sparse,method=:ML,g_tol=1e-8)
    fit = timed.value;theta = copy(fit.theta)
    out["fit_seconds_including_compilation"] = timed.time
    out["fit_bytes"] = timed.bytes;out["fit_converged"] = is_converged(fit)
    out["theta"] = theta;out["parameters"] = length(theta)
    out["stored_gradient"] = fit.nllgrad !== nothing
    mode = DRModels._profile_autodiff_mode(fit.nll,fit.nllgrad,theta)
    out["profile_derivative_mode"] = string(mode)
    a = exp.(Zg*coef(fit,:sd_phylo))
    # Residual noise is observation-level; repeat only the phylogenetic part.
    V = Diagonal(exp.(2X*coef(fit,:sigma))) + ((a*a').*K)[g,g]
    C = cholesky(Symmetric(V));r = y-X*coef(fit,:mu)
    dense_nll = (length(y)*log(2pi)+logdet(C)+dot(r,C\r))/2
    out["dense_nll_error"] = abs(dense_nll-fit.nll(theta))
    out["status"] = "CONSTRAINED_SOLVE";save()
    calls = Ref(0)
    k = 2; ids = [i for i in eachindex(theta) if i!=k]
    value = theta[k] + 0.5stderror(fit)[k]
    function obj(u)
        t = copy(theta);t[k]=value;t[ids]=u;calls[]+=1
        fit.nll(t)
    end
    constrained = @timed DRModels._profile_optimize(obj,theta[ids],mode)
    res = constrained.value
    out["constrained_seconds"] = constrained.time;out["constrained_bytes"] = constrained.bytes
    out["objective_calls"] = calls[]
    out["optimizer_iterations"] = DRModels.Optim.iterations(res)
    out["optimizer_converged"] = DRModels.Optim.converged(res)
    out["optimizer_function_calls"] = DRModels.Optim.f_calls(res)
    out["optimizer_gradient_calls"] = DRModels.Optim.g_calls(res)
    uhat = DRModels.Optim.minimizer(res)
    score = map(eachindex(uhat)) do j
        h = cbrt(eps(Float64))*max(1,abs(uhat[j]));up=copy(uhat);um=copy(uhat)
        up[j]+=h;um[j]-=h;(obj(up)-obj(um))/(2h)
    end
    out["nuisance_score_maxabs"] = maximum(abs,score)
    out["profile_deviance"] = 2*(DRModels.Optim.minimum(res)-fit.nll(theta))
    samples = [(@timed fit.nll(theta)) for _ in 1:5]
    out["warm_objective_seconds_median"] = median(t.time for t in samples)
    out["warm_objective_bytes_median"] = median(t.bytes for t in samples)
    out["status"] = "MEASURED"
catch e
    out["status"] = "ERROR";out["error"] = sprint(showerror,e,catch_backtrace())
end
out["source_after"] = sources();out["source_unchanged"] = out["source_before"]==out["source_after"]
save();println("PROFILE_COST_",out["status"])
out["status"] == "MEASURED" && out["source_unchanged"] || exit(1)
