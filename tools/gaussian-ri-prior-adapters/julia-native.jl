# Usage: julia --project=. julia-native.jl INPUT.toml OUTPUT.toml [--fit]
# INPUT contains fixture data and metadata only, never oracle expected values.
using DRModels, ForwardDiff, LinearAlgebra, TOML

length(ARGS) in (2, 3) || error("usage: julia-native.jl INPUT.toml OUTPUT.toml [--fit]")
length(ARGS) == 2 || ARGS[3] == "--fit" || error("unknown option")
f = TOML.parsefile(ARGS[1])
haskey(f, "expected") && error("oracle expected values must not enter adapter input")
raw = f["raw_data"]
dat = (y=Float64.(raw["y"]), x=Float64.(raw["x"]),
       scale_x=Float64.(raw["scale_x"]), fB=Float64.(raw["f"] .== "B"),
       group=Int.(raw["group"]))
mean_cols = f["formula"]["mean_columns"]
scale_cols = f["formula"]["scale_columns"]
factor_mean = mean_cols == ["x", "I(as.numeric(f == B))"]
factor_mean || mean_cols == ["(Intercept)", "x"] || error("unknown mean formula")
varying = scale_cols == ["(Intercept)", "scale_x"]
varying || scale_cols == ["(Intercept)"] || error("unknown scale formula")
mu = factor_mean ? @formula(y ~ 0 + x + fB + (1 | group)) : @formula(y ~ 1 + x + (1 | group))
sigma = varying ? @formula(sigma ~ 1 + scale_x) : @formula(sigma ~ 1)
form = bf(mu, sigma)
rhs = Dict(form.forms)
fixed_mu = first(DRModels._split_ranef(rhs[:mu]))
fixed_sigma = first(DRModels._split_ranef(rhs[:sigma]))
_, X, names_mu = DRModels._design(:y, fixed_mu, dat)
_, W, names_sigma = DRModels._design(:y, fixed_sigma, dat)
mat(key) = reduce(vcat, [permutedims(Float64.(row)) for row in f[key]])
X == mat("X") && W == mat("W") || error("package model matrices differ from frozen fixture")
specs = map(f["priors"]) do p
    block = p["block"] == "beta" ? :mu : p["block"] == "gamma" ? :sigma : error("unknown block")
    names = block === :mu ? names_mu : names_sigma
    name = names[Int(p["index"]) + 1]
    p["kind"] == "normal" ? drm_prior_normal(dpar=string(block), coef=name, mean=p["mean"], sd=p["sd"]) :
    p["kind"] == "ridge" ? drm_prior_ridge(dpar=string(block), coef=name, lambda=p["lam"]) : error("unknown prior")
end
fit = drm(form, Gaussian(); data=dat, priors=drm_priors(specs...))
p = f["point"]; tau = Float64(p["tau"])
tau >= 0 || error("negative random-intercept SD")
out = Dict{String,Any}("id"=>f["id"], "mean_names"=>names_mu, "scale_names"=>names_sigma,
    "matrix_verified"=>true, "coordinate_order"=>vcat(names_mu, names_sigma, ["log_group_sd"]),
    "point_tau"=>tau, "boundary"=>tau == 0)
if tau > 0
    theta = vcat(Float64.(p["beta"]), Float64.(p["gamma"]), log(tau))
    objective = fit.nll(theta); data_nll = fit.penalty.data_nll(theta)
    out["point"] = Dict("theta"=>theta, "data_nll"=>data_nll,
        "prior"=>objective-data_nll, "objective"=>objective,
        "gradient"=>ForwardDiff.gradient(fit.nll, theta),
        "hessian"=>[collect(row) for row in eachrow(ForwardDiff.hessian(fit.nll, theta))])
else
    # The log-SD parameterization cannot represent zero. Evaluate its exact
    # Gaussian boundary through the package's fixed-effect route, separately.
    fixed_form = bf(factor_mean ? @formula(y ~ 0 + x + fB) : @formula(y ~ 1 + x), sigma)
    fixed_fit = drm(fixed_form, Gaussian(); data=dat, priors=drm_priors(specs...))
    theta = vcat(Float64.(p["beta"]), Float64.(p["gamma"]))
    data_nll = fixed_fit.penalty.data_nll(theta)
    out["point"] = Dict("theta_fixed"=>theta, "data_nll"=>data_nll,
        "prior"=>fixed_fit.nll(theta)-data_nll, "objective"=>fixed_fit.nll(theta),
        "boundary_derivatives"=>"unavailable_in_log_sd_parameterization")
end
if length(ARGS) == 3
    out["fit"] = Dict("theta"=>fit.theta, "tau"=>exp(last(fit.theta)),
        "data_nll"=>fit.penalty.data_nll(fit.theta), "objective"=>fit.nll(fit.theta),
        "gradient"=>ForwardDiff.gradient(fit.nll, fit.theta), "converged"=>fit.converged)
end
open(ARGS[2], "w") do io
    TOML.print(io, out)
end
