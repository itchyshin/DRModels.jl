# Julia side of the bivariate missing-response agreement check against drmTMB 0.8.1.
# Usage (from the repository root): julia --project=. docs/dev-log/2026-11-09-bivariate-predict/check_agreement.jl
# Reads r_out.txt, data_seed*.csv and fitted_seed*.csv written by make_r.R. Rule (pre-registered in the dev-log note):
#   |dLL| <= 1e-4; mean and rho12 coefficients within 1e-3 * SE_R; log SD coefficients within 5e-3;
#   fitted values within 1e-6; nobs equal.
using DRModels, DelimitedFiles, Printf
const DIR = @__DIR__
function load(seed)
    m, h = readdlm(joinpath(DIR, "data_seed$seed.csv"), ',', Any; header = true)
    cols = Pair{Symbol,Any}[]
    for (j, n) in enumerate(String.(vec(h)))
        c = m[:, j]
        push!(cols, Symbol(n) => any(x -> x == "NA", c) ? Union{Missing,Float64}[x == "NA" ? missing : Float64(x) for x in c] : Float64.(c))
    end
    return (; cols...)
end
function parse_r()
    res = Dict{String,Dict{String,Any}}(); key = ""
    for l in eachline(joinpath(DIR, "r_out.txt"))
        if startswith(l, "seed=")
            m = match(r"seed=(\d) mode=(\w+) cell=(\w+) logLik=(\S+) nobs=(\d+)", l)
            key = "$(m[1])|$(m[2])|$(m[3])"
            res[key] = Dict{String,Any}("ll" => parse(Float64, m[4]), "nobs" => parse(Int, m[5]))
        elseif startswith(strip(l), "coef:")
            res[key]["coef"] = [parse(Float64, split(t, '=')[2]) for t in split(strip(l)[7:end])]
        elseif startswith(strip(l), "se:")
            res[key]["se"] = [parse(Float64, t) for t in split(strip(l)[5:end])]
        end
    end
    return res
end
forms(cell) = cell == "both" ? (@formula(y1m ~ x1), @formula(y2m ~ x2)) :
              cell == "y1" ? (@formula(y1m ~ x1), @formula(y2 ~ x2)) : (@formula(y1 ~ x1), @formula(y2m ~ x2))
R = parse_r(); fails = 0; total = 0
@printf("%-26s %10s %5s %6s %9s %9s %9s %s\n", "cell", "dLL", "nobs", "dnobs", "max|dβ|/SE", "max|dlogsd|", "max|dfit|", "verdict")
for seed in 1:3, mode in ("default", "include", "drop"), cell in ("both", "y1", "y2")
    global fails, total
    r = R["$seed|$mode|$cell"]; d = load(seed); f1, f2 = forms(cell)
    fm = bf(mu1 = f1, mu2 = f2, sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = mode == "default" ? drm(fm, Gaussian(); data = d) :
          drm(fm, Gaussian(); data = d, missing = miss_control(response = mode))
    th = coef(fit)
    # coefficient order in both engines: mu1(2), mu2(2), sigma1, sigma2, rho12
    se = r["se"]; Δ = abs.(th .- r["coef"])
    mean_idx = [1, 2, 3, 4, 7]; lsd_idx = [5, 6]
    zb = maximum(Δ[mean_idx] ./ se[mean_idx]); zs = maximum(Δ[lsd_idx])
    fv = fitted(fit); Rf = readdlm(joinpath(DIR, "fitted_seed$(seed)_$(mode)_$cell.csv"), ',', Float64)
    Jf = hcat(fv[:mu1], fv[:mu2])
    dfit = size(Jf) == size(Rf) ? maximum(abs.(Jf .- Rf)) : Inf
    dll = abs(loglik(fit) - r["ll"]); dn = nobs(fit) - r["nobs"]
    ok = dll <= 1e-4 && zb <= 1e-3 && zs <= 5e-3 && dfit <= 1e-6 && dn == 0
    fails += !ok; total += 1
    @printf("%-26s %10.2e %5d %6d %9.2e %9.2e %9.2e %s\n", "seed$seed $mode $cell", dll, nobs(fit), dn, zb, zs, dfit, ok ? "AGREE" : "FAIL")
end
println("\n$(total - fails) of $total agree")
exit(fails == 0 ? 0 : 1)
