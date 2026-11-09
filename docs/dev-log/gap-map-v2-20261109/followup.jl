# Follow-up probes for the gap-map v2 triage (Julia side). Run with the same env as run_julia.jl.
include(joinpath(@__DIR__, "run_julia_data.jl"))   # defines d and F
using DRModels
t(nm, f) = try r = f(); println("== ", nm, ": ok ", r isa DrmFit ? string("ll=", round(loglikelihood(r), digits = 4)) : typeof(r)); r catch e; println("== ", nm, ": ERR ", first(replace(sprint(showerror, e), '\n' => ' '), 300)); nothing end
fit = drm(bf(F("y_g ~ x1 + f"), F("sigma ~ 1")), DRModels.Gaussian(); data = d)
nd(rows) = NamedTuple{keys(d)}(map(v -> v[rows], values(d)))
t("predict 1 row f=a", () -> predict(fit, nd([1])))
t("predict 1 row (levels of that row only)", () -> predict(fit, nd([findfirst(==("c"), d.f)])))
t("predict 3 rows, levels a,b,c present", () -> predict(fit, nd([findfirst(==("a"), d.f), findfirst(==("b"), d.f), findfirst(==("c"), d.f)])))
t("predict 2 rows, c absent", () -> predict(fit, nd([findfirst(==("a"), d.f), findfirst(==("b"), d.f)])))
# mixed family
mf = t("mixed gaussian+poisson", () -> drm(bf(mu1 = F("y_g ~ x1"), mu2 = F("y_pois ~ x1"), sigma1 = F("sigma1 ~ 1")), (DRModels.Gaussian(), DRModels.Poisson()); data = d))
if mf !== nothing
    println(typeof(mf)); println(keys(mf))
    for k in keys(mf); v = getfield(mf, k); println("   ", k, " = ", v isa AbstractVector ? first(v, 4) : v); end
end
# gaussian REML with factor sigma
t("REML f*x1 sigma=x1+f", () -> drm(bf(F("y_g ~ f * x1"), F("sigma ~ x1 + f")), DRModels.Gaussian(); data = d, method = :REML))
t("REML f*x1 sigma=x1", () -> drm(bf(F("y_g ~ f * x1"), F("sigma ~ x1")), DRModels.Gaussian(); data = d, method = :REML))
t("REML x1 sigma=x1+f", () -> drm(bf(F("y_g ~ x1"), F("sigma ~ x1 + f")), DRModels.Gaussian(); data = d, method = :REML))
t("REML f*x1 sigma=f", () -> drm(bf(F("y_g ~ f * x1"), F("sigma ~ f")), DRModels.Gaussian(); data = d, method = :REML))
# weights with NA response
t("weights + NA response", () -> drm(bf(F("y_gm ~ x1"), F("sigma ~ 1")), DRModels.Gaussian(); data = d, weights = d.w))
# meta_V scale argument silently ignored?
t("meta_V(vi) vs meta_V(vi, scale=...)", () -> drm(bf(F("yi ~ x1 + meta_V(vi)"), F("sigma ~ 1")), DRModels.Gaussian(); data = d))
# skew-normal fixed effects: optimum and params
sk = t("skew y_g", () -> drm(bf(F("y_g ~ x1"), F("sigma ~ x2"), F("nu ~ x1")), DRModels.SkewNormal(); data = d))
sk === nothing || (println("coef: ", round.(coef(sk), digits = 4)); println("conv: ", sk.converged))
t("cumul sigma ~ 1 (what is estimated)", () -> drm(bf(F("y_ord ~ x1"), F("sigma ~ 1")), DRModels.CumulativeLogit(); data = d))
