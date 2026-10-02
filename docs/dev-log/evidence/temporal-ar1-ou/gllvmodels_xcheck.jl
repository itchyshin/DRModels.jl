# GLLVModels.jl origin/main 5712e35d5, DRModels.jl claude/temporal-ar1-ou; Julia 1.10.12, Totoro, 2026-10-01.
# One-off cross-check: DRModels temporal() vs GLLVModels fit_temporal_gllvm.
# GLLVModels' temporal port admits >= 3 traits, so the fixture is stacked as three
# IDENTICAL traits under temporal_indep: traits are then independent with shared
# phi/decay and sigma_eps, and by symmetry the joint MLE has equal trait SDs and
# intercepts, so its logLik must be exactly 3 x the DRModels single-trait logLik.
using DRModels, GLLVModels, DelimitedFiles, StatsModels, Logging
# Run from an environment that develops both packages (paths below are Totoro's).
const FX = expanduser("~/lanes/DRModels.jl-temporal-j110/test/fixtures/temporal")
function load(file, tcol)
    raw, hdr = readdlm(joinpath(FX, file), ','; header = true)
    col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
    (y = Float64.(col("y")), x = Float64.(col("x")), id = String.(col("id")),
     t = tcol == "occ" ? Int.(col(tcol)) : Float64.(col(tcol)))
end
for (file, tcol, st) in (("ar1_gapped.csv", "occ", :ar1), ("ou_irregular.csv", "elapsed", :ou))
    d = load(file, tcol)
    f = st === :ar1 ? bf(@formula(y ~ x + temporal(1 | id, t, ar1)), @formula(sigma ~ 1)) :
                      bf(@formula(y ~ x + temporal(1 | id, t, ou)), @formula(sigma ~ 1))
    m = with_logger(NullLogger()) do; drm(f, Gaussian(); data = d); end
    n = length(d.y)
    long = (y = repeat(d.y, 3), x = repeat(d.x, 3), id = repeat(d.id, 3), t = repeat(d.t, 3),
            trait = repeat(["a", "b", "c"], inner = n))
    g = GLLVModels.fit_temporal_gllvm(long; formula = @formula(y ~ 0 + trait + x),
        temporal = GLLVModels.temporal_indep(:(0 + trait | id), :t; structure = st), g_tol = 1e-10)
    println(file, " (", st, ", no ordinary intercept)")
    println("  DRModels   logLik = ", loglik(m), "  ", temporal_parameters(m), "  beta = ", coef(m, :mu))
    println("  GLLVModels logLik / 3 = ", g.loglik / 3, "  par = ", round.(g.parameters; digits = 5))
    println("  |diff| = ", abs(loglik(m) - g.loglik / 3))
end
# OU + ordinary (1 | id): GLLVModels' `(1 | g)` has ONE SD shared across traits,
# so the intercept grouping is made per (id, trait) — `idt` — which keeps the three
# stacked traits independent and the symmetry argument (logLik = 3 x) intact.
d = load("ou_irregular.csv", "elapsed")
m = with_logger(NullLogger()) do
    drm(bf(@formula(y ~ x + (1 | id) + temporal(1 | id, t, ou)), @formula(sigma ~ 1)), Gaussian(); data = d)
end
n = length(d.y)
long = (y = repeat(d.y, 3), x = repeat(d.x, 3), id = repeat(d.id, 3), t = repeat(d.t, 3),
        trait = repeat(["a", "b", "c"], inner = n), idt = repeat(d.id, 3) .* "_" .* repeat(["a", "b", "c"], inner = n))
g = GLLVModels.fit_temporal_gllvm(long; formula = @formula(y ~ 0 + trait + x),
    temporal = GLLVModels.temporal_indep(:(0 + trait | id), :t; structure = :ou),
    structure = [:((1 | idt))], g_tol = 1e-10)
println("ou_irregular.csv (ou, + (1 | id))")
println("  DRModels   logLik = ", loglik(m), "  ", temporal_parameters(m))
println("  GLLVModels logLik / 3 = ", g.loglik / 3, "  par = ", round.(g.parameters; digits = 5))
println("  |diff| = ", abs(loglik(m) - g.loglik / 3))

