include(joinpath(@__DIR__, "run_julia_data.jl"))
using DRModels
nd(rows) = NamedTuple{keys(d)}(map(v -> v[rows], values(d)))
fit = drm(bf(F("y_g ~ x1 + f"), F("sigma ~ 1")), DRModels.Gaussian(); data = d)
ib = findall(==("b"), d.f); ic = findall(==("c"), d.f); ia = findall(==("a"), d.f)
for (nm, rows) in (("rows with f in {b,c} only", [ib[1], ic[1]]), ("one row f=b", [ib[1]]), ("one row f=c", [ic[1]]), ("rows with f in {a,b,c}", [ia[1], ib[1], ic[1]]), ("rows with f in {a,c}", [ia[1], ic[1]]))
    try r = predict(fit, nd(rows)); println(nm, ": ok ", round.(r, digits = 4)) catch e; println(nm, ": ERR ", first(replace(sprint(showerror, e), '\n' => ' '), 200)) end
end
println("reference (all rows) for those: ", round.(predict(fit, d)[[ib[1], ic[1]]], digits = 4))
