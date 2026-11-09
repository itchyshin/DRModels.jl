include(joinpath(@__DIR__, "run_julia_data.jl"))
using DRModels
nd(rows) = NamedTuple{keys(d)}(map(v -> v[rows], values(d)))
ia = findall(==("a"), d.f); ib = findall(==("b"), d.f); ic = findall(==("c"), d.f)
for (fam, y, dp) in (("Poisson", "y_pois", nothing), ("Gaussian", "y_g", "sigma ~ 1"), ("Gamma", "y_gam", "sigma ~ 1"), ("Binomial", "y_bin", nothing))
    F0 = F(y * " ~ x1 + f"); fm = dp === nothing ? bf(F0) : bf(F0, F(dp))
    fit = drm(fm, getfield(DRModels, Symbol(fam))(); data = d)
    for (nm, rows) in (("one row f=a", [ia[1]]), ("one row f=b", [ib[1]]), ("rows f in {a,b}", [ia[1], ib[1]]), ("rows f in {a,b,c}", [ia[1], ib[1], ic[1]]))
        try r = predict(fit, nd(rows)); println(fam, " ", nm, ": ok") catch e; println(fam, " ", nm, ": ERR ", first(replace(sprint(showerror, e), '\n' => ' '), 150)) end
    end
end
