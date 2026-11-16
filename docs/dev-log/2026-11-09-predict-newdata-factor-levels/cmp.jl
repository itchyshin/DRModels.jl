include(joinpath(@__DIR__, "..", "..", "..", "test", "test_predict_newdata_factor_levels.jl"))
using DelimitedFiles
open(joinpath(@__DIR__, "pn_data.csv"), "w") do io
    println(io, "x,f,g,y_poisson,y_gamma,y_beta")
    for i in 1:_PN_N
        println(io, join([_PN_X[i], _PN_F[i], _PN_G[i], _PN_Y[:poisson][i], _PN_Y[:gamma][i], _PN_Y[:beta][i]], ","))
    end
end
nd = (; x = [0.3, -0.5, 1.0, 0.0], f = ["c", "a", "d", "b"])
out = Any[]
for (key, fam, extra) in ((:poisson, Poisson(), ()), (:gamma, Gamma(), (@formula(sigma ~ 1),)), (:beta, Beta(), (@formula(sigma ~ 1),)))
    fit = drm(bf(@formula(y ~ x + f), extra...), fam; data = _data(key))
    for (lab, rows) in (("single", [1]), ("subset", [2,3]), ("reordered", [4,3,2,1]))
        l = predict(fit, (; x = nd.x[rows], f = nd.f[rows]); type = :link)
        r = predict(fit, (; x = nd.x[rows], f = nd.f[rows]))
        for k in eachindex(rows); push!(out, (key, lab, rows[k], l[k], r[k], loglikelihood(fit))); end
    end
end
open(joinpath(@__DIR__, "pn_julia.csv"), "w") do io
    println(io, "family,case,row,link,response,loglik")
    for o in out; println(io, join(o, ",")); end
end
