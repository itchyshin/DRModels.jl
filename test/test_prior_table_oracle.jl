using Test, DRModels, ForwardDiff
using SpecialFunctions: loggamma
include("fixtures/prior_table_oracle.jl")
const PTO = TableOracleFixture

@testset "prior tables against independent dense oracle" begin
    f = bf(@formula(y ~ x + (1 | site) + (1 | observer)), @formula(sigma ~ x))
    site = ("gamma(0.6666666666666666,2)", "ranef_sd", "site", "mu")
    observer = ("gamma(2.0588235294117645,3.5)", "ranef_sd", "observer", "mu")
    meanprior = ("normal(0.25,1.4)", "fixef", "(Intercept)", "mu")
    scaleprior = ("normal(-0.1,0.7)", "fixef", "x", "sigma")
    rows = Dict("site"=>[site], "observer"=>[observer], "both"=>[site,observer],
                "mixed"=>[meanprior,scaleprior,site,observer])
    fits = Dict()
    for (dataset, y) in PTO.Y, (case, entries) in rows
        table = (prior=first.(entries), class=getindex.(entries,2),
                 coef=getindex.(entries,3), dpar=getindex.(entries,4))
        data = (y=y, x=PTO.X, site=PTO.SITE, observer=PTO.OBSERVER)
        fits[(dataset,case)] = drm(f, Gaussian(); data=data, priors=table)
    end
    for (dataset,case,q,obj,grad,hess) in PTO.RECORDS
        fit = fits[(dataset,case)]
        @test abs(fit.nll(q)-obj) <= 1e-8*max(1,abs(obj))
        @test maximum(abs.(ForwardDiff.gradient(fit.nll,q).-grad)) <= 1e-6
        @test all(abs.(ForwardDiff.hessian(fit.nll,q).-hess) .<= 1e-5 .* max.(1,abs.(hess)))
    end
    # Deliberately incorrect objectives must differ from the frozen oracle.
    record = first(filter(r -> r[1] == "0.6" && r[2] == "mixed", PTO.RECORDS))
    _, _, q, oracle, _, _ = record
    correct = fits[("0.6", "mixed")]
    data = (y=PTO.Y["0.6"], x=PTO.X, site=PTO.SITE, observer=PTO.OBSERVER)
    plain = drm(f, Gaussian(); data=data)
    swapped_rows = [meanprior, scaleprior,
        (site[1], "ranef_sd", "observer", "mu"),
        (observer[1], "ranef_sd", "site", "mu")]
    swapped_table = (prior=first.(swapped_rows), class=getindex.(swapped_rows,2),
                     coef=getindex.(swapped_rows,3), dpar=getindex.(swapped_rows,4))
    swapped = drm(f, Gaussian(); data=data, priors=swapped_table)
    # The target identity is independent of both formula order and table order.
    reverse_form = bf(@formula(y ~ x + (1 | observer) + (1 | site)), @formula(sigma ~ x))
    reversed_rows = reverse(rows["mixed"])
    reversed_table = (prior=first.(reversed_rows), class=getindex.(reversed_rows,2),
                      coef=getindex.(reversed_rows,3), dpar=getindex.(reversed_rows,4))
    reversed = drm(reverse_form, Gaussian(); data=data, priors=reversed_table)
    perm = [1,2,3,4,6,5]; qrev = q[perm]
    @test reversed.nll(qrev) ≈ oracle atol=1e-8
    @test maximum(abs.(ForwardDiff.gradient(reversed.nll,qrev) .- record[5][perm])) <= 1e-6
    @test all(abs.(ForwardDiff.hessian(reversed.nll,qrev) .- record[6][perm,perm]) .<= 1e-5 .* max.(1,abs.(record[6][perm,perm])))
    penalty = correct.nll(q) - plain.nll(q)
    wrongrate_rows = [meanprior, scaleprior,
        ("gamma(1.5,3)", "ranef_sd", "site", "mu"), observer]
    wrongrate_table = (prior=first.(wrongrate_rows), class=getindex.(wrongrate_rows,2),
                       coef=getindex.(wrongrate_rows,3), dpar=getindex.(wrongrate_rows,4))
    wrongrate = drm(f, Gaussian(); data=data, priors=wrongrate_table)
    normalizers = log(1.4) + log(0.7) + log(2π) - 2log(3) + loggamma(2) - 3.5log(1.7) + loggamma(3.5)
    fixed_penalty = 0.5*((q[1]-0.25)/1.4)^2 + log(1.4) + 0.5log(2π) +
                    0.5*((q[4]+0.1)/0.7)^2 + log(0.7) + 0.5log(2π)
    site_penalty = 3exp(q[5]) - q[5] - 2log(3) + loggamma(2)
    observer_penalty = 1.7exp(q[6]) - 2.5q[6] - 3.5log(1.7) + loggamma(3.5)
    # Each crossed grouping has three levels; this mutant wrongly repeats
    # only its SD penalty by level, while leaving the fixed penalty unchanged.
    per_level = plain.nll(q) + fixed_penalty + 3site_penalty + 3observer_penalty
    wrong = (swapped.nll(q), wrongrate.nll(q), correct.nll(q)-normalizers,
             plain.nll(q), correct.nll(q)+penalty, per_level, correct.nll(q)-q[5]-q[6])
    for value in wrong
        @test abs(value-oracle) > 1e-6
    end
end
