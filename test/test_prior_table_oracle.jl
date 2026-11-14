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
    rows = Dict("none"=>[], "site"=>[site], "observer"=>[observer], "both"=>[site,observer],
                "mixed"=>[meanprior,scaleprior,site,observer])
    fits = Dict()
    for (dataset, y) in PTO.Y, (case, entries) in rows
        data = (y=y, x=PTO.X, site=PTO.SITE, observer=PTO.OBSERVER)
        if case == "none"
            fits[(dataset,case)] = drm(f, Gaussian(); data=data)
        else
            table = (prior=first.(entries), class=getindex.(entries,2),
                     coef=getindex.(entries,3), dpar=getindex.(entries,4))
            fits[(dataset,case)] = drm(f, Gaussian(); data=data, priors=table)
        end
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

# Frozen independent fit references; poorly conditioned cases use objective and score.
const FIT_REFS = [
    ("0.0", "site", [0.7872589819672865, 0.49119924264289433, -0.930288275979998, 0.20223007908831026, -2.1783762004723175, -1.830960726456006], 6.955856310778509, 0.651594897536796),
    ("0.0", "observer", [0.8149605226279506, 0.5822923023302564, -0.9993609452883626, 0.3557918801387766, -16.512010260409255, -0.8418250686993594], 9.217787620436706, 5.434378656311831e-13),
    ("0.0", "both", [0.8113362429885886, 0.5813734906399938, -0.9708309774197271, 0.3142870941197911, -2.0892947817558984, -0.8060481090056852], 9.857406912147674, 1.234986706316373),
    ("0.0", "mixed", [0.7802184868694478, 0.5555163756687934, -0.9567053595611478, 0.19846385711801015, -2.0603491398419433, -0.8171095680772049], 11.87627835693656, 1.2368787050409606),
    ("0.03", "site", [0.7996308544575852, 0.5079627829538144, -0.9258027251289521, 0.21524732472586777, -2.183137327057964, -1.9401612866886209], 6.884262084462223, 0.4743401009973526),
    ("0.03", "observer", [0.8305747906884244, 0.6077735417037295, -0.9954678349090593, 0.35875007623973443, -16.938366784261454, -0.8631867901067788], 9.231267127782193, 2.1663421436278942e-13),
    ("0.03", "both", [0.8264895857574446, 0.6073856726218252, -0.9706162760800403, 0.32408654276672005, -2.0751895726003537, -0.8205654733899002], 9.854398710118645, 1.2476679356764186),
    ("0.03", "mixed", [0.79415632772586, 0.5791150229577176, -0.9559449966751347, 0.20531561197927925, -2.047976410049195, -0.8304809164606421], 11.883277124826975, 1.2440184979933653),
    ("0.6", "site", [1.0042670373849314, 0.7823754425803628, -0.7326986537120348, 0.2144310500747022, -1.6925484802883977, -15.311250489186639], 8.699854346565562, 6.282608142704478e-13),
    ("0.6", "observer", [1.1188734723823153, 1.1455317860241663, -0.9517287750120511, 0.48168816089194133, -1.3724086601941448, -0.5978633800680337], 11.319982940325225, 1.8106976844137708),
    ("0.6", "both", [1.1179646932486962, 1.1468397877200296, -0.9591663756632951, 0.4911149674744522, -1.2938126832034267, -0.5857227059342195], 11.246107709642493, 2.3235245128334805),
    ("0.6", "mixed", [1.0252264889415064, 1.0516532773247043, -0.9035204150064543, 0.29168167944460177, -1.321918665368938, -0.6092849294689564], 13.470573859611765, 2.1318646414734275),
]

@testset "prior-table fitted optima against dense oracle" begin
f=bf(@formula(y~x+(1|site)+(1|observer)),@formula(sigma~x))
rows=[("normal(0.25,1.4)","fixef","(Intercept)","mu"),("normal(-0.1,0.7)","fixef","x","sigma"),("gamma(0.6666666666666666,2)","ranef_sd","site","mu"),("gamma(2.0588235294117645,3.5)","ranef_sd","observer","mu")]
choices=Dict("site"=>[3],"observer"=>[4],"both"=>[3,4],"mixed"=>[1,2,3,4])
@test length(FIT_REFS) == 12
for (dataset,case,ref,objective,eig) in FIT_REFS
 d=(y=PTO.Y[dataset],x=PTO.X,site=PTO.SITE,observer=PTO.OBSERVER); rr=rows[choices[case]]
 p=(prior=first.(rr),class=getindex.(rr,2),coef=getindex.(rr,3),dpar=getindex.(rr,4))
 fit=drm(f,Gaussian();data=d,priors=p)
 delta=maximum(abs.(fit.theta.-ref)); diff=abs(fit.nll(fit.theta)-objective); score=maximum(abs.(ForwardDiff.gradient(fit.nll,fit.theta)))
 selected = case == "site" ? [5] : case == "observer" ? [6] : [5,6]
 @test all(isfinite.(fit.theta[selected]) .& (fit.theta[selected] .> -17.99))
 @test diff<1e-6
 @test score<1e-4
 if eig>1e-4; @test delta<1e-5; end
end
end
