using DRModels, Test, Random, LinearAlgebra, Statistics

# A known-parameter fit isolates simulation from optimization. The covariance
# below is hand-written from the shared path lengths of this height-two tree.
function _boot_lss_fixture(; with_phylo=true, iid=Symbol[], varying_phylo=true, missing_tip=false)
    tree = DRModels.augmented_phy("(oak:2,(beech:1,cedar:1):1,(elm:1,(fir:0.5,gum:0.5):0.5):1);")
    labels = ["oak", "beech", "cedar", "elm", "fir", "gum"]
    idx = repeat([5, 1, 6, 3, 4, 2], inner=8)
    ztip = [-1.1, -.4, .2, .7, 1.3, 1.8]
    x = sin.(0.4 .* (1:length(idx)))
    study = repeat([3, 1, 4, 2], 12)
    y = .2 .+ .4 .* x .+ cos.(0.73 .* (1:length(idx)))
    missing_tip && (y[idx .== 2] .= NaN)
    data = (; y, x, z=ztip[idx], species=labels[idx], study)
    # Explicit formula variants keep formula parsing under test.
    f = if with_phylo && iid == [:species, :study]
        bf(@formula(y ~ x + (1 | species) + (1 | study) + phylo(1 | species)),
           @formula(sigma ~ x), @formula(sd(species) ~ z), @formula(sd(study) ~ 1),
           @formula(sd(species, phylogenetic) ~ z))
    elseif with_phylo && iid == [:study] && !varying_phylo
        bf(@formula(y ~ x + (1 | study) + phylo(1 | species)),
           @formula(sigma ~ x), @formula(sd(study) ~ 1))
    elseif with_phylo
        bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x),
           @formula(sd(species, phylogenetic) ~ z))
    else
        bf(@formula(y ~ x + (1 | species) + (1 | study)),
           @formula(sigma ~ x), @formula(sd(species) ~ z), @formula(sd(study) ~ 1))
    end
    theta = [.2, .4, -1., .1]
    blocks = [:mu=>1:2, :sigma=>3:4]
    names = [:mu=>["(Intercept)","x"], :sigma=>["(Intercept)","x"]]
    if !isempty(iid)
        a = iid == [:study] ? [-.9] : [-.6, .2, -.9]
        lo = length(theta)+1; append!(theta,a)
        push!(blocks,:sd=>lo:length(theta))
        push!(names,:sd=>(iid == [:study] ? ["(Intercept)"] :
            ["species: (Intercept)","species: z","study: (Intercept)"]))
    end
    if with_phylo
        a = varying_phylo ? [-.2,.4] : [-.2]
        lo=length(theta)+1; append!(theta,a)
        push!(blocks,:sd_phylo=>lo:length(theta))
        push!(names,:sd_phylo=>(varying_phylo ? ["(Intercept)","z"] : ["(Intercept)"]))
    end
    observed = isfinite.(y)
    fit=DRModels._withformula(DRModels.DrmFit(Gaussian(),blocks,names,theta,
        Matrix{Float64}(I,length(theta),length(theta)),0.,count(observed),true,
        Dict(:mu=>(.2 .+ .4 .* x[observed])),Dict(:mu=>y[observed]),
        Dict(:sigma=>exp.(-1. .+ .1 .* x[observed]))),f)
    K=[1. 0 0 0 0 0; 0 1 .5 0 0 0; 0 .5 1 0 0 0;
       0 0 0 1 .5 .5; 0 0 0 .5 1 .75; 0 0 0 .5 .75 1]
    (;fit,tree,data,K,idx,ztip,iid,with_phylo,varying_phylo,observed)
end

function _boot_lss_oracle(q, seed)
    rng=MersenneTwister(seed)
    out=.2 .+ .4 .* q.data.x
    for g in q.iid
        vals=getproperty(q.data,g); levels=unique(vals)
        index=[findfirst(==(v),levels) for v in vals]
        scale = g==:study ? fill(exp(-.9),length(levels)) :
            [exp(-.6+.2*q.data.z[findfirst(==(v),vals)]) for v in levels]
        out .+= (scale .* randn(rng,length(levels)))[index]
    end
    if q.with_phylo
        scale=q.varying_phylo ? exp.(-.2 .+ .4 .* q.ztip) : fill(exp(-.2),6)
        out .+= (scale .* (cholesky(Symmetric(q.K)).L * randn(rng,6)))[q.idx]
    end
    out .+= exp.(-1. .+ .1 .* q.data.x) .* randn(rng,length(out))
    out[.!q.observed] .= NaN
    out
end

@testset "Gaussian LSS bootstrap contract" begin
@testset "LSS marginal simulator preserves all named components and mask" begin
    cases=[(;with_phylo=true,iid=Symbol[]), (;with_phylo=true,iid=[:species,:study]),
           (;with_phylo=true,iid=[:study],varying_phylo=false),
           (;with_phylo=false,iid=[:species,:study]),
           (;with_phylo=true,iid=Symbol[],missing_tip=true)]
    for args in cases
        q=_boot_lss_fixture(;args...)
        sim=DRModels._marginal_simulator(q.fit,q.data;tree=q.tree)
        @test sim !== nothing
        if sim !== nothing
            for seed in (19,83)
                draw=sim(MersenneTwister(seed)); expected=_boot_lss_oracle(q,seed)
                @test length(draw)==length(expected)
                if length(draw)==length(expected)
                    @test isnan.(draw)==isnan.(expected)
                    @test draw[q.observed] ≈ expected[q.observed] atol=1e-12 rtol=1e-12
                end
            end
        end
    end
end

@testset "Public LSS REML bootstrap preserves estimator and seeds" begin
    q=_boot_lss_fixture(with_phylo=false,iid=[:species,:study])
    # A real fitted model and manual REML refits verify the public dispatch.
    fit=drm(q.fit.formula,Gaussian();data=q.data,method=:REML)
    @test estimation_method(fit)==:REML
    B=3; seed=921
    sim=DRModels._marginal_simulator(fit,q.data)
    if sim === nothing
        @test sim !== nothing
    else
        seeds=rand(MersenneTwister(seed),UInt,B)
        draws=hcat([coef(drm(fit.formula,Gaussian();
            data=merge(q.data,(;y=sim(MersenneTwister(s)))),method=:REML)) for s in seeds]...)'
        expected=vec(std(draws;dims=1))
        result=bootstrap_result(fit;data=q.data,B,rng=MersenneTwister(seed))
        @test result.seeds==seeds
        @test [r.std_error for r in result.summary] ≈ expected atol=1e-10 rtol=1e-10
        threaded=bootstrap_result(fit;data=q.data,B,rng=MersenneTwister(seed),threads=true)
        @test threaded.summary==result.summary
        @test threaded.failed==result.failed==0
    end
end

@testset "Existing single-phylo simulator isolates REML refit loss" begin
    q=_boot_lss_fixture()
    fit=drm(q.fit.formula,Gaussian();data=q.data,tree=q.tree,method=:REML)
    B=3; seed=409
    sim=DRModels._marginal_simulator(fit,q.data;tree=q.tree)
    @test sim !== nothing
    if sim !== nothing
        seeds=rand(MersenneTwister(seed),UInt,B)
        draws=hcat([coef(drm(fit.formula,Gaussian();tree=q.tree,
            data=merge(q.data,(;y=sim(MersenneTwister(s)))),method=:REML)) for s in seeds]...)'
        result=bootstrap_result(fit;data=q.data,tree=q.tree,B,rng=MersenneTwister(seed))
        @test result.seeds==seeds
        @test [r.std_error for r in result.summary] ≈ vec(std(draws;dims=1)) atol=1e-10 rtol=1e-10
    end
end

@testset "LSS simulator refuses corrupt contracts and preserves actual missing" begin
    q=_boot_lss_fixture(with_phylo=true,iid=[:species,:study],missing_tip=true)
    missing_y=Union{Missing,Float64}[isnan(y) ? missing : y for y in q.data.y]
    dat=merge(q.data,(;y=missing_y))
    sim=DRModels._marginal_simulator(q.fit,dat;tree=q.tree)
    got=sim(MersenneTwister(19)); expected=_boot_lss_oracle(q,19)
    @test isnan.(got)==ismissing.(missing_y)
    @test got[q.observed] ≈ expected[q.observed] atol=1e-12 rtol=1e-12
    @test_throws ArgumentError DRModels._marginal_simulator(q.fit,dat)
    bad=deepcopy(q.fit); Dict(bad.coefnames)[:sd][1]="wrong-group: (Intercept)"
    @test_throws ArgumentError DRModels._marginal_simulator(bad,dat;tree=q.tree)
    for block in (:mu,:sigma)
        wrong=deepcopy(q.fit); reverse!(Dict(wrong.coefnames)[block])
        @test_throws ArgumentError DRModels._marginal_simulator(wrong,dat;tree=q.tree)
    end
    short=map(v -> v[2:end],dat) # Remove an observed row, not the wholly masked last tip.
    @test count(!ismissing,short.y) != q.fit.nobs
    @test_throws ArgumentError DRModels._marginal_simulator(q.fit,short;tree=q.tree)
    for index in (1,3,5,8)
        wrong=deepcopy(q.fit); wrong.theta[index]=Inf
        @test_throws ArgumentError DRModels._marginal_simulator(wrong,dat;tree=q.tree)
    end
    labels=copy(dat.species);labels[findfirst(ismissing,missing_y)]="unknown"
    @test_throws ArgumentError DRModels._marginal_simulator(q.fit,merge(dat,(;species=labels));tree=q.tree)
    # study has a scalar random intercept, but no explicit sd(study) submodel.
    f=bf(@formula(y ~ x + (1 | species) + (1 | study) + phylo(1 | species)),
         @formula(sigma ~ x), @formula(sd(species) ~ z),
         @formula(sd(species, phylogenetic) ~ z))
    scalar_fit=DRModels._withformula(q.fit,f)
    simscalar=DRModels._marginal_simulator(scalar_fit,dat;tree=q.tree)
    draw=simscalar(MersenneTwister(19))
    @test draw[q.observed] ≈ expected[q.observed] atol=1e-12 rtol=1e-12
end

@testset "Formula and fitted LSS bootstrap use the same marginal model" begin
    q=_boot_lss_fixture(with_phylo=false,iid=[:species,:study])
    fit=drm(q.fit.formula,Gaussian();data=q.data)
    a=bootstrap_result(fit;data=q.data,B=3,rng=MersenneTwister(910))
    b=bootstrap_result(q.fit.formula,Gaussian();data=q.data,B=3,rng=MersenneTwister(910))
    @test a.seeds==b.seeds
    @test a.summary==b.summary
    @test a.failed==b.failed==0
end

end
