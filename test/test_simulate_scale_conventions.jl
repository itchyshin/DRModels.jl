using DRModels, Test, Random, LinearAlgebra, SparseArrays
import Distributions

function _scale_draw_fixture(fam; coupled=false, extra=Dict{Symbol,Vector{Float64}}())
    n=32
    mu=collect(range(1.4,3.8;length=n))
    sigma=collect(range(0.35,0.8;length=n))
    fam isa DRModels.Gamma && coupled && (sigma=collect(range(2.0,5.0;length=n)))
    scales=merge(Dict(:sigma=>sigma),extra)
    fit=DRModels.DrmFit(fam,[:mu=>1:1,:sigma=>2:2],[ :mu=>["Intercept"],:sigma=>["Intercept"]],
        zeros(2),Matrix{Float64}(I,2,2),-1.,n,true,Dict(:mu=>mu),Dict(:mu=>ones(n)),scales)
    if coupled
        kind=fam isa DRModels.Gamma ? Val(:gamma) : Val(:nb2)
        fit=DRModels._withnll(fit,DRModels.LocScaleObjective(kind,ones(n),ones(n,1),ones(n,1),ones(Int,n),1,sparse(1.0I,1,1);whitened=true))
    end
    return fit
end

function _positive_nb_reference(rng, size, prob)
    d=Distributions.NegativeBinomial(size,prob)
    y=rand(rng,d)
    while y==0
        y=rand(rng,d)
    end
    return Float64(y)
end

@testset "simulation uses fitted sigma conventions" begin
    for mode in (:plain,:zi,:hu,:truncated)
        fam=mode===:truncated ? DRModels.TruncatedNegBinomial2() : DRModels.NegBinomial2()
        extra=mode===:zi ? Dict(:zi=>fill(0.2,32)) : mode===:hu ? Dict(:hu=>fill(0.2,32)) : Dict{Symbol,Vector{Float64}}()
        fit=_scale_draw_fixture(fam;extra)
        rr=MersenneTwister(9182)
        expected=map(1:fit.nobs) do i
            r=fit.scales[:sigma][i]^-2; mu=fit.means[:mu][i]; prob=r/(r+mu)
            if mode===:zi
                rand(rr)<0.2 ? 0.0 : rand(rr,Distributions.NegativeBinomial(r,prob))
            elseif mode===:hu
                rand(rr)<0.2 ? 0.0 : _positive_nb_reference(rr,r,prob)
            elseif mode===:truncated
                _positive_nb_reference(rr,r,prob)
            else
                rand(rr,Distributions.NegativeBinomial(r,prob))
            end
        end
        @test simulate(fit;rng=MersenneTwister(9182))==expected
    end
    for coupled in (false,true)
        fit=_scale_draw_fixture(DRModels.Gamma();coupled)
        rr=MersenneTwister(4419)
        expected=map(1:fit.nobs) do i
            shape=coupled ? fit.scales[:sigma][i] : fit.scales[:sigma][i]^-2
            rand(rr,Distributions.Gamma(shape,fit.means[:mu][i]/shape))
        end
        @test simulate(fit;rng=MersenneTwister(4419))≈expected rtol=1e-14
    end
end

@testset "bootstrap auxiliary override is per-draw and does not mutate fit" begin
    for fam in (DRModels.NegBinomial2(),DRModels.Gamma(),DRModels.Beta(),DRModels.BetaBinomial())
        fit=_scale_draw_fixture(fam;coupled=fam isa DRModels.Gamma,
            extra=fam isa DRModels.BetaBinomial ? Dict(:trials=>fill(12.,32)) : Dict{Symbol,Vector{Float64}}())
        mu=fam isa Union{DRModels.Beta,DRModels.BetaBinomial} ? collect(range(.15,.75;length=32)) : fill(2.3,32)
        sigma=fill(.6,32); before=deepcopy(fit.scales); rr=MersenneTwister(337)
        expected=map(1:32) do i
            if fam isa DRModels.Gamma
                rand(rr,Distributions.Gamma(sigma[i],mu[i]/sigma[i]))
            elseif fam isa DRModels.NegBinomial2
                size=sigma[i]^-2
                rand(rr,Distributions.NegativeBinomial(size,size/(size+mu[i])))
            else
                precision=sigma[i]^-2
                d=fam isa DRModels.Beta ? Distributions.Beta(mu[i]*precision,(1-mu[i])*precision) :
                    Distributions.BetaBinomial(12,mu[i]*precision,(1-mu[i])*precision)
                rand(rr,d)
            end
        end
        @test DRModels._simulate_once(fit,MersenneTwister(337);mu,sigma)≈expected rtol=1e-13
        @test fit.scales==before
    end
end
