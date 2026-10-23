using DRModels, Test, Random

@testset "Threaded bootstrap keeps every successful replicate" begin
    # No fitting: every refit succeeds and returns a fixed valid result. This
    # isolates bookkeeping from optimizer convergence and simulation variability.
    f=bf(@formula(y ~ 1))
    dat=(;y=[1.,2.,3.,4.])
    fit=DRModels._withformula(DRModels.DrmFit(Gaussian(),[:mu=>1:1,:sigma=>2:2],
        [:mu=>["(Intercept)"],:sigma=>["(Intercept)"]],[2.5,0.],
        [1. 0.;0. 1.],-5.,4,true,Dict(:mu=>fill(2.5,4)),
        Dict(:mu=>copy(dat.y)),Dict(:sigma=>ones(4))),f)
    B=250 # Deliberately not aligned to BitVector machine-word boundaries.
    counts=[DRModels._bootstrap_result(fit,f,dat,B,.95,MersenneTwister(k),true,
        _->fit;simulate_fn=_->copy(dat.y)).used for k in 1:300]
    @test all(==(B),counts)
    @test Threads.nthreads()==1 || minimum(counts)==B
    println("BOOTSTRAP_FLAG_COUNTS min=",minimum(counts)," max=",maximum(counts)," batches=",length(counts)," B=",B)
end
