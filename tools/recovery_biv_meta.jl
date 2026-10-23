# ADEMP recovery for the bivariate known-V meta path.
# The claim under test is the one the model exists to make: are the HETEROGENEITY
# components recovered, and does rho12 stay separated from the KNOWN sampling
# correlation baked into V?
using DRModels, Random, LinearAlgebra, Statistics, Printf

const BF = bf(mu1=@formula(y1 ~ x), mu2=@formula(y2 ~ x),
              sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
              rho12=@formula(rho12 ~ 1))

function one_rep(seed; n, rho_het, scor, s1, s2)
    rng = MersenneTwister(seed)
    x = randn(rng, n)
    mu1 = 0.2 .+ 0.5 .* x; mu2 = -0.1 .- 0.35 .* x
    v1 = 0.01 .+ 0.03 .* rand(rng, n); v2 = 0.01 .+ 0.04 .* rand(rng, n)
    y1 = similar(x); y2 = similar(x)
    for i in 1:n
        c12 = scor*sqrt(v1[i]*v2[i]) + rho_het*s1*s2
        S = [v1[i]+s1^2  c12; c12  v2[i]+s2^2]
        z = cholesky(Symmetric(S)).L * randn(rng, 2)
        y1[i] = mu1[i]+z[1]; y2[i] = mu2[i]+z[2]
    end
    V = meta_vcov_bivariate(v1, v2; cor12 = scor)
    d = (; x, y1, y2)
    fit  = drm(BF, Gaussian(); data=d, V=V)          # V consumed
    novi = drm(BF, Gaussian(); data=d)               # V IGNORED (the control)
    (fit.scales[:sigma1][1], fit.scales[:sigma2][1], fit.scales[:rho12][1],
     coef(fit,:mu1)[2], novi.scales[:rho12][1], is_converged(fit))
end

function study(; n, rho_het, scor, s1=0.45, s2=0.55, nrep=50)
    S1=Float64[]; S2=Float64[]; R=Float64[]; B=Float64[]; Rn=Float64[]; conv=Ref(0)
    for r in 1:nrep
        a,b,c,dd,e,ok = one_rep(7000+r; n=n, rho_het=rho_het, scor=scor, s1=s1, s2=s2)
        ok && (conv[] += 1)
        push!(S1,a); push!(S2,b); push!(R,c); push!(B,dd); push!(Rn,e)
    end
    rep(v,t)= @sprintf("%7.4f  bias %+8.4f  mcse %.4f", mean(v), mean(v)-t, std(v)/sqrt(length(v)))
    println("--- n=$n  rho_het=$rho_het  sampling_cor=$scor   converged $(conv[])/$nrep")
    println("    sigma1  ", rep(S1,s1))
    println("    sigma2  ", rep(S2,s2))
    println("    rho12   ", rep(R,rho_het))
    println("    beta_mu1 slope ", rep(B,0.5))
    println("    rho12 WITHOUT V (contaminated control): ", @sprintf("%7.4f", mean(Rn)),
            "   pulled by ", @sprintf("%+.4f", mean(Rn)-rho_het))
end
study(n=120, rho_het=-0.35, scor=0.6)
study(n=120, rho_het=0.25,  scor=-0.4)
study(n=300, rho_het=-0.35, scor=0.6)
