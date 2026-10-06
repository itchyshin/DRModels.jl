# Misspecified cor12: the analyst GUESSES the sampling correlation. How much does
# a wrong guess cost the heterogeneity correlation?
using DRModels, Random, LinearAlgebra, Statistics, Printf
const BF = bf(mu1=@formula(y1 ~ x), mu2=@formula(y2 ~ x),
              sigma1=@formula(sigma1 ~ 1), sigma2=@formula(sigma2 ~ 1),
              rho12=@formula(rho12 ~ 1))
function gen(seed; n, rho_het, scor_true, s1=0.45, s2=0.55)
    rng = MersenneTwister(seed); x = randn(rng,n)
    mu1 = 0.2 .+0.5 .*x; mu2 = -0.1 .-0.35 .*x
    v1 = 0.01 .+0.03 .*rand(rng,n); v2 = 0.01 .+0.04 .*rand(rng,n)
    y1=similar(x); y2=similar(x)
    for i in 1:n
        c12 = scor_true*sqrt(v1[i]*v2[i]) + rho_het*s1*s2
        S=[v1[i]+s1^2 c12; c12 v2[i]+s2^2]
        z=cholesky(Symmetric(S)).L*randn(rng,2)
        y1[i]=mu1[i]+z[1]; y2[i]=mu2[i]+z[2]
    end
    ((; x, y1, y2), v1, v2)
end
n=150; rho_het=-0.35; scor_true=0.6; nrep=40
println("TRUE sampling cor = $scor_true, TRUE heterogeneity rho = $rho_het, n=$n, $nrep reps")
println(rpad("assumed cor12",16), rpad("mean rho12",12), rpad("bias",11), "mcse")
for assumed in (0.0, 0.3, 0.6, 0.9)
    R=Float64[]
    for r in 1:nrep
        d, v1, v2 = gen(8000+r; n=n, rho_het=rho_het, scor_true=scor_true)
        V = meta_vcov_bivariate(v1, v2; cor12 = assumed)
        f = drm(BF, Gaussian(); data=d, V=V)
        push!(R, f.scales[:rho12][1])
    end
    @printf("%-16s %-12.4f %+-11.4f %.4f%s\n", assumed, mean(R), mean(R)-rho_het,
            std(R)/sqrt(length(R)), assumed==scor_true ? "   <- correct" : "")
end
