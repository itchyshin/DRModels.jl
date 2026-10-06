# DRModels.jl#496: does the scale-axis phylo-variance bias shrink with N, or persist?
#
# DESIGN NOTE (the trap this avoids). Cell B's committed shape uses ntip=16 with
# branch_length=0.25, giving a balanced tree of height EXACTLY 1.0 — which is why
# its truth can be LAM_B itself, with raw and correlation scale coinciding. Naively
# raising ntip at fixed branch_length raises the height (1.25 at 32, 1.5 at 64) and
# would rescale Sigma_a, confounding the bias with a scale error. So branch_length
# is set to 1/log2(ntip) at every rung, holding height at 1.0 and isolating N.
#
# Fits only: the question is about the POINT ESTIMATE, so no intervals are computed.
using DRModels, Random, LinearAlgebra, Printf, DelimitedFiles

const LAM = Symmetric([0.25 0.10 0.05 0.00
                       0.10 0.25 0.00 0.04
                       0.05 0.00 0.16 0.02
                       0.00 0.04 0.02 0.16])
const B_MU1 = (b0=0.5, b1=0.3); const B_MU2 = (b0=-0.2, b1=0.4)
const LS1 = -0.6; const LS2 = -0.6; const PER = 8

function sim_fit(ntip::Int, seed::Int)
    bl = 1 / log2(ntip)                       # height == 1.0 at every rung
    rng = MersenneTwister(seed)
    phy = DRModels.random_balanced_tree(ntip; branch_length = bl)
    Kraw = DRModels.sigma_phy_dense(phy)
    @assert isapprox(Kraw[1,1], 1.0; atol=1e-10) "height != 1 at ntip=$ntip"
    U = cholesky(Symmetric(Kraw)).L * randn(rng, ntip, 4) * transpose(cholesky(LAM).L)
    tip = repeat(1:ntip, inner=PER); n = ntip*PER
    x = randn(rng, n)
    y1 = B_MU1.b0 .+ B_MU1.b1.*x .+ U[tip,1] .+ exp.(LS1 .+ U[tip,3]).*randn(rng,n)
    y2 = B_MU2.b0 .+ B_MU2.b1.*x .+ U[tip,2] .+ exp.(LS2 .+ U[tip,4]).*randn(rng,n)
    dat = (; y1, y2, x, species = [phy.leaf_names[k] for k in tip])
    form = bf(mu1=@formula(y1 ~ x + phylo(1|species)), mu2=@formula(y2 ~ x + phylo(1|species)),
              sigma1=@formula(sigma1 ~ 1 + phylo(1|species)), sigma2=@formula(sigma2 ~ 1 + phylo(1|species)),
              rho12=@formula(rho12 ~ 1))
    drm(form, Gaussian(); data=dat, tree=phy, method=:REML, q4_vcov=false)
end

function main()
    ntip  = parse(Int, ARGS[findfirst(==("--ntip"), ARGS)+1])
    s0    = parse(Int, ARGS[findfirst(==("--seed-start"), ARGS)+1])
    cnt   = parse(Int, ARGS[findfirst(==("--seed-count"), ARGS)+1])
    out   = ARGS[findfirst(==("--out"), ARGS)+1]
    truth = DRModels.cov_to_lc(Matrix(LAM))
    open(out, "w") do io
        println(io, "seed\tntip\tconv\tentry\testimate\ttruth")
        for s in s0:(s0+cnt-1)
            local fit
            try; fit = sim_fit(ntip, s); catch; continue; end
            ci = findfirst(pr -> pr[1] === :phylocov, fit.coefnames)
            bi = findfirst(pr -> pr[1] === :phylocov, fit.blocks)
            (ci === nothing || bi === nothing) && continue
            nms  = fit.coefnames[ci][2]
            rng_ = fit.blocks[bi][2]
            for (j,k) in enumerate(rng_)
                println(io, "$s\t$ntip\t$(Int(fit.converged))\t$(nms[j])\t$(fit.theta[k])\t$(truth[j])")
            end
        end
    end
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
