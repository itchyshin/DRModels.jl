# #908 (owner decision 17): the ordinary Binomial `(1 | g)` default is 5-node adaptive
# Gauss-Hermite (was 3). Evidence: MASS::bacteria (binary, 50 subjects, sparse), where the
# 3-node default left sd_mu 5.1% low and 0.22 nat below the 61-node reference.
# Measured at fit optima (Julia 1.10, Totoro):
#                 sd_mu     logLik       gap to K=61
#   K = 3        1.28663   -101.037224   -0.217 nat, sd -5.1%
#   K = 5        1.34754   -100.834796   -0.0143 nat, sd -0.64%
#   K = 61       1.35624   -100.820515   reference
using DRModels
using Test

function _read_bacteria()
    lines = readlines(joinpath(@__DIR__, "fixtures", "binomial_default_nodes", "bacteria.csv"))
    rows = [split(replace(l, "\"" => ""), ',') for l in lines[2:end]]
    return (; y = parse.(Float64, [r[1] for r in rows]),
              week = parse.(Float64, [r[2] for r in rows]),
              g = String[r[3] for r in rows])
end

@testset "Binomial (1|g) default nodes = 5 (#908)" begin
    @test DRModels._BINOMIAL_RANEF_AGHQ_K == 5

    d = _read_bacteria()
    f = bf(@formula(y ~ 1 + week + (1 | g)))
    s = d.y; ntr = ones(length(s))
    fixed_mu, re, _, _ = DRModels._split_ranef(Dict(f.forms)[:mu])
    gidx, G = DRModels._group_index(getproperty(d, re[1][2]))
    _, Xμ, nmμ = DRModels._design(f.response, fixed_mu, d)
    kfit(K) = DRModels._fit_binomial_ranef(DRModels.Binomial(), s, ntr, Xμ, gidx, G, nmμ, :g, 1e-8; nq = K)

    ref = kfit(61)
    fit = drm(f, DRModels.Binomial(); data = d, se = false)
    @test fit.converged
    # the public default is the 5-node fit, bit for bit
    @test fit.theta == kfit(5).theta

    sd(ft) = exp(ft.theta[end])
    # default within a stated tolerance of the 61-node reference (measured 0.64% sd, 0.0143 nat)
    @test abs(sd(fit) - sd(ref)) / sd(ref) < 0.01
    @test abs(loglik(fit) - loglik(ref)) < 0.02
    @test maximum(abs.(fit.theta[1:end-1] .- ref.theta[1:end-1]) ./ abs.(ref.theta[1:end-1])) < 5e-3

    # ... and strictly closer than the old 3-node default, which was ~5% low on sd_mu
    old = kfit(3)
    @test abs(sd(old) - sd(ref)) / sd(ref) > 0.04
    @test abs(loglik(old) - loglik(ref)) > 0.2
    @test abs(sd(fit) - sd(ref)) < abs(sd(old) - sd(ref)) / 5

    # an explicit node count is honoured: nq = 3 reproduces the old fit, nq = 5 the default
    @test kfit(3).theta != fit.theta
    @test kfit(5).theta == fit.theta
end
