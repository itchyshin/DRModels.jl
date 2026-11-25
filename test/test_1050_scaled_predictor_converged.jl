# #1050 (remaining routes): `converged` is the unit-free Newton step in SE units
# (plus the iteration-cap check), not `Optim.converged && Optim.g_converged`. The
# absolute `g_tol = 1e-8` is not unit-free: the same model with the predictor
# multiplied by 1e3 or 1e4 stalls just above it while the Newton step is a
# negligible fraction of a standard error. Each case below is a replicate that
# read `converged = false` on main (Julia 1.10.12, Totoro) and is the same model
# at the optimum. Only the flag is asserted; the exact |g|inf is CPU dependent.
using DRModels
using Test, Random
import Distributions

function _rtweedie_1050(μ, φ, p)
    λ = μ^(2 - p) / (φ * (2 - p)); γ = φ * (p - 1) * μ^(p - 1); sh = (2 - p) / (p - 1)
    N = rand(Distributions.Poisson(λ))
    N == 0 ? 0.0 : rand(Distributions.Gamma(N * sh, γ))
end

# One replicate: the same draws for every family, as in the Totoro probe.
function _gen_1050_scaled(rep)
    Random.seed!(1050000 + rep)
    n = 150; x0 = randn(n); G = 15; g = repeat(1:G, inner = 10)
    b = 0.5 .* randn(G)
    yln = exp.(0.5 .+ 0.4 .* x0 .+ b[g] .+ 0.3 .* randn(n))
    ytw = [_rtweedie_1050(exp(0.5 + 0.3 * x0[i] + b[g][i]), 2.0, 1.5) for i in 1:n]
    yc = Float64.(clamp.(round.(Int, 2.5 .+ 0.8 .* x0 .+ b[g] .+ randn(n)), 1, 4))
    yt = 1 .+ 0.5 .* x0 .+ b[g] .+ 0.7 .* randn(n) ./ sqrt.(sum(abs2, randn(n, 4); dims = 2) ./ 4)[:]
    return (; n, x0, G, g, b, yln, ytw, yc, yt)
end

@testset "#1050 scaled-predictor replicates read converged" begin
    cases = [
        (:student_re, 4, 1e4), (:student_re, 16, 1e4), (:student_re, 25, 1e3),
        (:student_re, 33, 1e3), (:student_re, 40, 1e3), (:student_re, 51, 1e3),
        (:cumulative_re, 6, 1e3), (:cumulative_re, 18, 1e3), (:cumulative_re, 42, 1e4),
        (:cumulative_re, 49, 1e3), (:tweedie_re, 11, 1e3), (:tweedie_re, 14, 1.0),
        (:tweedie_re, 24, 1e3), (:tweedie_re, 30, 1e3), (:tweedie_re, 31, 1e3),
        (:tweedie_re, 55, 1e4), (:lognormal_re, 30, 1e4),
    ]
    for (fam, rep, sc) in cases
        d = _gen_1050_scaled(rep)
        x = d.x0 .* sc; g = d.g
        fit = if fam === :student_re
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)),
                Student(); data = (; y = d.yt, x, g))
        elseif fam === :cumulative_re
            drm(bf(@formula(y ~ x + (1 | g))), CumulativeLogit(); data = (; y = d.yc, x, g))
        elseif fam === :tweedie_re
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1), @formula(nu ~ 1)),
                Tweedie(); data = (; y = d.ytw, x, g))
        elseif fam === :lognormal_re
            drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), LogNormal();
                data = (; y = d.yln, x, g))
        end
        @test fit.converged
    end
end
