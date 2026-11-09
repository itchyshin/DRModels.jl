using DRModels, Test
import DelimitedFiles

# Binomial independent random effects with a covariate in the SD formulas
# (`sd(g, coef(.)) ~ z`, group G12 of the wave-8 R-vs-Julia disagreement map).
# From the single default start L-BFGS stopped at a smooth, converged interior point
# with every SD well above the boundary, 1.3 to 6 nats below a point the same objective
# scores higher; the boundary trigger of `_q2_boundary_restarts` never fires there.
# These fits now restart unconditionally (accepted only on a clear objective gain) and
# record the outcome in `fit.scales[:q2_restarts]` = [ran, winner (0 = default start),
# gain, converged flipped]. Committed data: test/fixtures/binomial_indep_restarts/
# (`g12_binomial.csv`: the wave-8 fuzz data; `seed29_binomial.csv`: seed 29 of the
# committed generator `gen_binomial_indep.jl`). Reference: drmTMB 0.8.1 ML on g12_binomial.csv,
# both cells logLik -253.5291 (is_converged() = FALSE); Julia's default start gave
# -255.2241 (q=2) and -254.5596 (q=3).

const _BIR_DIR = joinpath(@__DIR__, "fixtures", "binomial_indep_restarts")

function _bir_data(name)
    m, h = DelimitedFiles.readdlm(joinpath(_BIR_DIR, name), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

_bir_formula(q; z = true) = begin
    rhs = z ? (@formula(sd(g, coef(intercept)) ~ z), @formula(sd(g, coef(x1)) ~ z), @formula(sd(g, coef(x2)) ~ z)) :
              (@formula(sd(g, coef(intercept)) ~ 1), @formula(sd(g, coef(x1)) ~ 1), @formula(sd(g, coef(x2)) ~ 1))
    q == 2 ? bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), rhs[1], rhs[2]) :
             bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g) + (0 + x2 | g)), rhs...)
end

@testset "G12 cells: restarts lift Julia's optimum above its first start and above drmTMB's reported value" begin
    d = _bir_data("g12_binomial.csv")
    llR = -253.5291
    for (q, ll_default) in ((2, -255.2241433762659), (3, -254.5596))
        fit = drm(_bir_formula(q), Binomial(:logit); data = d, se = false)
        @test fit.converged
        # The two programs do not score one point identically (R's own estimate is -253.5291 in R and
        # -253.2382 in Julia, a gap of 0.29), so this compares each program's objective at its own optimum.
        # It supports 'Julia's optimum is not below R's reported value', not 'Julia reaches R's value'.
        @test loglikelihood(fit) > llR
        @test loglikelihood(fit) > ll_default + 1.3          # was 1.3 to 2.0 nats short
        rs = fit.scales[:q2_restarts]
        @test rs[1] == 1.0 && rs[2] >= 1.0
        @test rs[3] ≈ loglikelihood(fit) - ll_default atol = 5e-3
    end
    # Julia's objective at R's q=2 estimate is -253.2382, while R reports -253.5291 at the same estimate:
    # the objectives differ by 0.29 nats (cause not established; R Laplace against Julia AGHQ is a candidate).
    # Only 0.09 of the 0.38 margin of Julia's optimum over R's reported value is gain under one objective.
    thR = [0.472920237952, 0.217143847112, 0.23791642793, -0.273548232169,
           -54.264625208, 56.949833433, -0.972683864797, 0.149484497541]
    fit2 = drm(_bir_formula(2), Binomial(:logit); data = d, se = false)
    @test -fit2.nll(thR) ≈ -253.2381590624687 atol = 1e-6
    @test loglikelihood(fit2) >= -fit2.nll(thR)
end

# Acceptance rule (review of the wave-10 restarts): an UNCONVERGED restart must not replace a CONVERGED first fit
# unless it gains more than `_BINOMIAL_INDEP_BIG_GAIN` (1 nat). Generated seed 11 (q = 2) is the committed case
# on Julia 1.10: the first fit is converged at -254.6933 and a restart reaches -254.5145 (+0.18) but does not
# converge; with the generic 1e-3 rule that fit came back unconverged (converged = false, Hessian at a
# non-stationary point). Whether that restart converges depends on the Julia version (on 1.12.6 it does, and is
# then accepted legitimately), so the integration test states the invariant, and the deterministic test below
# exercises the rule itself on every version. The generator is the committed splitmix64 + Box-Muller one.
module _BIRGen
include(joinpath(@__DIR__, "fixtures", "binomial_indep_restarts", "gen_binomial_indep.jl"))
end
@testset "a restart does not turn a converged fit into an unconverged one (seed 11)" begin
    d = _BIRGen.gen_binomial_indep(11)
    fit = drm(_bir_formula(2), Binomial(:logit); data = d, se = false)
    rs = fit.scales[:q2_restarts]
    @test fit.converged
    @test !(rs[4] == 1.0 && !fit.converged)                      # no converged -> unconverged flip
    @test loglikelihood(fit) >= -254.69330873885656 - 1e-6       # never worse than the first optimum
    @test DRModels._BINOMIAL_INDEP_BIG_GAIN >= 1.0
end

@testset "_q2_boundary_restarts: unconverged alternative vs converged incumbent" begin
    Optim = DRModels.Optim
    mkres(v) = Optim.optimize(x -> (x[1] - 1)^2 + v, [0.0], Optim.LBFGS())
    ok_set = Set{Float64}()
    function run(gain_first_alt; big_gain)
        # first run: converged, objective 5.0; first alternative: unconverged, objective 5.0 - gain; the rest are worse
        calls = Ref(0)
        r0 = mkres(5.0); r1 = mkres(5.0 - gain_first_alt); rw = mkres(9.0)
        empty!(ok_set); push!(ok_set, Optim.minimum(r0), Optim.minimum(rw))
        run_from(_) = (calls[] += 1; calls[] == 1 ? r1 : rw)
        info = Ref{Any}()
        out = DRModels._q2_boundary_restarts(run_from, Float64[], r0, (a, b, c) -> nothing, th -> 1.0,
                                             r -> Optim.minimum(r) in ok_set; info = info, force = true,
                                             big_gain = big_gain)
        return out === r0, info[]
    end
    # generic 1e-3 rule: a 0.18 gain replaces the converged incumbent (the behaviour the review found)
    kept, info = run(0.18; big_gain = 1e-3)
    @test !kept && info.flipped
    # route rule: 1 nat needed, so the converged incumbent stands and nothing is recorded as a gain
    kept, info = run(0.18; big_gain = DRModels._BINOMIAL_INDEP_BIG_GAIN)
    @test kept && !info.flipped && info.gain == 0.0 && info.winner == 0
    kept, info = run(0.98; big_gain = DRModels._BINOMIAL_INDEP_BIG_GAIN)
    @test kept
    # a large documented gain (above 1 nat) still wins, flagged as a convergence flip
    kept, info = run(2.0; big_gain = DRModels._BINOMIAL_INDEP_BIG_GAIN)
    @test !kept && info.flipped
    @test info.gain ≈ 2.0 atol = 1e-6
end
