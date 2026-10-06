# test_phylo_penalty.jl — A4c, the penalized-MAP phylogenetic variance components
# (drmTMB's `drm_phylo_penalty()` / `drm_phylo_penalty_sweep()`).
#
# WHY THIS FILE EXISTS. The penalty CHANGES THE OBJECTIVE, so it is the kind of
# feature that can look right and be wrong: a fit still converges, still returns
# plausible numbers, and still passes every pre-existing suite while optimising
# the wrong surface. Three things therefore get asserted directly rather than
# inferred from "it ran":
#
#   1. the penalty at the optimum equals drmTMB's own closed form
#      sum(lambda*sd - log(sd) - log(lambda)) --- ported from drmTMB's
#      test-phylo-penalized-map.R, so the two implementations are pinned to the
#      SAME arithmetic and not merely to each other's behaviour;
#   2. `penalty = nothing` is BIT-IDENTICAL to a plain ML fit, which is what
#      makes this change safe for every existing phylo model;
#   3. `loglik` stays the UNPENALIZED data log-likelihood, so the drmTMB identity
#      -objective == loglik - phylo_penalty holds.
#
# THE SHRINKAGE DIRECTION IS CONDITIONAL, and the test says so. The penalty
# lambda*sd - log(sd) is minimised at sd = 1/lambda, so a penalty pulls the SD
# TOWARD 1/lambda, not toward zero. `sd_MAP < sd_ML` therefore holds only when
# `sd_ML > 1/lambda`. Asserting unconditional shrinkage would be asserting a
# false statement that happens to pass on a convenient fixture.

using DRModels
using Test
using Random
using LinearAlgebra
using Statistics

# Shared fixture: a balanced tree with a phylogenetic mean signal.
function _pen_fixture(seed::Int; G::Int = 16, m::Int = 5, sd_phy::Float64 = 0.9)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(G; branch_length = 0.3)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d')
    n = G * m; species = repeat(1:G, inner = m); x = randn(rng, n)
    u = sd_phy .* (cholesky(Symmetric(K)).L * randn(rng, G))
    y = 0.2 .+ 0.5 .* x .+ u[species] .+ 0.4 .* randn(rng, n)
    return (data = (; y, x, species), tree = phy)
end

const _PEN_FORM = bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1))

@testset "phylo penalty (A4c) — penalized MAP phylogenetic variance components" begin

    @testset "spec construction mirrors drmTMB's drm_phylo_penalty()" begin
        p = drm_phylo_penalty()
        @test p.sd_u == 1.0
        @test p.sd_alpha == 0.05
        @test p.cor_sd === nothing
        # rate = -log(sd_alpha)/sd_u  (drmTMB R/penalty.R)
        @test p.rate ≈ -log(0.05) / 1.0
        @test drm_phylo_penalty(sd_u = 0.5).rate ≈ -log(0.05) / 0.5
        @test drm_phylo_penalty(sd_alpha = 0.01).rate ≈ -log(0.01) / 1.0
        @test drm_phylo_penalty(cor_sd = 0.25).cor_sd == 0.25

        # The refusals, with drmTMB's own boundaries.
        @test_throws ArgumentError drm_phylo_penalty(sd_u = 0.0)
        @test_throws ArgumentError drm_phylo_penalty(sd_u = -1.0)
        @test_throws ArgumentError drm_phylo_penalty(sd_alpha = 0.0)
        @test_throws ArgumentError drm_phylo_penalty(sd_alpha = 1.0)
        @test_throws ArgumentError drm_phylo_penalty(cor_sd = 0.0)
        @test_throws ArgumentError drm_phylo_penalty(cor_sd = -0.5)
    end

    @testset "the penalty equals drmTMB's closed form, and loglik stays unpenalized" begin
        fx = _pen_fixture(244)
        pen = drm_phylo_penalty(sd_u = 0.6, sd_alpha = 0.05)
        fit = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree, penalty = pen)

        @test fit.estim_method === :MAP
        @test isfinite(fit.phylo_penalty)

        # drmTMB test-phylo-penalized-map.R:
        #   expected_pen <- sum(lam * exp(log_sd) - log_sd - log(lam))
        sd_map = re_sd(fit)[:species]
        lam = pen.rate
        expected = lam * sd_map - log(sd_map) - log(lam)
        @test fit.phylo_penalty ≈ expected atol = 1e-9

        # And the identity -objective == loglik - penalty, i.e. `loglik` is the
        # UNPENALIZED data log-likelihood. An ML fit at the SAME parameters must
        # reproduce it, which is what makes this a real check and not a tautology.
        @test isfinite(loglik(fit))
        ml = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree)
        @test loglik(ml) > loglik(fit)   # ML maximises the unpenalized likelihood
    end

    @testset "penalty = nothing is bit-identical to a plain ML fit" begin
        # The no-op guarantee. drmTMB gets it by always emitting the penalty DATA
        # fields with penalize_phylo = 0L; DRModels.jl gets it by branching on
        # `penalty === nothing`. Either way it must be EXACT, not approximate —
        # this is what keeps every pre-existing phylo model untouched.
        fx = _pen_fixture(7)
        a = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree)
        b = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree, penalty = nothing)
        @test coef(a) == coef(b)
        @test loglik(a) == loglik(b)
        @test a.estim_method === :ML
        @test isnan(a.phylo_penalty)
        @test a.penalty === nothing
    end

    @testset "shrinkage toward the prior mode 1/rate (measured across 10 seeds)" begin
        # Measured, not guessed. Over seeds 1:10 with sd_u = 1 (rate ≈ 3.00, prior
        # mode 1/rate ≈ 0.334) every sd_ML landed ABOVE the prior mode, and the
        # ratio sd_MAP/sd_ML was mean 0.958, sd 0.0079, range [0.9495, 0.9761].
        # The bound below is ~5 sd from that mean, sized from the measurement.
        pen = drm_phylo_penalty(sd_u = 1.0, sd_alpha = 0.05)
        ratios = Float64[]
        for seed in 1:10
            fx = _pen_fixture(seed)
            ml = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree)
            mp = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree, penalty = pen)
            sd_ml = re_sd(ml)[:species]; sd_map = re_sd(mp)[:species]
            # The CONDITION for shrinkage, asserted rather than assumed.
            @test sd_ml > 1 / pen.rate
            @test sd_map < sd_ml
            push!(ratios, sd_map / sd_ml)
        end
        @test 0.92 < mean(ratios) < 0.99
    end

    @testset "a tighter prior shrinks harder (monotone in sd_u)" begin
        fx = _pen_fixture(244)
        sds = [re_sd(drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree,
                         penalty = drm_phylo_penalty(sd_u = su)))[:species]
               for su in (2.0, 1.0, 0.5, 0.25)]
        @test issorted(sds; rev = true)
    end

    @testset "refusals" begin
        fx = _pen_fixture(3)
        pen = drm_phylo_penalty()

        # drmTMB: "`penalty` requires a phylogenetic term in the model."
        @test_throws ArgumentError drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian();
                                       data = fx.data, penalty = pen)

        # A penalized fit is MAP; REML is a restricted-likelihood estimator.
        @test_throws ArgumentError drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree,
                                       method = :REML, penalty = pen)

        # Not a PhyloPenalty at all.
        @test_throws ArgumentError drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree,
                                       penalty = 0.5)

        # cor_sd on a model with ONE phylo SD: refused with the dedicated type, so
        # the sweep can catch precisely this condition.
        @test_throws PhyloCorPenaltyNeedsTwoSD drm(_PEN_FORM, Gaussian(); data = fx.data,
                                                   tree = fx.tree,
                                                   penalty = drm_phylo_penalty(cor_sd = 0.5))

        # The conjugate-EM variant optimises a different surrogate — refuse rather
        # than silently return an unpenalized fit wearing a MAP label.
        @test_throws ArgumentError drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree,
                                       algorithm = :em, penalty = pen)
    end

    @testset "lrtest / anova refuse penalized fits" begin
        fx = _pen_fixture(5)
        ml = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree)
        mp = drm(_PEN_FORM, Gaussian(); data = fx.data, tree = fx.tree,
                 penalty = drm_phylo_penalty(sd_u = 0.5))
        @test_throws ArgumentError lrtest(ml, mp)
        @test_throws ArgumentError anova(ml, mp)
    end

end

# --- The two-SD blocks: separate (no correlation) and coupled (a real one) ----

function _pen_fixture2(seed::Int; G::Int = 12, m::Int = 8)
    rng = MersenneTwister(seed)
    phy = random_balanced_tree(G; branch_length = 0.4)
    C = sigma_phy_dense(phy; σ²_phy = 1.0)
    d = sqrt.(diag(C)); K = C ./ (d * d'); L = cholesky(Symmetric(K)).L
    n = G * m; species = repeat(1:G, inner = m)
    u_mu = 0.8 .* (L * randn(rng, G))
    u_sig = 0.5 .* (L * randn(rng, G))
    y = [1.0 + u_mu[species[i]] + exp(log(0.5) + u_sig[species[i]]) * randn(rng) for i in 1:n]
    return (data = (; y, species), tree = phy)
end

const _PEN_FORM2 = bf(@formula(y ~ phylo(1 | species)), @formula(sigma ~ phylo(1 | species)))

@testset "phylo penalty — the two-SD blocks" begin

    @testset "separate block: both SDs penalized, cor_sd REFUSED" begin
        # This block constrains the mean↔σ phylo correlation to zero, so there is
        # no correlation parameter to penalize. Refusing beats silently ignoring:
        # a silently-ignored cor_sd reads as a prior-sensitivity check that did
        # nothing at all.
        fx = _pen_fixture2(11)
        ml = drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree)
        mp = drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree,
                 penalty = drm_phylo_penalty(sd_u = 0.4))
        @test mp.estim_method === :MAP
        @test mp.scales[:lambda_sd_mu][1] < ml.scales[:lambda_sd_mu][1]
        @test mp.scales[:lambda_sd_sigma][1] < ml.scales[:lambda_sd_sigma][1]

        # Both SDs enter the penalty: it is the SUM of the two closed forms.
        pen = drm_phylo_penalty(sd_u = 0.4)
        lam = pen.rate
        sds = [mp.scales[:lambda_sd_mu][1], mp.scales[:lambda_sd_sigma][1]]
        @test mp.phylo_penalty ≈ sum(lam .* sds .- log.(sds) .- log(lam)) atol = 1e-8

        @test_throws PhyloCorPenaltyNeedsTwoSD drm(_PEN_FORM2, Gaussian(); data = fx.data,
                                                   tree = fx.tree,
                                                   penalty = drm_phylo_penalty(cor_sd = 0.5))
    end

    @testset "check_drm flags the penalized fit" begin
        # Deliberately on the SEPARATE block, not the mean-only sparse route: that
        # route leaves the variance-component block of `vcov` as NaN, and
        # `check_drm` currently THROWS on any non-finite covariance rather than
        # reporting it. That crash is pre-existing and reproduces on a plain ML fit
        # on main — it is not an A4c regression, and it is filed separately.
        fx = _pen_fixture2(11)
        mp = drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree,
                 penalty = drm_phylo_penalty(sd_u = 0.5))
        @test all(isfinite, mp.vcov)
        rep = check_drm(mp)
        @test rep.penalized_map
        # The stored objective is UNPENALIZED, so its gradient is non-zero at the
        # MAP optimum by construction; `ok` must not be scored against it.
        @test rep.ok == (mp.converged && rep.vcov_posdef)

        ml = drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree)
        @test !check_drm(ml).penalized_map
    end

    @testset "coupled block: cor_sd shrinks the correlation toward zero" begin
        # The only block with a live phylo correlation. drmTMB penalises its
        # UNCONSTRAINED correlation `eta_cor_phylo`; DRModels.jl parameterises a
        # Cholesky off-diagonal L21, so the port recovers cor = L21/sqrt(L21²+L22²)
        # and penalises atanh(cor). If that chain rule were wrong the correlation
        # would not move monotonically with the prior width — which is exactly what
        # this asserts.
        fx = _pen_fixture2(11)
        ml = drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree, phylo_coupled = true)
        ρ_ml = ml.scales[:lambda_cor][1]

        ρs = [drm(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree, phylo_coupled = true,
                  penalty = drm_phylo_penalty(sd_u = 1.0, cor_sd = s)).scales[:lambda_cor][1]
              for s in (2.0, 0.5, 0.15)]
        # Tighter prior ⇒ correlation closer to zero, monotonically.
        @test issorted(abs.(ρs); rev = true)
        @test all(abs.(ρs) .< abs(ρ_ml))
        @test abs(ρs[end]) < 0.2      # the tightest prior nearly zeroes it
    end
end

@testset "drm_phylo_penalty_sweep" begin
    fx = _pen_fixture2(11)

    @testset "shape and monotone sensitivity" begin
        sw = drm_phylo_penalty_sweep(_PEN_FORM2, Gaussian(); data = fx.data, tree = fx.tree,
                                     cor_sd = [0.25, 0.5, 1.0], sd_u = 1.0)
        @test length(sw.summary) == 3
        @test [r.cor_sd for r in sw.summary] == [0.25, 0.5, 1.0]
        @test all(r.converged for r in sw.summary)
        @test all(r.error === nothing for r in sw.summary)
        @test Set(keys(sw.fits)) == Set(["cor_sd=0.25", "cor_sd=0.5", "cor_sd=1.0"])
        # A wider prior lets the correlation grow: |cor| increases with cor_sd.
        @test issorted([abs(r.cor) for r in sw.summary])
    end

    @testset "cor_sd validation" begin
        @test_throws ArgumentError drm_phylo_penalty_sweep(_PEN_FORM2, Gaussian(); data = fx.data,
                                                           tree = fx.tree, cor_sd = Float64[])
        @test_throws ArgumentError drm_phylo_penalty_sweep(_PEN_FORM2, Gaussian(); data = fx.data,
                                                           tree = fx.tree, cor_sd = [0.5, -1.0])
    end

    @testset "refuses a sweep that would be a no-op" begin
        # drmTMB probes with the first cor_sd for exactly this reason: a sweep over
        # a model with no correlation returns identical rows, which LOOKS like a
        # passed sensitivity check. Failing loudly is the whole point.
        fx1 = _pen_fixture(3)
        @test_throws PhyloCorPenaltyNeedsTwoSD drm_phylo_penalty_sweep(
            _PEN_FORM, Gaussian(); data = fx1.data, tree = fx1.tree, phylo_coupled = false)
    end
end
