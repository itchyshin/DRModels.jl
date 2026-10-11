# Regression tests for the J1 lane (bootstrap, likelihood-ratio tests, model
# comparison):
#   #1036 two `marginal = :VA` fits are refused by lrtest / anova / lrt_boundary
#   #1053 equal `nobs` but different rows is "different samples"
#   #1002 lrt_boundary argument order and `q`; non-converged fits warn
#   #1046 `:recov` / `:phylocov` draws are unpacked for the boundary flag
#   #997  the marginal simulator refuses a random slope and warns on fallback
#   #1004 q=4 correlations of a zero-SD axis are NaN, not 0.0; ridge reported
#   #1051 fit_mixed_family bootstrap reports a status
#   #1011 the q2 bridge entry points reject unknown option keys
module TestLaneJ1BootstrapComparison

using DRModels
using Test
using Random
using Statistics
using LinearAlgebra
using SparseArrays

function _j1_argmsg(f)
    try
        f()
        error("expected an ArgumentError")
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
end

_j1_with_converged(fit::DrmFit, flag::Bool) =
    DrmFit((f === :converged ? flag : getfield(fit, f) for f in fieldnames(typeof(fit)))...)

@testset "J1 lane: bootstrap / LRT / comparison" begin
    rng = MersenneTwister(20261011)
    n = 60
    g = repeat(1:10, inner = 6)
    x = randn(rng, n)
    u = 0.6 .* randn(rng, 10)
    y = 0.4 .+ 0.5 .* x .+ u[g] .+ randn(rng, n)
    data = (; y, x, g)

    @testset "#1036 two VA fits are not compared" begin
        red = drm(bf(@formula(y ~ 1 + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
        full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
        red_va = DRModels._withmarginal(red, :VA)
        full_va = DRModels._withmarginal(full, :VA)
        msg = _j1_argmsg(() -> lrtest(red_va, full_va))
        @test occursin("lrtest", msg)
        @test occursin(":VA", msg)
        @test occursin("ELBO", msg)
        @test_throws ArgumentError anova(red_va, full_va)
        @test_throws ArgumentError lrt_boundary(full_va, red_va)
        # aic refuses the same fit, so lrtest and aic now agree.
        @test_throws ArgumentError aic(full_va)
        # The same pair with the default integrator is still compared.
        t = lrtest(red, full)
        @test t.dof == 1
        @test 0 <= t.pvalue <= 1
    end

    @testset "#1053 same nobs, different rows" begin
        form_full = bf(@formula(y ~ x), @formula(sigma ~ 1))
        form_red = bf(@formula(y ~ 1), @formula(sigma ~ 1))
        first_rows = (y = y[1:(n - 1)], x = x[1:(n - 1)])
        last_rows = (y = y[2:n], x = x[2:n])
        full = drm(form_full, Gaussian(); data = first_rows)
        red_other = drm(form_red, Gaussian(); data = last_rows)
        @test nobs(full) == nobs(red_other)
        msg = _j1_argmsg(() -> lrtest(red_other, full))
        @test occursin("different samples", msg)
        @test occursin("nobs = $(n - 1)", msg)
        @test occursin("different samples", _j1_argmsg(() -> lrt_boundary(full, red_other)))

        # Same rows: compared. The same rows in a different order are the same
        # sample, so a permuted data set is still accepted.
        red_same = drm(form_red, Gaussian(); data = first_rows)
        @test lrtest(red_same, full).dof == 1
        perm = reverse(1:(n - 1))
        red_perm = drm(form_red, Gaussian(); data = (y = first_rows.y[perm], x = first_rows.x[perm]))
        @test lrtest(red_perm, full).statistic ≈ lrtest(red_same, full).statistic atol = 1e-6
    end

    @testset "#1002 lrt_boundary order, q, and convergence" begin
        full = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
        red = drm(bf(@formula(y ~ x), @formula(sigma ~ 1)), Gaussian(); data)
        ok = lrt_boundary(full, red; q = 1)
        @test ok.q == 1
        # Swapped arguments used to give a negative statistic and p = 1.
        smsg = _j1_argmsg(() -> lrt_boundary(red, full; q = 1))
        @test occursin("more parameters", smsg)
        @test occursin("reverse of", smsg)
        # q larger than the parameters the full model adds.
        qmsg = _j1_argmsg(() -> lrt_boundary(full, red; q = 2))
        @test occursin("q = 2", qmsg)

        nc = _j1_with_converged(full, false)
        @test !nc.converged
        t = @test_logs (:warn, r"did not converge") match_mode = :any lrtest(red, nc)
        @test t.statistic ≈ 2 * (loglik(full) - loglik(red))
        tb = @test_logs (:warn, r"did not converge") match_mode = :any lrt_boundary(nc, red; q = 1)
        @test tb.pvalue == ok.pvalue
    end

    @testset "#1046 Cholesky blocks feed the boundary flag" begin
        # Only `blocks` and `coefnames` are read, so a NamedTuple stands in for a fit.
        recov_fit = (blocks = [:mu => 1:1, :recov => 2:4],
                     coefnames = [:mu => ["(Intercept)"], :recov => ["g:L11", "g:L22", "g:L21"]])
        nd = 40
        clean = hcat(randn(MersenneTwister(1), nd), fill(log(0.8), nd),
                     fill(log(0.5), nd), fill(0.1, nd))
        @test isempty(DRModels._bootstrap_boundary_params(recov_fit, clean))

        # Slope SD sqrt(L21² + L22²) ≈ 2e-9 in 10 of 40 draws: slope variance 0.
        slope0 = copy(clean)
        slope0[1:10, 3] .= -20.0
        slope0[1:10, 4] .= 0.0
        @test DRModels._bootstrap_boundary_params(recov_fit, slope0) ==
              ["recov:g:L11", "recov:g:L22", "recov:g:L21"]
        @test DRModels._bootstrap_status(nd, 0, true) == "bootstrap_at_boundary"

        # Intercept-slope correlation at ±1 (L22 tiny, L21 large).
        cor1 = copy(clean)
        cor1[1:4, 3] .= log(1e-3)
        cor1[1:4, 4] .= 1.0
        @test DRModels._bootstrap_cov_block_at_boundary(:recov, cor1[:, 2:4])

        # Too few draws for a share: silent.
        @test isempty(DRModels._bootstrap_boundary_params(recov_fit, slope0[1:19, :]))

        # q = 2 log-Cholesky `:phylocov` (column-major [log L11, L21, log L22]).
        phylo_fit = (blocks = [:mu1 => 1:1, :phylocov => 2:4],
                     coefnames = [:mu1 => ["(Intercept)"],
                                  :phylocov => ["Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L22"]])
        pclean = hcat(zeros(nd), fill(log(0.7), nd), fill(0.2, nd), fill(log(0.6), nd))
        @test isempty(DRModels._bootstrap_boundary_params(phylo_fit, pclean))
        pzero = copy(pclean)
        pzero[1:8, 3] .= 0.0
        pzero[1:8, 4] .= -20.0
        @test DRModels._bootstrap_boundary_params(phylo_fit, pzero) ==
              ["phylocov:Sigma_a:L11", "phylocov:Sigma_a:L21", "phylocov:Sigma_a:L22"]
    end

    @testset "#997 marginal simulator: slope refused, fallback warns" begin
        xs = randn(MersenneTwister(5), n)
        ys = 0.2 .+ 0.4 .* xs .+ u[g] .+ 0.3 .* randn(MersenneTwister(6), 10)[g] .* xs .+
             randn(MersenneTwister(7), n)
        dslope = (y = ys, x = xs, g = g)
        fslope = drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1)), Gaussian(); data = dslope)
        smsg = _j1_argmsg(() -> DRModels._marginal_simulator(fslope, dslope))
        @test occursin("random-slope", smsg)
        @test occursin("1 + x | g", smsg)
        @test_throws ArgumentError bootstrap_result(fslope; data = dslope, B = 2,
                                                    rng = MersenneTwister(1))

        h = repeat(1:6, outer = 10)
        dcross = (y = y .+ 0.5 .* randn(MersenneTwister(8), 6)[h], x = x, g = g, h = h)
        fcross = drm(bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1)), Gaussian();
                     data = dcross)
        sim = @test_logs (:warn, r"could not build the marginal simulator") match_mode = :any DRModels._marginal_simulator(fcross, dcross)
        @test sim === nothing

        # The intercept-only bar still builds a simulator, with no warning.
        fint = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data)
        sint = @test_logs DRModels._marginal_simulator(fint, data)
        @test sint !== nothing
        @test length(sint(MersenneTwister(2))) == n
    end

    @testset "#1004 q=4 correlations of a zero-SD axis" begin
        Σ = [1.0 0.3 0.0; 0.3 1.0 0.0; 0.0 0.0 0.0]
        c = DRModels._q4_cor_offdiag(Σ, ((1, 2), (1, 3), (2, 3)))
        @test c[1] ≈ 0.3
        @test isnan(c[2]) && isnan(c[3])           # was 0.0, a fake "estimated near 0"

        row = DRModels._q4_boot_row(:cor_mu1_sigma2, NaN, [0.1, NaN, 0.2, 0.3, NaN], 0.025)
        @test 0.1 <= row.lower <= row.upper <= 0.3
        @test isfinite(row.std_error)
        none = DRModels._q4_boot_row(:cor_mu1_sigma2, NaN, [NaN, 0.2], 0.025)
        @test isnan(none.lower) && isnan(none.upper)

        Q = sparse(1.0I, 3, 3)
        _, r0 = DRModels._q4_re_prior_chol(Q, [1.0 0.2; 0.2 1.0])
        @test r0 == 0.0
        _, r1 = DRModels._q4_re_prior_chol(Q, [1.0 1.0; 1.0 1.0])
        @test r1 > 0
    end

    @testset "#1051 fit_mixed_family bootstrap status" begin
        good(k) = [(; rho_latent = 0.40 + 0.01 * i, converged = true, loglik = -100.0) for i in 1:k]
        ci, kept, status = DRModels._mf_boot_ci(Any[good(12)...], 12, 0.95)
        @test kept == 12 && status == "bootstrap"
        @test all(isfinite, ci)

        fits = Any[good(12)...; (; rho_latent = 0.9, converged = false, loglik = -100.0); nothing]
        out = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._mf_boot_ci(fits, 14, 0.95)
        @test out[2] == 12
        @test out[3] == "bootstrap_incomplete"

        few = Any[good(3)...; nothing; nothing]
        out2 = @test_logs (:warn, r"bootstrap_unavailable") match_mode = :any DRModels._mf_boot_ci(few, 5, 0.95)
        @test all(isnan, out2[1])
        @test out2[3] == "bootstrap_unavailable"

        edge = Any[[(; rho_latent = 0.995, converged = true, loglik = -100.0) for _ in 1:3]...;
                   good(27)...]
        out3 = @test_logs (:warn, r"bootstrap_at_boundary") match_mode = :any DRModels._mf_boot_ci(edge, 30, 0.95)
        @test out3[3] == "bootstrap_at_boundary"
    end

    @testset "#1011 q2 bridge rejects unknown option keys" begin
        tree = "((a:1,b:1):1,(c:1,d:1):1);"
        sp = repeat(1:4, inner = 10)
        Y = randn(MersenneTwister(9), 40, 2)
        X = reshape(randn(MersenneTwister(10), 40), :, 1)
        msg = _j1_argmsg(() -> DRModels.drm_bridge_q2_phylo(; Y, X, species = sp, tree,
            options = Dict("iteratons" => 5, "bogus_key" => 1)))
        @test occursin("drm_bridge_q2_phylo", msg)
        @test occursin("iteratons", msg) && occursin("bogus_key", msg)
        @test occursin("iterations", msg)           # the known keys are listed
        Q = Matrix{Float64}(I, 4, 4) .* 2.0
        kmsg = _j1_argmsg(() -> DRModels.drm_bridge_q2_known_precision(; Y, X, group = sp, Q,
            options = Dict("g_tool" => 1e-4)))
        @test occursin("drm_bridge_q2_known_precision", kmsg)
        @test occursin("g_tool", kmsg)
    end

    @testset "#1052 a thrown refit is recorded by default (decision 18)" begin
        form = bf(@formula(y ~ x), @formula(sigma ~ 1))
        d0 = (y = y, x = x)
        fit0 = drm(form, Gaussian(); data = d0)
        calls = Ref(0)
        flaky = datab -> (calls[] += 1; calls[] == 2 ? error("forced refit failure") :
                          drm(form, Gaussian(); data = datab))
        res = @test_logs (:warn, r"bootstrap_incomplete") match_mode = :any DRModels._bootstrap_result(
            fit0, form, d0, 4, 0.95, MersenneTwister(3), false, flaky)
        @test res.attempted == 4
        @test res.used == 3
        @test res.failed == 1
        @test res.status == "bootstrap_incomplete"
        @test occursin("forced refit failure", res.failures[1].message)
        # The abort is still available on request.
        calls[] = 0
        @test_throws ErrorException DRModels._bootstrap_result(
            fit0, form, d0, 4, 0.95, MersenneTwister(3), false, flaky; failures = :error)
        # Every replicate failing is still an error under the default.
        @test_throws ErrorException DRModels._bootstrap_result(
            fit0, form, d0, 2, 0.95, MersenneTwister(3), false, _ -> error("always"))
    end

    @testset "#951 bridge refuses keywords the family's drm does not take" begin
        rp = MersenneTwister(11)
        xp = randn(rp, 50)
        dp = Dict("y" => Float64.(rand(rp, 0:4, 50)), "x" => xp)
        # Poisson's drm has no `algorithm` keyword: a named refusal, not a MethodError.
        amsg = _j1_argmsg(() -> drm_bridge(; formula = "y ~ x", family = "poisson", data = dp,
            options = Dict{String,Any}("algorithm" => "lbfgs")))
        @test occursin("algorithm", amsg)
        @test occursin("Poisson()", amsg)
        # Student's drm has no covariance-provider keywords.
        ds = Dict("y" => randn(rp, 50), "x" => xp)
        tmsg = _j1_argmsg(() -> drm_bridge(; formula = "y ~ x", family = "student", data = ds,
            tree = "((a:1,b:1):1,(c:1,d:1):1);"))
        @test occursin("tree", tmsg)
        @test occursin("structured", tmsg)
        # A keyword the family does declare is still forwarded.
        ok = drm_bridge(; formula = "y ~ x", family = "poisson", data = dp,
                        options = Dict{String,Any}("se" => false))
        @test isfinite(ok["loglik"])
    end
end

end # module
