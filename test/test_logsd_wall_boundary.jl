# Independent-term Laplace route: the objective returns its failure value when a random-effect
# log SD leaves [-30, 30]. The guard stays at 30. A fit that ends at the lower end has a zero SD
# for every practical purpose; it must be reported as a boundary fit (with a warning that points to
# the mixture LRT and the profile interval), not as `:degenerate`.
# Cell: Binomial `(1 | g) + (0 + x1 | g)` with `sd(g, coef(intercept)) ~ z` (the committed
# fixtures/binomial_indep_restarts data): the intercept-SD likelihood is a monotone ridge, drmTMB 0.8.1
# follows it to a log SD near -111 (logLik -253.5513); Julia stops at the wall of 30.
using DRModels, Test, Logging
import DelimitedFiles

function _lsw_data()
    m, h = DelimitedFiles.readdlm(joinpath(@__DIR__, "fixtures", "binomial_indep_restarts", "g12_binomial.csv"),
                                  ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
            x3 = Float64.(col("x3")), z = Float64.(col("z")), g = String.(col("g")))
end

@testset "log-SD wall: a fit that ends there is a flagged boundary fit" begin
    d = _lsw_data()
    f = bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g)), @formula(sd(g, coef(intercept)) ~ z))
    fit = @test_logs (:warn, r"lower search bound") match_mode = :any drm(f, Binomial(:logit); data = d)
    @test !fit.converged                                  # the optimiser did stop at a wall
    @test convergence_status(fit) === :boundary           # not :degenerate
    @test !is_converged(fit)
    # The guard value is unchanged: the fit ends where it did before, and the wall still rejects.
    @test abs(loglikelihood(fit) - (-253.9951592)) <= 1e-4
    th = copy(fit.theta); th[5] = -1000.0
    @test fit.nll(th) >= 1e17
    th = copy(fit.theta); th[5] = -31.0; th[6] = 0.0     # log SD of the intercept term below -30
    @test fit.nll(th) >= 1e17
    # An independent-term fit that does not touch the wall carries no wall flag.
    ok = drm(bf(@formula(y ~ x1 + x2 + x3 + (1 | g) + (0 + x1 | g))), Binomial(:logit); data = d)
    @test !haskey(ok.scales, :cond_logsd_wall)
end
