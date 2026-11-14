# vcov at a flagged boundary coordinate (DRModels.jl#1383).
#
# When the Hessian guard flags a coordinate as numerically flat, that coordinate is
# dropped before the rest of the Hessian is inverted, and its row and column of
# `vcov` are NaN. Before, the pseudo-inverse inverted finite-difference noise of
# either sign on the flagged coordinate (-1.0e8 in the issue's fit).

using DRModels
using Test
using LinearAlgebra
using Logging

const _DVG = DRModels

@testset "vcov boundary NaN (#1383)" begin
    @testset "unit: flagged rows and columns are NaN, the rest is inv of the block" begin
        for noise in (-1e-8, -1e-9, 1e-9, 1e-8)
            # Third coordinate flat; weak coupling to the others must not leak in.
            H = [39.8 -14.5 1e-7 3.8
                 -14.5 85.6 -2e-7 0.4
                 1e-7 -2e-7 noise 1e-7
                 3.8 0.4 1e-7 12.1]
            V = @test_logs (:warn,) _DVG._vcov_from_hessian(H)
            @test all(isnan, V[3, :])
            @test all(isnan, V[:, 3])
            keep = [1, 2, 4]
            Hs = Matrix(Symmetric((H + H') / 2))
            @test V[keep, keep] ≈ inv(Hs[keep, keep]) rtol = 1e-12
            @test all(diag(V)[keep] .> 0)
            @test issymmetric(V[keep, keep])
        end
    end

    @testset "unit: several flagged coordinates, and all coordinates flagged" begin
        H = [50.0 1.0 0.0 0.0
             1.0 30.0 0.0 0.0
             0.0 0.0 1e-9 0.0
             0.0 0.0 0.0 -1e-9]
        V = @test_logs (:warn,) _DVG._vcov_from_hessian(H)
        @test all(isnan, V[3:4, :]) && all(isnan, V[:, 3:4])
        @test V[1:2, 1:2] ≈ inv(H[1:2, 1:2])
        V0 = @test_logs (:warn,) _DVG._vcov_from_hessian(zeros(3, 3))
        @test all(isnan, V0)
    end

    @testset "unit: no flagged coordinate keeps the old behaviour" begin
        # Well conditioned: plain inverse, no warning.
        H = [39.8 -14.5 3.8; -14.5 85.6 0.4; 3.8 0.4 12.1]
        @test _DVG._vcov_from_hessian(H) == Matrix(Symmetric(inv(Matrix(Symmetric(H)))))
        # Singular along a combination of coordinates: no diagonal entry is flat, so
        # nothing is flagged and the pseudo-inverse is used as before.
        H2 = [50.0 49.9999999 0.0
              49.9999999 50.0 0.0
              0.0 0.0 20.0]
        V2 = @test_logs (:warn,) _DVG._vcov_from_hessian(H2)
        @test V2 ≈ pinv(Matrix(Symmetric(H2)))
        @test all(isfinite, V2)
    end

    @testset "fit: Binomial (1 | g), marginal = :Laplace, group SD at zero" begin
        x = [1.43941117899116, 0.778826426368403, 0.192743160380563, -0.3244252191031, 2.09766874641634, 0.369686510094048, 1.79478191004505, 0.606502031264434, 0.0312916522780309, -0.910572582677395, 0.806725615836395, 0.612734177592478, -0.611163641309853, -0.159343666027289, 1.05500813010401, -1.06458870705857, 1.12065628331957, 0.387953130608195, -0.224004141372002, -0.109023652368118]
        g = repeat(["g01", "g02", "g03", "g04"], 5)
        s = [4, 6, 4, 7, 6, 3, 4, 6, 3, 5, 5, 6, 5, 7, 7, 5, 6, 5, 2, 5]
        fl = [4, 2, 4, 1, 2, 5, 4, 2, 5, 3, 3, 2, 3, 1, 1, 3, 2, 3, 6, 3]
        d = (; x, g, s, fl)
        fit = Logging.with_logger(Logging.NullLogger()) do
            drm(bf(@formula(cbind(s, fl) ~ x + (1 | g))), Binomial(); data = d, marginal = :Laplace)
        end
        V = vcov(fit)
        # the issue's assertion
        @test all(d -> isnan(d) || d >= 0, diag(V))
        @test !any(<(0), diag(V))
        @test all(isnan, V[3, :]) && all(isnan, V[:, 3])
        @test all(isfinite, V[1:2, 1:2])
        @test isposdef(Symmetric(V[1:2, 1:2]))
        # the location SEs are the ones the issue quotes
        @test sqrt(V[1, 1]) ≈ 0.1802 atol = 1e-4
        @test sqrt(V[2, 2]) ≈ 0.1980 atol = 1e-4
        # code that takes sqrt of the diagonal no longer throws
        @test (sqrt.(diag(V)); true)
        se = stderror(fit)
        @test all(isfinite, se[1:2]) && isnan(se[3])
        @test bridge_diagnostics(fit).boundary == [3]
        @test convergence_status(fit) === :boundary
        # printing and tables: no error, NaN rows
        ct = coeftable(fit)
        @test length(ct.rownms) == 3
        out = sprint(show, MIME"text/plain"(), fit)
        @test occursin("NaN", out)
        @test occursin("Wald SEs withheld", out)
        # Wald interval: location finite, flagged coordinate unbounded
        ci = confint(fit; method = :wald)
        @test all(isfinite, (ci[1].lower, ci[1].upper, ci[2].lower, ci[2].upper))
        @test ci[3].lower == -Inf && ci[3].upper == Inf
        # profile interval does not read vcov for its end points
        cip = confint(fit; method = :profile)
        @test cip[3].lower == 0.0 && isfinite(cip[3].upper)
        # delta method: a quantity of the flagged coordinate gets no number,
        # one that does not involve it keeps its standard error
        bc_flag = bias_correct(fit, θ -> exp(θ[3]))
        @test isnan(bc_flag.corrected) && !isfinite(bc_flag.se)
        bc_loc = bias_correct(fit, θ -> θ[1] + θ[2])
        @test isfinite(bc_loc.se) && isfinite(bc_loc.corrected)
        @test bc_loc.se ≈ sqrt(sum(V[1:2, 1:2])) rtol = 1e-10
    end
end
