# Bivariate Gaussian with an ordinary `(1 | p | g)`: fixes from the independent review.
#
# 1. Interior fixtures pinned to drmTMB 0.8.1 (the first fixture sits on the group
#    correlation boundary): balanced, unbalanced with singleton groups, and
#    covariates on sigma1 and rho12; ML and REML. Truth: group SD 0.8 / 0.6, group
#    corr 0.5, residual rho12 0.3; sample group corr 0.09 / 0.77 / 0.62.
# 2. `simulate` and the parametric bootstrap draw FRESH group effects from the fitted
#    2x2 group covariance. Before, bootstrap fell back to `re = :zero` and refitted
#    data with no group effect (bootstrap SE of the mu1 intercept 0.069 against model
#    SE 0.116 in the review).
# 3. Keywords of other routes (`tree`, `K`, `A`, ...) are refused by name.
# 4. Naming: the group covariance is `fit.ranef.Sigma_g` and the coefficients are
#    `Sigma_g:L11/L21/L22` (the structured routes own `Sigma_a`).
# 5. `residuals` of this fit are marginal (y minus X b), not conditional on BLUPs.
include(joinpath(@__DIR__, "biv_ordinary_re_helpers.jl"))
using StableRNGs, Statistics

@testset "bivariate ordinary (1 | p | g): review fixes" begin
    with_logger(NullLogger()) do
        @testset "interior fixtures pinned to drmTMB" begin
            dbal = _bre_data("interior_bal.csv")
            dun = _bre_data("interior_unbal.csv")
            dcov = _bre_data("interior_cov.csv")
            @testset "balanced 40 x 8, ML" begin
                _bre_check("int_bal_ml", drm(_BRE_F0, Gaussian(); data = dbal), 7)
            end
            @testset "unbalanced (30 groups of 2-12 plus 4 singletons), ML" begin
                _bre_check("int_unbal_ml", drm(_BRE_F0, Gaussian(); data = dun), 7)
            end
            @testset "unbalanced, REML" begin
                _bre_check("int_unbal_reml", drm(_BRE_F0, Gaussian(); data = dun, method = :REML), 7)
            end
            @testset "sigma1 ~ x1 and rho12 ~ x1, ML" begin
                _bre_check("int_cov_ml", drm(_BRE_FC, Gaussian(); data = dcov), 9)
            end
            @testset "sigma1 ~ x1 and rho12 ~ x1, REML" begin
                _bre_check("int_cov_reml", drm(_BRE_FC, Gaussian(); data = dcov, method = :REML), 9)
            end
        end

        dbal = _bre_data("interior_bal.csv")
        fit = drm(_BRE_F0, Gaussian(); data = dbal)

        @testset "naming: Sigma_g in code, names and ranef" begin
            @test haskey(fit.ranef, :Sigma_g)
            @test !haskey(fit.ranef, :Sigma_a)
            @test Dict(fit.coefnames)[DRModels._BIV_RE_GROUP_BLOCK] == ["Sigma_g:L11", "Sigma_g:L21", "Sigma_g:L22"]
        end

        @testset "residuals are marginal (y - X b), not conditional on BLUPs" begin
            r = DRModels.residuals(fit)
            @test r[:mu1] ≈ dbal.y1 .- fit.means[:mu1]
            @test r[:mu2] ≈ dbal.y2 .- fit.means[:mu2]
        end

        @testset "keywords of other routes are refused by name" begin
            nd = (; g = ["a"], x1 = [0.0])
            for kw in (:tree, :K, :A, :coords, :spatial_range)
                err = try
                    drm(_BRE_F0, Gaussian(); data = dbal, NamedTuple{(kw,)}((kw === :spatial_range ? 1.0 : nd,))...)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("`$kw`", sprint(showerror, err))
            end
        end

        @testset "q4_* tuning keywords are refused by name" begin
            for (kw, v) in (:q4_g_tol => 1e-5, :q4_f_reltol => 1e-12, :q4_iterations => 10,
                            :q4_n_newton => 5, :q4_vcov => false)
                err = try
                    drm(_BRE_F0, Gaussian(); data = dbal, NamedTuple{(kw,)}((v,))...)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("`$kw`", sprint(showerror, err))
            end
        end

        @testset "simulate draws the fitted between-group covariance" begin
            # Balanced 8 rows per group, constant sigma and rho12: the covariance of
            # the group mean of (y - X b) is Sigma_g + Sigma_res / 8. 400 draws x 40
            # groups = 16000 group means, so each entry is good to about 1.5 percent;
            # the bound is 6 percent of the larger variance (a dropped group effect
            # would lose about 80 percent of it).
            sims = simulate(fit; data = dbal, nsim = 400, rng = StableRNG(11))
            gl = unique(dbal.g)
            idx = [findall(==(g), dbal.g) for g in gl]
            M = zeros(2, 0)
            for s in sims
                gm = hcat([[mean(s[:mu1][i] .- fit.means[:mu1][i]), mean(s[:mu2][i] .- fit.means[:mu2][i])] for i in idx]...)
                M = hcat(M, gm)
            end
            emp = (M * M') ./ size(M, 2)
            s1, s2, ρ = fit.scales[:sigma1][1], fit.scales[:sigma2][1], fit.scales[:rho12][1]
            target = fit.ranef.Sigma_g + [s1^2 ρ*s1*s2; ρ*s1*s2 s2^2] ./ 8
            @test maximum(abs.(emp - target)) <= 0.06 * maximum(diag(target))
            # population-level draws carry only the residual part
            z = simulate(fit; re = :zero, nsim = 400, rng = StableRNG(12))
            Mz = hcat([hcat([[mean(s[:mu1][i] .- fit.means[:mu1][i]), mean(s[:mu2][i] .- fit.means[:mu2][i])] for i in idx]...) for s in z]...)
            @test (Mz * Mz')[1, 1] / size(Mz, 2) < 0.4 * target[1, 1]
        end

        @testset "simulate refuses rather than guessing" begin
            @test_throws ArgumentError simulate(fit)                     # needs `data`
            short = merge(dbal, (; g = dbal.g[1:end-1], x1 = dbal.x1[1:end-1], x2 = dbal.x2[1:end-1],
                                   y1 = dbal.y1[1:end-1], y2 = dbal.y2[1:end-1]))
            @test_throws ArgumentError simulate(fit; data = short)        # rows differ from the fit
        end

        @testset "bootstrap redraws the group effect" begin
            # 24 groups of 3, group SD 1.0 / 0.8, residual SD 0.5: the group effect
            # dominates the mu1 intercept SE (model SE about 0.21; with no group
            # effect it would be about 0.06). B = 40 refits: the bootstrap SE of an
            # SD-like quantity then has about 11 percent Monte Carlo error, so the
            # ratio to the model SE must lie in [0.65, 1.5] (the buggy fallback gave
            # about 0.3 here).
            rng = StableRNG(2026)
            G, k = 24, 3
            g = repeat(["g$i" for i in 1:G], inner = k)
            u1 = 1.0 .* randn(rng, G); u2 = 0.8 .* (0.5 .* (u1 ./ 1.0) .+ sqrt(0.75) .* randn(rng, G))
            gi = repeat(1:G, inner = k); n = G * k
            x1 = randn(rng, n); x2 = randn(rng, n)
            e1 = randn(rng, n); e2 = 0.3 .* e1 .+ sqrt(0.91) .* randn(rng, n)
            d = (; g, x1, x2, y1 = 0.3 .+ 0.5 .* x1 .+ u1[gi] .+ 0.5 .* e1,
                   y2 = -0.3 .+ 0.4 .* x2 .+ u2[gi] .+ 0.5 .* e2)
            f0 = drm(_BRE_F0, Gaussian(); data = d)
            se_model = sqrt(f0.vcov[1, 1])
            br = DRModels.bootstrap_result(f0; data = d, B = 40, rng = StableRNG(5))
            @test br.used >= 36
            row = first(r for r in br.summary if r.param === :mu1)
            ratio = row.std_error / se_model
            @test 0.65 <= ratio <= 1.5
            # the group-SD coefficient must actually vary across refits (it collapsed
            # to a constant boundary value under the dropped-effect fallback)
            lrow = first(r for r in br.summary if r.param === DRModels._BIV_RE_GROUP_BLOCK && endswith(r.coef, "L11"))
            @test lrow.std_error < 3.0 && lrow.upper - lrow.lower > 0.1
            # a missing-response row is dropped by the fit, so the data no longer line
            # up with the fit: refuse (never fall back to the group-free draw)
            dm = merge(d, (; y1 = Union{Missing,Float64}[missing; d.y1[2:end]]))
            fm = drm(_BRE_F0, Gaussian(); data = dm)
            @test_throws ArgumentError DRModels.bootstrap_result(fm; data = dm, B = 2, rng = StableRNG(1))
        end
    end
end
