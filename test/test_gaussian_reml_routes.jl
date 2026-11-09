# Gaussian REML on the generic univariate routes that drmTMB 0.8.1 fits with REML = TRUE:
#   A  slope only                    y ~ x + (0 + x | g)
#   B  independent scalar terms      (1 | g) + (0 + x1 | g) + ..., and crossed (1 | g) + (1 | g2)
#   C  relmat(1 | g) / animal(1 | g) (dense structured intercept)
#   D  phylo(1 + x | g)              (two SDs)
# Every cell is exact for a Gaussian model: REML = ML + ½ logdet(X'V⁻¹X) − ½ p log(2π).
# Agreement thresholds are fixed in docs/dev-log/2026-11-09-gaussian-reml-routes.md and in
# fixtures/gaussian_reml_routes/compare.jl (committed data, R reference from gen_ref.R).
using DRModels, Test, Statistics

include(joinpath(@__DIR__, "fixtures", "gaussian_reml_routes", "compare.jl"))

@testset "Gaussian REML routes agree with R drmTMB 0.8.1" begin
    T = REML_ROUTE_THRESH
    for (grp, shapes) in REML_ROUTE_GROUPS
        @testset "$grp" begin
            for shape in shapes, seed in (11, 12, 13)
                r = reml_route_compare(shape, seed)
                @test r.estim === :REML
                @test r.converged
                @test r.r_converged
                @test r.ll_diff <= T.loglik
                @test r.mu_se_units <= T.mu_se_units
                r.boundary || @test r.sd_rel <= T.sd_rel
            end
        end
    end
end

@testset "REML refusals keep accurate wording" begin
    d = (; y = randn(40), x1 = randn(40), g = repeat(string.('a':'j'), inner = 4),
         g2 = repeat(string.('p':'t'), outer = 8))
    # a correlated block beside another term: ML only, no restricted objective
    err = try
        drm(bf(DRModels.@formula(y ~ x1 + (1 + x1 | g) + (1 | g2)), DRModels.@formula(sigma ~ 1)),
            Gaussian(); data = d, method = :REML); nothing
    catch e; e; end
    @test err isa ArgumentError
    @test occursin("correlated", err.msg) && occursin("maximum likelihood only", err.msg)
    # a heteroscedastic sigma on a multi-term route is not admitted
    err2 = try
        drm(bf(DRModels.@formula(y ~ x1 + (1 | g) + (1 | g2)), DRModels.@formula(sigma ~ x1)),
            Gaussian(); data = d, method = :REML); nothing
    catch e; e; end
    @test err2 isa ArgumentError && occursin("not implemented", err2.msg)
end
