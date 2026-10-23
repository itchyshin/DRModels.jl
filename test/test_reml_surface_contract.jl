# test_reml_surface_contract.jl — #624 (mirror of drmTMB #1142): the REML
# capability surface the census in reml-surface.md measured must stay honest
# at two boundaries:
#
#   (i)  drm_bridge(...) must report which estimator actually ran
#        (`estim_method`) and carry the restricted vs plain log-likelihood
#        (`reml_loglik`/`ml_loglik`) instead of leaving the R caller unable
#        to tell a REML fit's `loglik`/`aic`/`bic` from an ML one's.
#  (ii)  the generic univariate REML refusal (src/gaussian_core.jl) must name
#        the real supported set — the census found REML also fits on every
#        sd() LSS route and both bivariate structured routes, not just the
#        two cells the old message claimed.
#  (iii) a positive control: three cheap cells the census marked FITS must
#        still fit by REML with a restricted log-likelihood different from
#        the ML one.
using DRModels
using Test, Random, LinearAlgebra

@testset "REML capability-surface contract (#624)" begin
    @testset "(i) drm_bridge reports estim_method and the REML/ML loglik" begin
        Random.seed!(20260614)
        p11 = 16; m11 = 5
        phy11 = random_balanced_tree(p11; branch_length = 0.3)
        K11 = DRModels._phylo_correlation(phy11)
        LC11 = cholesky(Symmetric(K11)).L
        Lam_among = [0.25 0.10 0.05 0.00; 0.10 0.25 0.00 0.04;
                     0.05 0.00 0.16 0.02; 0.00 0.04 0.02 0.16]
        U11 = LC11 * randn(p11, 4) * cholesky(Symmetric(Lam_among)).L'
        sp11 = repeat(1:p11, inner = m11); n11 = length(sp11); x11 = randn(n11)
        y1_11 = 0.5 .+ 0.3 .* x11 .+ U11[sp11, 1] .+ exp.(-0.6 .+ U11[sp11, 3]) .* randn(n11)
        y2_11 = -0.2 .+ 0.4 .* x11 .+ U11[sp11, 2] .+ exp.(-0.6 .+ U11[sp11, 4]) .* randn(n11)
        sp_name11 = phy11.leaf_names[sp11]
        data_q4 = (; y1 = y1_11, y2 = y2_11, x = x11, species = sp_name11)
        formula_str = Dict(
            "mu1" => "y1 ~ x + phylo(1 | species)",
            "mu2" => "y2 ~ x + phylo(1 | species)",
            "sigma1" => "sigma1 ~ 1 + phylo(1 | species)",
            "sigma2" => "sigma2 ~ 1 + phylo(1 | species)",
            "rho12" => "rho12 ~ 1",
        )

        res_reml = drm_bridge(; formula = formula_str, family = "biv_gaussian", data = data_q4,
                               tree = phy11, options = Dict("method" => "REML", "q4_vcov" => false))
        res_ml = drm_bridge(; formula = formula_str, family = "biv_gaussian", data = data_q4,
                             tree = phy11, options = Dict("method" => "ML", "q4_vcov" => false))

        @test res_reml["estim_method"] == "REML"
        @test res_ml["estim_method"] == "ML"
        @test haskey(res_reml, "reml_loglik")
        @test isfinite(res_reml["reml_loglik"])
        @test haskey(res_reml, "ml_loglik")
        @test haskey(res_ml, "ml_loglik")
        # ML fit carries no restricted objective to report.
        @test !haskey(res_ml, "reml_loglik")
        # The bridge-level loglik genuinely differs between the two estimators
        # on identical data (not silently the same value under two labels).
        @test res_reml["loglik"] != res_ml["loglik"]
        # infocrit_basis / warnings crossing convention (chosen over dropping
        # aic/bic outright): REML fits keep aic/bic but flag their basis and
        # carry the SAME caveat text `_reml_infocrit_warn` prints server-side.
        @test res_reml["infocrit_basis"] == "reml"
        @test res_ml["infocrit_basis"] == "ml"
        @test haskey(res_reml, "warnings")
        @test !isempty(res_reml["warnings"])
        @test any(occursin("REML log-likelihoods are only comparable", w) for w in res_reml["warnings"])
        @test !haskey(res_ml, "warnings")
    end

    @testset "(ii) generic univariate REML refusal names the real supported set" begin
        Random.seed!(20260817)
        G = 12; m = 4; n2 = G * m
        g = repeat(1:G, inner = m)
        x2 = randn(n2)
        b = 0.8 .* randn(G)
        y2 = 0.4 .+ 0.5 .* x2 .+ b[g] .+ 0.6 .* randn(n2)
        data_ri = (; y = y2, x = x2, g)

        err = nothing
        try
            # A random effect on sigma trips the generic gate (census row 3). The mean
            # correlated slope (1 + x | g) is now supported (test_lme4_twins.jl).
            drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1 + (1 | g))), Gaussian();
                data = data_ri, method = :REML)
        catch e
            err = e
        end
        @test err isa ArgumentError
        msg = sprint(showerror, err)
        @test occursin("sd() LSS route", msg) || occursin("sd() route", msg)
        @test occursin("bivariate structured", msg) || occursin("q=2 and q=4", msg)
    end

    @testset "(iii) positive control: three census FITS cells stay REML with a restricted loglik" begin
        # Univariate fixed-effect Gaussian location-scale (census row 1).
        Random.seed!(20260901)
        n = 60
        x = randn(n); z = randn(n)
        y = 1.5 .+ 0.7 .* x .- 0.4 .* z .+ 1.3 .* randn(n)
        data_fe = (; y, x, z)
        f1 = drm(bf(@formula(y ~ 1 + x), @formula(sigma ~ 1)), Gaussian(); data = data_fe, method = :REML)
        @test estimation_method(f1) === :REML
        @test isfinite(reml_loglik(f1))
        @test reml_loglik(f1) != ml_loglik(f1)

        # Gaussian mean random intercept (1 | g) (census row 2).
        Random.seed!(20260817)
        G = 12; m = 4; n2 = G * m
        g = repeat(1:G, inner = m)
        x2 = randn(n2)
        b = 0.8 .* randn(G)
        y2 = 0.4 .+ 0.5 .* x2 .+ b[g] .+ 0.6 .* randn(n2)
        data_ri = (; y = y2, x = x2, g)
        f2 = drm(bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1)), Gaussian(); data = data_ri, method = :REML)
        @test estimation_method(f2) === :REML
        @test isfinite(reml_loglik(f2))
        @test reml_loglik(f2) != ml_loglik(f2)

        # LSS sd(g) single-component (census row 6, #558).
        Random.seed!(20260828)
        Gl = 15; ml = 6; nl = Gl * ml
        gl = repeat(1:Gl, inner = ml)
        xl = randn(nl)
        zgl = randn(Gl)
        zl = zgl[gl]
        log_sig_b = 0.2 .+ 0.5 .* zgl
        bl = exp.(log_sig_b) .* randn(Gl)
        log_sig_e = -0.3 .+ 0.2 .* xl
        yl = 1.0 .+ 0.7 .* xl .+ bl[gl] .+ exp.(log_sig_e) .* randn(nl)
        data_lss1 = (; y = yl, x = xl, z = zl, g = gl)
        flss1 = bf(@formula(y ~ 1 + x + (1 | g)), @formula(sigma ~ 1 + x), @formula(sd(g) ~ 1 + z))
        f3 = drm(flss1, Gaussian(); data = data_lss1, method = :REML)
        @test estimation_method(f3) === :REML
        @test f3.converged
        @test isfinite(reml_loglik(f3))
        @test reml_loglik(f3) != ml_loglik(f3)
    end
end
