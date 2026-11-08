# Tweedie phylo(1 | g) and relmat(1 | g) mean intercepts, with and without an
# `sd(g, …) ~ …` formula. Fixtures are committed files (test/fixtures/
# tweedie_structured_sd), so nothing here depends on a sampler's output.
# R reference values are drmTMB 0.8.1 estimates on the same files (link scale,
# order: mu, sigma, nu, sd), from agreement_r.R; both engines maximise the same
# Laplace objective, so estimates are compared at the optimum, never log-likelihoods.
using DRModels, Test, DelimitedFiles, LinearAlgebra
import ForwardDiff

const _TSD_DIR = joinpath(@__DIR__, "fixtures", "tweedie_structured_sd")

function _tsd_data(name)
    m, h = readdlm(joinpath(_TSD_DIR, name * ".csv"), ',', Any; header = true)
    h = String.(vec(h)); col(n) = m[:, findfirst(==(n), h)]
    return (; y = Float64.(col("y")), x = Float64.(col("x")), z = Float64.(col("z")),
            g = String.(col("g")))
end
_tsd_tree(name) = strip(read(joinpath(_TSD_DIR, name * "_tree.nwk"), String))
_tsd_K(name) = Matrix{Float64}(readdlm(joinpath(_TSD_DIR, name * "_K.csv"), ',', Float64;
                                       skipstart = 1))

@testset "Tweedie structured SD" begin
    @testset "per-observation eta derivatives match finite differences" begin
        for (y, eta) in ((0.0, 0.3), (1.7, 0.2), (0.4, -0.5)), (phi, p) in ((0.4, 1.5), (0.9, 1.3))
            nll(e) = -DRModels._logpdf_tweedie(y, exp(e), phi, p)
            d1, d2 = DRModels._tweedie_eta_derivs(y, eta, phi, p)
            h = 1e-5
            @test d1 ≈ (nll(eta + h) - nll(eta - h)) / 2h rtol = 1e-6 atol = 1e-8
            @test d2 ≈ (nll(eta + h) - 2nll(eta) + nll(eta - h)) / h^2 rtol = 1e-4 atol = 1e-5
        end
    end

    @testset "phylo(1 | g): bare, sd ~ 1 and sd ~ z agree with R" begin
        d, nwk = _tsd_data("phylo_2"), _tsd_tree("phylo_2")
        bare = drm(bf(@formula(y ~ x + phylo(1 | g))), Tweedie(); data = d, tree = nwk)
        @test bare.converged
        @test coef(bare) ≈ [0.58501578, -0.2866463, -0.19374878, -0.08377664, 0.08190024] atol = 1e-4
        one = drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sd(g, phylogenetic) ~ 1)),
                  Tweedie(); data = d, tree = nwk)
        @test coef(one) ≈ coef(bare) atol = 1e-6
        withz = drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sd(g, phylogenetic) ~ z)),
                    Tweedie(); data = d, tree = nwk)
        @test withz.converged
        @test coef(withz) ≈ [0.66245785, -0.27695708, -0.20016022, -0.10993684, -0.25824751, 0.32060831] atol = 1e-4
        @test all(isfinite, sqrt.(diag(vcov(withz))))
        pp = predict_parameters(withz, d)
        @test length(pp[:sd]) == length(d.y)
        @test pp[:sd] ≈ exp.(coef(withz, :sd)[1] .+ coef(withz, :sd)[2] .* d.z) rtol = 1e-8
    end

    @testset "phylo(1 | g) second dataset" begin
        d, nwk = _tsd_data("phylo_3"), _tsd_tree("phylo_3")
        fit = drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sd(g, phylogenetic) ~ z)),
                  Tweedie(); data = d, tree = nwk)
        @test fit.converged
        @test coef(fit) ≈ [0.51870767, 0.34811546, -0.64422215, 0.60131871, -0.88651986, 0.66920516] atol = 1e-4
    end

    @testset "relmat(1 | g): bare and sd ~ z agree with R" begin
        d1, K1 = _tsd_data("relmat_1"), _tsd_K("relmat_1")
        bare = drm(bf(@formula(y ~ x + relmat(1 | g))), Tweedie(); data = d1, K = K1)
        @test bare.converged
        @test coef(bare) ≈ [0.54646586, 0.3410769, -0.48593459, 0.02911859, -0.29799234] atol = 1e-4
        d2, K2 = _tsd_data("relmat_2"), _tsd_K("relmat_2")
        withz = drm(bf(@formula(y ~ x + relmat(1 | g)), @formula(sd(g, relmat) ~ z)),
                    Tweedie(); data = d2, K = K2)
        @test withz.converged
        @test coef(withz) ≈ [0.64738734, 0.28322583, -0.328759, -0.23723325, -0.64870321, 0.68920955] atol = 1e-4
    end

    @testset "refusals keep their wording" begin
        d, nwk = _tsd_data("phylo_2"), _tsd_tree("phylo_2")
        @test_throws ArgumentError drm(bf(@formula(y ~ x)), Tweedie(); data = d, tree = nwk)
        err = try
            drm(bf(@formula(y ~ x + phylo(1 | g))), Tweedie(); data = d, K = _tsd_K("relmat_1"))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("`K` would be ignored", sprint(showerror, err))
        err = try
            drm(bf(@formula(y ~ x + phylo(1 | g))), Tweedie(); data = d)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("needs `tree", sprint(showerror, err))
        # shapes still refused say so in route-specific words, not as a missing SD target
        iid = merge(d, (; g2 = d.g))
        for f in (bf(@formula(y ~ x + phylo(1 | g) + (1 | g2))), bf(@formula(y ~ phylo(1 + x | g))))
            err = try; drm(f, Tweedie(); data = iid, tree = nwk); catch e; e; end
            @test err isa ArgumentError
            @test occursin("only for one `phylo(1 | g)`", sprint(showerror, err))
        end
        err = try
            drm(bf(@formula(y ~ x + phylo(1 | g)), @formula(sigma ~ x)), Tweedie(); data = d, tree = nwk)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("Tweedie phylo SD regression requires fixed `sigma ~ 1`", sprint(showerror, err))
    end
end
