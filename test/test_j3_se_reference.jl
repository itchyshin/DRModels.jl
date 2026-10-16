# test_j3_se_reference.jl -- #1073 lane J3: SEs from the exact gradient vs native drmTMB sdreport.
# One @testset per route, see j3_se_harness.jl for what each route is checked against.
include("j3_se_harness.jl")

function j3_data(route; numeric = Symbol[])
    c = j3_read_csv(route)
    return NamedTuple{Tuple(keys(c))}(Tuple(k in numeric ? j3_float(v) : v for (k, v) in c))
end

@testset "J3: Hessian from the exact gradient vs drmTMB sdreport (#1073)" begin
    @testset "Poisson phylo (sparse_laplace_glmm.jl :619)" begin
        d = j3_data("poisson_phylo"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), Poisson(); data = d,
                  tree = j3_tree("poisson_phylo"))
        j3_check("poisson_phylo", fit)
    end
    @testset "NB2 phylo (:898)" begin
        d = j3_data("nb2_phylo"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), NegBinomial2();
                  data = d, tree = j3_tree("nb2_phylo"))
        j3_check("nb2_phylo", fit)
    end
    @testset "NB2 phylo, covariate dispersion (:1091)" begin
        d = j3_data("nb2_phylo_hetero"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)), NegBinomial2();
                  data = d, tree = j3_tree("nb2_phylo_hetero"))
        j3_check("nb2_phylo_hetero", fit)
    end
    @testset "Binomial phylo (:1244)" begin
        d = j3_data("binomial_phylo"; numeric = [:x, :succ, :fail])
        fit = drm(bf(@formula(cbind(succ, fail) ~ x + phylo(1 | species))), Binomial();
                  data = d, tree = j3_tree("binomial_phylo"))
        j3_check("binomial_phylo", fit)
    end
    @testset "Poisson crossed, two intercepts (:1851)" begin
        d = j3_data("poisson_crossed2"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | a) + (1 | b))), Poisson(); data = d)
        j3_check("poisson_crossed2", fit)
    end
    @testset "Poisson crossed, three components (:2023)" begin
        d = j3_data("poisson_crossed3"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | a) + (1 | b) + (1 | c))), Poisson(); data = d)
        j3_check("poisson_crossed3", fit)
    end
    @testset "Binomial crossed (:2997)" begin
        d = j3_data("binomial_crossed"; numeric = [:x, :succ, :fail])
        fit = drm(bf(@formula(cbind(succ, fail) ~ x + (1 | a) + (1 | b))), Binomial(); data = d)
        j3_check("binomial_crossed", fit)
    end
    @testset "ordinary Laplace Poisson (ordinary_laplace.jl :193)" begin
        d = j3_data("ordinary_poisson"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = d, marginal = :Laplace)
        j3_check("ordinary_poisson", fit)
    end
    @testset "ordinary Laplace NB2 (:193, scale kernel)" begin
        d = j3_data("ordinary_nb2"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                  data = d, marginal = :Laplace)
        j3_check("ordinary_nb2", fit)
    end
    @testset "cumulative logit phylo (cumulative.jl :614)" begin
        d = j3_data("cumulative_phylo"; numeric = [:x])
        d = merge(d, (; y = parse.(Int, d.y)))
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), CumulativeLogit(); data = d,
                  tree = j3_tree("cumulative_phylo"))
        j3_check("cumulative_phylo", fit)
    end
    @testset "Gaussian phylo mean, ML (location_only.jl :3540)" begin
        d = j3_data("gaussian_phylo"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), Gaussian();
                  data = d, tree = j3_tree("gaussian_phylo"))
        j3_check("gaussian_phylo", fit)
    end
    @testset "bivariate Gaussian q=2 phylo, NO exact gradient: KNOWN-NOISY (gaussian_bivariate.jl :887)" begin
        d = j3_data("biv_phylo"; numeric = [:x, :y1, :y2])
        fit = drm(bf(mu1 = @formula(y1 ~ x + phylo(1 | species)),
                     mu2 = @formula(y2 ~ x + phylo(1 | species)),
                     sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1),
                     rho12 = @formula(rho12 ~ 1)), Gaussian();
                  data = d, tree = j3_tree("biv_phylo"), method = :ML)
        # The route is a sparse CHOLMOD objective with no gradient (ForwardDiff cannot pass
        # through it), so the Hessian stays a second difference of the VALUE. Marked
        # known-noisy: only a 10% sanity band is asserted and it is not counted as native
        # parity for the exact-gradient routes.
        j3_check("biv_phylo", fit; gradient = false, known_noisy = true)
    end
end
