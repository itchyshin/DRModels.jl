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
        C, tips = j3_cov("poisson_phylo"); sidx = [findfirst(==(s), tips) for s in d.species]
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), indicator(sidx, length(tips)),
                         obs_poisson(d.y), prior_cov(C, 3))
        j3_check("poisson_phylo", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "NB2 phylo (:898)" begin
        d = j3_data("nb2_phylo"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), NegBinomial2();
                  data = d, tree = j3_tree("nb2_phylo"))
        C, tips = j3_cov("nb2_phylo"); sidx = [findfirst(==(s), tips) for s in d.species]
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), indicator(sidx, length(tips)),
                         obs_nb2(d.y, 3), prior_cov(C, 4))
        j3_check("nb2_phylo", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "NB2 phylo, covariate dispersion (:1091)" begin
        d = j3_data("nb2_phylo_hetero"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ x)), NegBinomial2();
                  data = d, tree = j3_tree("nb2_phylo_hetero"))
        C, tips = j3_cov("nb2_phylo_hetero"); sidx = [findfirst(==(s), tips) for s in d.species]
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), indicator(sidx, length(tips)),
                         obs_nb2_hetero(d.y, d.x, 3), prior_cov(C, 5))
        j3_check("nb2_phylo_hetero", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "Binomial phylo (:1244)" begin
        d = j3_data("binomial_phylo"; numeric = [:x, :succ, :fail])
        fit = drm(bf(@formula(cbind(succ, fail) ~ x + phylo(1 | species))), Binomial();
                  data = d, tree = j3_tree("binomial_phylo"))
        C, tips = j3_cov("binomial_phylo"); sidx = [findfirst(==(s), tips) for s in d.species]
        spec = DenseSpec(2, hcat(ones(length(d.x)), d.x), indicator(sidx, length(tips)),
                         obs_binomial(d.succ, d.succ .+ d.fail), prior_cov(C, 3))
        j3_check("binomial_phylo", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "Poisson crossed, two intercepts (:1851)" begin
        d = j3_data("poisson_crossed2"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | a) + (1 | b))), Poisson(); data = d)
        ai, Ga = j3_levels(d.a); bi, Gb = j3_levels(d.b)
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), hcat(indicator(ai, Ga), indicator(bi, Gb)),
                         obs_poisson(d.y), prior_iid([Ga, Gb], [3, 4]))
        j3_check("poisson_crossed2", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "Poisson crossed, three components (:2023)" begin
        d = j3_data("poisson_crossed3"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | a) + (1 | b) + (1 | c))), Poisson(); data = d)
        ai, Ga = j3_levels(d.a); bi, Gb = j3_levels(d.b); ci, Gc = j3_levels(d.c)
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x),
                         hcat(indicator(ai, Ga), indicator(bi, Gb), indicator(ci, Gc)),
                         obs_poisson(d.y), prior_iid([Ga, Gb, Gc], [3, 4, 5]))
        j3_check("poisson_crossed3", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "Binomial crossed (:2997)" begin
        d = j3_data("binomial_crossed"; numeric = [:x, :succ, :fail])
        fit = drm(bf(@formula(cbind(succ, fail) ~ x + (1 | a) + (1 | b))), Binomial(); data = d)
        ai, Ga = j3_levels(d.a); bi, Gb = j3_levels(d.b)
        spec = DenseSpec(2, hcat(ones(length(d.x)), d.x), hcat(indicator(ai, Ga), indicator(bi, Gb)),
                         obs_binomial(d.succ, d.succ .+ d.fail), prior_iid([Ga, Gb], [3, 4]))
        j3_check("binomial_crossed", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "ordinary Laplace Poisson (ordinary_laplace.jl :193)" begin
        d = j3_data("ordinary_poisson"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | g))), Poisson(); data = d, marginal = :Laplace)
        gi, Gg = j3_levels(d.g)
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), indicator(gi, Gg), obs_poisson(d.y),
                         prior_iid([Gg], [3]))
        j3_check("ordinary_poisson", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "ordinary Laplace NB2 (:193, scale kernel)" begin
        d = j3_data("ordinary_nb2"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + (1 | g)), @formula(sigma ~ 1)), NegBinomial2();
                  data = d, marginal = :Laplace)
        gi, Gg = j3_levels(d.g)
        spec = DenseSpec(2, hcat(ones(length(d.y)), d.x), indicator(gi, Gg), obs_nb2(d.y, 3),
                         prior_iid([Gg], [4]))
        j3_check("ordinary_nb2", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "cumulative logit phylo (cumulative.jl :614)" begin
        d = j3_data("cumulative_phylo"; numeric = [:x])
        d = merge(d, (; y = parse.(Int, d.y)))
        fit = drm(bf(@formula(y ~ x + phylo(1 | species))), CumulativeLogit(); data = d,
                  tree = j3_tree("cumulative_phylo"))
        C, tips = j3_cov("cumulative_phylo"); sidx = [findfirst(==(s), tips) for s in d.species]
        # the cumulative model has NO intercept: eta = beta * x
        spec = DenseSpec(1, reshape(d.x, :, 1), indicator(sidx, length(tips)),
                         obs_cumlogit3(d.y, 2), prior_cov(C, 4))
        j3_check("cumulative_phylo", fit; dense_se = th -> dense_laplace_se(th, spec))
    end
    @testset "Gaussian phylo mean, ML (location_only.jl :3540)" begin
        d = j3_data("gaussian_phylo"; numeric = [:x, :y])
        fit = drm(bf(@formula(y ~ x + phylo(1 | species)), @formula(sigma ~ 1)), Gaussian();
                  data = d, tree = j3_tree("gaussian_phylo"))
        C, tips = j3_cov("gaussian_phylo"); sidx = [findfirst(==(s), tips) for s in d.species]
        Cobs = C[sidx, sidx]; Xg = hcat(ones(length(d.y)), d.x)
        dense_se = th -> (H = ForwardDiff.hessian(t -> dense_gaussian_phylo_nll(t, d.y, Xg, Cobs), th);
                          sqrt.(diag(inv(Symmetric((H .+ H') ./ 2)))))
        j3_check("gaussian_phylo", fit; dense_se = dense_se)
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
