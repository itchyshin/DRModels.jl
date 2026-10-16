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
end
