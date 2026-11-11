# Owner answer P15 = b: the group covariance of the bivariate ordinary `(1 | p | g)` route is
# reported in a block named `_BIV_RE_GROUP_BLOCK` (`:groupcov`), not `:phylocov`. The truly
# phylogenetic bivariate routes keep `:phylocov`. Numbers are unchanged on both routes: the
# reference values below were produced by the commit before the rename (24b094726).
using DRModels
using Test, LinearAlgebra, Random

include("biv_ordinary_re_helpers.jl")

# From the commit before the rename (24b094726): interior_bal.csv, ML. (The phylogenetic fixture draws different data on Julia 1.10 and 1.13, so its numbers
# were compared against the old commit by a side-by-side run, not pinned here.)
const P15_REF_LL = -926.32736589695389
const P15_REF_THETA = [0.21367143612936837, 0.41118575196663942, -0.26076347480572376, 0.39637596654108809,
                       0.080990566737647998, -0.10734089153920659, 0.38838823233940678,
                       -0.46723255056092838, 0.040625883655253013, -0.80916777869774115]

_p15_hasname(s, nm) = occursin(string(nm), s)

@testset "bivariate ordinary (1|p|g): group covariance block is :groupcov, not :phylocov (P15)" begin
    @test DRModels._BIV_RE_GROUP_BLOCK === :groupcov
    d = _bre_data("interior_bal.csv")
    for method in (:ML, :REML)
        fit = drm(_BRE_F0, Gaussian(); data = d, method = method)
        blk = DRModels._BIV_RE_GROUP_BLOCK
        @test blk in first.(fit.blocks)
        @test !(:phylocov in first.(fit.blocks))
        @test Dict(fit.coefnames)[blk] == ["Sigma_g:L11", "Sigma_g:L21", "Sigma_g:L22"]
        @test !haskey(Dict(fit.coefnames), :phylocov)
        @test length(Dict(fit.blocks)[blk]) == 3
        @test DRModels._variance_component_blocks(fit) == [blk]

        shown = sprint(show, MIME"text/plain"(), fit)
        @test !_p15_hasname(shown, :phylocov) && !occursin("Σ_a", shown)
        @test occursin("Group-level covariance Σ_g", shown)
        @test !occursin("uncalibrated", shown)         # that wording is the q4 phylogenetic finding
        short = sprint(show, fit)
        @test _p15_hasname(short, blk) && !_p15_hasname(short, :phylocov)
        sm = sprint(show, MIME"text/plain"(), summary(fit))
        @test _p15_hasname(sm, blk) && !_p15_hasname(sm, :phylocov)
        ct = coeftable(fit)
        @test "$(blk): Sigma_g:L11" in ct.rownms
        @test !any(occursin("phylocov", r) for r in ct.rownms)
        ci = confint(fit)
        ctxt = sprint(show, ci)
        @test !_p15_hasname(ctxt, :phylocov)
        @test count(r -> r.param === blk, ci) == 3
        # a block selector by the new name works; the old name is refused
        cg = confint(fit; parm = blk)
        @test length(cg) == 3 && all(r.param === blk for r in cg)
        @test length(confint(fit; parm = blk => "Sigma_g:L11")) == 1
        old = try confint(fit; parm = :phylocov) catch; nothing end
        @test old === nothing || isempty(old)
        @test length(coef(fit, blk)) == 3
        @test_throws Exception coef(fit, :phylocov)
        @test !(blk in keys(predict_parameters(fit, d)))
        @test !any(r -> r.param === :phylocov, DRModels.profile_targets(fit))
        @test any(r -> r.param === blk, DRModels.profile_targets(fit))
    end

    # numbers unchanged against the commit before the rename (ML, interior_bal)
    fit = drm(_BRE_F0, Gaussian(); data = d)
    @test fit.loglik ≈ P15_REF_LL rtol = 1e-9
    @test fit.theta ≈ P15_REF_THETA rtol = 1e-7 atol = 1e-9
end

@testset "bivariate phylogenetic q=2 keeps :phylocov (P15)" begin
    rng = MersenneTwister(20260625)
    phy = DRModels.random_balanced_tree(14; branch_length = 0.2)
    beta = [0.20 -0.15; 0.25 0.10]
    Lambda = Matrix(Symmetric([0.22 0.07; 0.07 0.18]))
    rc = Matrix(Symmetric([0.12 0.04; 0.04 0.16]))
    Q_cond, leaf_pos, _ = DRModels.augmented_tree_precision(phy)
    P = DRModels.prior_precision(Q_cond, inv(Lambda))
    u = cholesky(Symmetric(P)).UP \ randn(rng, size(P, 1))
    species = repeat(1:phy.n_leaves, inner = 3); n = length(species)
    x = randn(rng, n); X = hcat(ones(n), x)
    L = cholesky(Symmetric(rc)).L; Y = zeros(n, 2)
    for i in 1:n
        b = 2 * (leaf_pos[species[i]] - 1)
        for a in 1:2
            Y[i, a] = sum(X[i, :] .* beta[:, a]) + u[b + a]
        end
        Y[i, :] .+= L * randn(rng, 2)
    end
    dat = (; y1 = Y[:, 1], y2 = Y[:, 2], x, species_name = [phy.leaf_names[k] for k in species])
    form = bf(mu1 = @formula(y1 ~ x + phylo(1 | species_name)), mu2 = @formula(y2 ~ x + phylo(1 | species_name)),
              sigma1 = @formula(sigma1 ~ 1), sigma2 = @formula(sigma2 ~ 1), rho12 = @formula(rho12 ~ 1))
    fit = drm(form, Gaussian(); data = dat, tree = phy, g_tol = 1e-4)
    @test :phylocov in first.(fit.blocks)
    @test !(DRModels._BIV_RE_GROUP_BLOCK in first.(fit.blocks))
    @test Dict(fit.coefnames)[:phylocov] == ["Sigma_a:L11", "Sigma_a:L21", "Sigma_a:L22"]
    shown = sprint(show, MIME"text/plain"(), fit)
    @test occursin("phylocov", shown) || occursin("Group-level covariance Σ_a", shown)
    @test !occursin("Σ_g", shown)
    @test DRModels._variance_component_blocks(fit) == [:phylocov]
end
