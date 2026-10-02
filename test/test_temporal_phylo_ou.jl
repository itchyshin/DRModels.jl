# Paired phylo(1 | species) + temporal(1 | species, elapsed, ou) on the Gaussian
# mean (wave 2; twin of drmTMB's `phylo(1 | species, tree = tree) +
# temporal(1 | species, time = elapsed, structure = "ou")`):
#     y = Xβ + a_species + b_species(t) + ε,
#     a ~ N(0, σ_a² C)  (C the tip CORRELATION matrix of the tree),
#     b an independent stationary OU path per species, ε ~ N(0, σ² I).
# Every likelihood / mode check is against an independent dense construction:
# the tree covariance here is built from shared branch lengths while the tree
# is generated, never through DRModels' tree code.
using DRModels
using Test, Random, LinearAlgebra, StableRNGs, Statistics, Logging
using Distributions: MvNormal, logpdf

@testset "paired phylo() + temporal OU (Gaussian mean)" begin
    # Random rooted binary tree over `names`. Returns the Newick string and the
    # dense tip covariance (Cov(i, j) = summed length of shared branches),
    # accumulated as clusters are joined. `ultrametric = false` draws every
    # branch length independently, so tip depths differ.
    function rand_tree(rng, names; ultrametric = true)
        m = length(names)
        C = zeros(m, m)
        cl = [(nwk = names[i], h = 0.0, mem = [i]) for i in 1:m]
        h = 0.0
        while length(cl) > 1
            i, j = randperm(rng, length(cl))[1:2]
            a, b = cl[i], cl[j]
            h += 0.2 + rand(rng)
            la = ultrametric ? h - a.h : 0.05 + rand(rng)
            lb = ultrametric ? h - b.h : 0.05 + rand(rng)
            for (c, l) in ((a, la), (b, lb)), p in c.mem, q in c.mem
                C[p, q] += l
            end
            new = (nwk = "($(a.nwk):$(la),$(b.nwk):$(lb))", h = h, mem = [a.mem; b.mem])
            deleteat!(cl, sort([i, j]))
            push!(cl, new)
        end
        return cl[1].nwk * ";", C
    end
    corrmat(C) = (d = sqrt.(diag(C)); C ./ (d * d'))

    # θ = [β; log σ; log σ_a; log σ_t; log λ]
    function dense_V(θ, p, sp, t, Cc, lev)
        σ2 = exp(2θ[p+1]); sa2 = exp(2θ[p+2]); st2 = exp(2θ[p+3]); λ = exp(θ[p+4])
        k = Dict(l => i for (i, l) in enumerate(lev))
        n = length(sp)
        V = [sa2 * Cc[k[sp[i]], k[sp[j]]] +
             (sp[i] == sp[j] ? st2 * exp(-λ * abs(t[i] - t[j])) : 0.0) for i in 1:n, j in 1:n]
        return Symmetric(V + σ2 * I)
    end
    dense_nll(θ, y, X, sp, t, Cc, lev) =
        -logpdf(MvNormal(X * θ[1:size(X, 2)], dense_V(θ, size(X, 2), sp, t, Cc, lev)), y)

    # Irregular elapsed times, unequal series lengths (2–6), rows shuffled.
    function sim_data(rng; m = 12, β = (0.25, 0.45), σa = 0.6, σt = 0.75, σ = 0.4, λ = 0.45,
                      ultrametric = true, kmin = 2, kmax = 6)
        lev = ["sp_$(lpad(i, 3, '0'))" for i in 1:m]
        nwk, C = rand_tree(rng, lev; ultrametric)
        Cc = corrmat(C)
        a = σa .* (cholesky(Symmetric(Cc)).L * randn(rng, m))
        sp = String[]; t = Float64[]; x = Float64[]; y = Float64[]
        for s in 1:m
            k = rand(rng, kmin:kmax)
            ts = sort(10 .* rand(rng, k))
            R = [exp(-λ * abs(u - v)) for u in ts, v in ts]
            xs = randn(rng, k)
            b = σt .* (cholesky(Symmetric(R)).L * randn(rng, k))
            append!(sp, fill(lev[s], k)); append!(t, ts); append!(x, xs)
            append!(y, β[1] .+ β[2] .* xs .+ a[s] .+ b .+ σ .* randn(rng, k))
        end
        p = randperm(rng, length(y))
        return (y = y[p], x = x[p], species = sp[p], elapsed = t[p]), nwk, Cc, lev
    end
    fP = bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)),
            @formula(sigma ~ 1))
    quiet(f) = with_logger(f, NullLogger())

    @testset "dense oracle (ultrametric and non-ultrametric trees, random θ, extremes)" begin
        rng = StableRNG(311)
        worst = 0.0
        for rep in 1:3, ultra in (true, false)
            d, nwk, Cc, lev = sim_data(rng; m = 6 + 3rep, ultrametric = ultra)
            fit = quiet(() -> drm(fP, Gaussian(); data = d, tree = nwk))
            X = hcat(ones(length(d.y)), d.x)
            @test length(fit.theta) == 6
            @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.species, d.elapsed, Cc, lev) atol = 1e-10
            for _ in 1:5
                θ = [randn(rng); randn(rng); 0.5randn(rng); 0.7randn(rng); 0.5randn(rng); 1.5randn(rng)]
                e = abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.species, d.elapsed, Cc, lev))
                worst = max(worst, e)
                @test e < 1e-10
            end
            # decay extremes (σ held at 0.3 so the DENSE oracle's V stays well
            # conditioned) and the stable-SD boundaries σ_a → 0, σ_a large
            for (ia, v) in ((6, 6.0), (6, -8.0), (4, -15.0), (4, 3.0), (5, -15.0))
                θ = copy(fit.theta); θ[3] = log(0.3); θ[ia] = v
                e = abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.species, d.elapsed, Cc, lev))
                worst = max(worst, e)
                @test e < 1e-10
            end
            # decay → 0 (λ = e^-20): the OU block tends to the rank-one σ_t² 11ᵀ,
            # where the dense Cholesky itself loses digits; relative check.
            θ = copy(fit.theta); θ[3] = log(0.3); θ[6] = -20.0
            @test fit.nll(θ) ≈ dense_nll(θ, d.y, X, d.species, d.elapsed, Cc, lev) rtol = 1e-10
        end
        println("paired phylo + OU dense oracle: worst |Δnll| = ", worst)
    end

    @testset "conditional modes against the dense model" begin
        d, nwk, Cc, lev = sim_data(StableRNG(312); m = 15, ultrametric = false)
        fit = quiet(() -> drm(fP, Gaussian(); data = d, tree = nwk))
        n = length(d.y); X = hcat(ones(n), d.x)
        θ = fit.theta
        sa2 = exp(2θ[4]); st2 = exp(2θ[5]); λ = exp(θ[6])
        w = dense_V(θ, 2, d.species, d.elapsed, Cc, lev) \ (d.y - X * coef(fit, :mu))
        R = [d.species[i] == d.species[j] ? exp(-λ * abs(d.elapsed[i] - d.elapsed[j])) : 0.0
             for i in 1:n, j in 1:n]
        @test ranef(fit)[:species] ≈ st2 * R * w atol = 1e-8
        k = Dict(l => i for (i, l) in enumerate(lev))
        Z = [k[d.species[i]] == s ? 1.0 : 0.0 for i in 1:n, s in eachindex(lev)]
        a_all = sa2 * Cc * (Z' * w)                       # tree order (lev)
        series = unique(d.species)                        # first-seen order
        @test ranef(fit)[:species_phylo] ≈ a_all[[k[s] for s in series]] atol = 1e-8
    end

    @testset "recovery smoke (m = 150 species, 4–6 times each)" begin
        d, nwk, _, _ = sim_data(StableRNG(2031); m = 150, kmin = 4, kmax = 6)
        fit = drm(fP, Gaussian(); data = d, tree = nwk)
        tp = temporal_parameters(fit)
        println("paired phylo + OU recovery (m = 150): sd_phylo = ", tp.sd_phylo, ", sd = ", tp.sd,
                ", decay = ", tp.decay, ", sigma = ", tp.sigma, ", beta = ", coef(fit, :mu))
        @test fit.converged
        @test tp.sd_phylo ≈ 0.6 atol = 0.3
        @test tp.sd ≈ 0.75 atol = 0.2
        @test tp.decay ≈ 0.45 atol = 0.3
        @test tp.sigma ≈ 0.4 atol = 0.15
        @test coef(fit, :mu)[2] ≈ 0.45 atol = 0.08
    end

    @testset "invariances" begin
        d, nwk, _, _ = sim_data(StableRNG(313); m = 14, ultrametric = false)
        fit = quiet(() -> drm(fP, Gaussian(); data = d, tree = nwk))
        # row order
        p = reverse(eachindex(d.y))
        @test quiet(() -> drm(fP, Gaussian(); data = map(c -> c[p], d), tree = nwk)).nll(fit.theta) ≈
              fit.nll(fit.theta) atol = 1e-9
        # rescaling every branch length leaves the tip CORRELATION, so the
        # objective, unchanged
        nwk3 = replace(nwk, r":([0-9.eE+-]+)" => s -> ":" * string(3.7 * parse(Float64, s[2:end])))
        @test quiet(() -> drm(fP, Gaussian(); data = d, tree = nwk3)).nll(fit.theta) ≈
              fit.nll(fit.theta) atol = 1e-9
        # an AugmentedPhy works as well as the Newick string
        @test quiet(() -> drm(fP, Gaussian(); data = d, tree = augmented_phy(nwk))).nll(fit.theta) ≈
              fit.nll(fit.theta) atol = 1e-12
        # time-origin shift
        d2 = merge(d, (elapsed = d.elapsed .+ 500.25,))
        @test quiet(() -> drm(fP, Gaussian(); data = d2, tree = nwk)).nll(fit.theta) ≈
              fit.nll(fit.theta) atol = 1e-9
    end

    @testset "accessors and labels" begin
        d, nwk, _, _ = sim_data(StableRNG(314); m = 20)
        fit = drm(fP, Gaussian(); data = d, tree = nwk)
        n = length(d.y); X = hcat(ones(n), d.x)
        @test first.(fit.blocks) == [:mu, :sigma, :resd, :temporal_decay]
        @test Dict(fit.coefnames)[:resd] == ["species_phylo", "species"]
        @test Dict(fit.coefnames)[:temporal_decay] == ["decay_temporal"]
        @test fit.phylo_scale === :correlation
        tp = temporal_parameters(fit)
        @test tp.structure === :ou && tp.phi === nothing && tp.sd_iid === nothing
        @test tp.sd_phylo ≈ re_sd(fit)[:species_phylo] && tp.sd ≈ re_sd(fit)[:species]
        @test tp.decay ≈ exp(only(coef(fit, :temporal_decay)))
        @test predict(fit, d) ≈ fitted(fit) ≈ X * coef(fit, :mu)
        pp = predict_parameters(fit, d)
        @test pp[:mu] ≈ fitted(fit) && all(≈(tp.sigma), pp[:sigma])
        @test Set(structured_effects(fit)) ==
              Set([(dpar = :mu, kind = :phylo, grouping = :species),
                   (dpar = :mu, kind = :temporal, grouping = :species)])
        @test length(coeftable(fit).rownms) == 6
        @test isfinite(aic(fit)) && dof(fit) == 6
        @test any(r -> r.param === :temporal_decay && r.scale === :log, profile_targets(fit))
        @test length(ranef(fit)[:species_phylo]) == 20
        # read-only callers run on the paired fit
        @test check_drm(fit) !== nothing
        @test sprint(show, MIME("text/plain"), fit) isa String
        @test residuals(fit) ≈ d.y .- fitted(fit)
        pr = confint(fit; method = :profile, parm = :mu => "x")
        @test only(pr).lower < coef(fit, :mu)[2] < only(pr).upper
        # temporal_parameters on a wave-1 fit reports no phylo SD
        f1 = drm(bf(@formula(y ~ x + temporal(1 | species, elapsed, ou)), @formula(sigma ~ 1)),
                 Gaussian(); data = d)
        @test temporal_parameters(f1).sd_phylo === nothing
    end

    @testset "simulate draws the full model (fresh phylo field + fresh OU + noise)" begin
        d, nwk, Cc, lev = sim_data(StableRNG(315); m = 40, kmin = 3, kmax = 5)
        fit = drm(fP, Gaussian(); data = d, tree = nwk)
        tp = temporal_parameters(fit)
        E = simulate(fit; nsim = 3000, rng = StableRNG(316)) .- fitted(fit)
        n = length(d.y)
        k = Dict(l => i for (i, l) in enumerate(lev))
        Vm = [tp.sd_phylo^2 * Cc[k[d.species[i]], k[d.species[j]]] +
              (d.species[i] == d.species[j] ? tp.sd^2 * exp(-tp.decay * abs(d.elapsed[i] - d.elapsed[j])) : 0.0) +
              (i == j ? tp.sigma^2 : 0.0) for i in 1:n, j in 1:n]
        Ve = (E * E') ./ size(E, 2)
        @test mean(diag(Ve)) ≈ mean(diag(Vm)) rtol = 0.03
        # between-species pairs carry ONLY the phylogenetic covariance
        bt = [(i, j) for i in 1:n, j in 1:n if i < j && d.species[i] != d.species[j]]
        @test cor([Ve[i, j] for (i, j) in bt], [Vm[i, j] for (i, j) in bt]) > 0.9
        @test mean(Ve[i, j] for (i, j) in bt) ≈ mean(Vm[i, j] for (i, j) in bt) atol = 0.02
        # within-species pairs add the OU covariance
        wt = [(i, j) for i in 1:n, j in 1:n if i < j && d.species[i] == d.species[j]]
        @test mean(Ve[i, j] for (i, j) in wt) ≈ mean(Vm[i, j] for (i, j) in wt) atol = 0.03
        # the parametric bootstrap uses the same draws and refits with the tree
        bc = quiet(() -> bootstrap_ci(fit; data = d, tree = nwk, B = 6, rng = StableRNG(317)))
        @test length(bc) == 6 && all(r -> isfinite(r.lower) && isfinite(r.upper), bc)
    end

    @testset "refusals (drmTMB's paired-provider rules)" begin
        d, nwk, _, lev = sim_data(StableRNG(318); m = 8, kmin = 3, kmax = 5)
        ae = ArgumentError
        g(f; data = d, tree = nwk, kw...) = drm(f, Gaussian(); data = data, tree = tree, kw...)
        msg(f; kw...) = try
            g(f; kw...); ""
        catch e
            e isa ArgumentError ? e.msg : rethrow()
        end
        fA = bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, occ, ar1)), @formula(sigma ~ 1))
        dA = merge(d, (occ = round.(Int, d.elapsed .* 10),))
        @test occursin("requires structure `ou`", msg(fA; data = dA))
        @test occursin("same grouping", msg(bf(@formula(y ~ x + phylo(1 | species) +
            temporal(1 | other, elapsed, ou)), @formula(sigma ~ 1)); data = merge(d, (other = d.species,))))
        @test occursin("does not allow an ordinary", msg(bf(@formula(y ~ x + (1 | species) +
            phylo(1 | species) + temporal(1 | species, elapsed, ou)), @formula(sigma ~ 1))))
        @test occursin("unlabelled", msg(bf(@formula(y ~ x + phylo(1 + x | species) +
            temporal(1 | species, elapsed, ou)), @formula(sigma ~ 1))))
        @test occursin("needs the tree", msg(fP; tree = nothing))
        # fewer than three species
        two = map(c -> c[in.(d.species, Ref(lev[1:2]))], d)
        @test occursin("at least three observed species", msg(fP; data = two))
        # a species with a single observation
        i1 = findall(==(lev[1]), d.species)
        one = map(c -> c[setdiff(eachindex(d.y), i1[2:end])], d)
        @test occursin("at least two distinct times", msg(fP; data = one))
        # only two distinct lags across the data
        lagd = (y = repeat(d.y[1:3], length(lev)), x = repeat(d.x[1:3], length(lev)),
                species = repeat(lev, inner = 3), elapsed = repeat([0.0, 1.0, 2.0], length(lev)))
        @test occursin("three distinct positive lags", msg(fP; data = lagd))
        # tree tips must match the observed species
        sub = map(c -> c[d.species .!= lev[end]], d)
        @test occursin("tips to match", msg(fP; data = sub))
        @test occursin("tips to match", msg(fP; tree = replace(nwk, lev[1] * ":" => "nobody:")))
        # still refused: other structured partners, AR1-free scope rules, REML
        @test_throws ae g(bf(@formula(y ~ x + relmat(1 | species) + temporal(1 | species, elapsed, ou)),
                             @formula(sigma ~ 1)); K = Matrix(1.0I, 8, 8))
        @test_throws ae g(bf(@formula(y ~ x + phylo(1 | species) + relmat(1 | species) +
                                 temporal(1 | species, elapsed, ou)), @formula(sigma ~ 1)); K = Matrix(1.0I, 8, 8))
        @test_throws ae g(fP; method = :REML)
        @test_throws ae g(bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)),
                             @formula(sigma ~ x)))
        @test_throws ae g(bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)),
                             @formula(sigma ~ 1 + phylo(1 | species))))
        miss = merge(d, (y = Union{Missing,Float64}[i == 1 ? missing : v for (i, v) in enumerate(d.y)],))
        @test_throws ae g(fP; data = miss)
        @test_throws ae g(fP; algorithm = :sparse)
        @test_throws ae g(fP; penalty = drm_phylo_penalty())
    end

    @testset "R bridge spelling (drmTMB keyword form)" begin
        d, nwk, _, _ = sim_data(StableRNG(319); m = 10)
        ref = loglik(drm(fP, Gaussian(); data = d, tree = nwk))
        # drmTMB's R side strips `tree = tree` from `phylo()` and ships the tree
        # as Newick (`drm_julia_formula_entry`); the temporal keywords arrive as
        # written and are translated here.
        out = drm_bridge(; formula = "y ~ x + phylo(1 | species) + temporal(1 | species, " *
                         "time = elapsed, structure = \"ou\"); sigma ~ 1",
                         family = "gaussian", data = d, tree = nwk)
        @test out["loglik"] ≈ ref atol = 1e-8
    end
end
