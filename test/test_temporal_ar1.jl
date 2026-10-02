# temporal(1 | id, time, ar1) on the Gaussian mean (D-310; twin of drmTMB's
# `temporal(1 | id, time = occ, structure = "ar1")`). The engine evaluates the
# exact marginal per series with tridiagonal algebra (src/temporal.jl); every
# check here is against an independent DENSE construction of the same model.
using DRModels
using Test, Random, LinearAlgebra, DelimitedFiles, StableRNGs, Statistics, Logging
using Distributions: MvNormal, Normal, logpdf

@testset "temporal AR1 (Gaussian mean)" begin
    # Dense oracle: −logpdf(MvNormal(Xβ, σ_t² R + σ_b² J + σ² I)), R_ij = φ^|t_i − t_j|
    # within a series, θ = [β; log σ; (log σ_b); log σ_t; atanh φ].
    function dense_nll(θ, y, X, id, t; ordinary = false)
        p = size(X, 2); n = length(y)
        σ2 = exp(2θ[p+1])
        sb2 = ordinary ? exp(2θ[p+2]) : 0.0
        st2 = exp(2θ[p + (ordinary ? 3 : 2)])
        φ = tanh(θ[p + (ordinary ? 4 : 3)])
        V = zeros(n, n)
        for i in 1:n, j in 1:n
            id[i] == id[j] && (V[i, j] = st2 * φ^Int(abs(t[i] - t[j])) + sb2)
        end
        V += σ2 * I
        return -logpdf(MvNormal(X * θ[1:p], Symmetric(V)), y)
    end
    # Unequal-length series on gapped integer occasions, rows shuffled.
    function sim_data(rng; S = 12, φ = 0.6, st = 0.8, σ = 0.5, sb = 0.0, origin = 0)
        id = String[]; t = Int[]; x = Float64[]; y = Float64[]
        for s in 1:S
            k = rand(rng, 2:7)
            ts = sort(collect(0:14)[sortperm(rand(rng, 15))[1:k]]) .+ origin
            R = [φ^abs(a - b) for a in ts, b in ts]
            xs = randn(rng, k)
            a = st .* (cholesky(Symmetric(R)).L * randn(rng, k))
            append!(id, fill("s$s", k)); append!(t, ts); append!(x, xs)
            append!(y, 1.0 .+ 0.5 .* xs .+ a .+ sb * randn(rng) .+ σ .* randn(rng, k))
        end
        p = sortperm(rand(rng, length(y)))
        return (y = y[p], x = x[p], id = id[p], occ = t[p])
    end
    fAR = bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1))
    fARi = bf(@formula(y ~ x + (1 | id) + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1))

    @testset "dense oracle to 1e-10 (gapped, unequal series, ± ordinary intercept)" begin
        rng = StableRNG(11)
        for rep in 1:3, ordinary in (false, true)
            d = sim_data(rng; S = 8 + 3rep, sb = ordinary ? 0.4 : 0.0)
            fit = with_logger(NullLogger()) do      # small fits may sit at a σ boundary; only the objective is used
                drm(ordinary ? fARi : fAR, Gaussian(); data = d)
            end
            X = hcat(ones(length(d.y)), d.x)
            np = length(fit.theta)
            @test np == (ordinary ? 6 : 5)
            for _ in 1:4
                θ = [randn(rng); randn(rng); 0.5randn(rng); 0.5 .* randn(rng, np - 4); 1.2randn(rng)]
                @test abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.id, d.occ; ordinary)) < 1e-10
            end
            # negative persistence and near-unit persistence
            for ψ in (-1.1, 2.5)
                θ = copy(fit.theta); θ[end] = ψ
                @test abs(fit.nll(θ) - dense_nll(θ, d.y, X, d.id, d.occ; ordinary)) < 1e-10
            end
            @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.id, d.occ; ordinary) atol = 1e-10
        end
    end

    @testset "fit recovery (S = 800 series, φ = 0.6, σ_t = 0.8, σ = 0.5)" begin
        # Monte Carlo check of this DGP at S = 300 (40 replicates, Totoro, 2026-10-01):
        # mean φ̂ = 0.596 (sd 0.071), mean σ̂_t = 0.808 (sd 0.073), median σ̂ = 0.510 —
        # unbiased. At S = 800 the sampling sd is ≈ 0.045, so 0.15 is a > 3-sd band.
        d = sim_data(StableRNG(2026); S = 800)
        fit = drm(fAR, Gaussian(); data = d)
        tp = temporal_parameters(fit)
        println("temporal AR1 recovery (S = 800): phi = ", tp.phi, ", sd = ", tp.sd, ", sigma = ", tp.sigma,
                ", beta = ", coef(fit, :mu))
        @test fit.converged
        @test tp.structure === :ar1 && tp.decay === nothing && tp.sd_iid === nothing
        @test tp.phi ≈ 0.6 atol = 0.15
        @test tp.sd ≈ 0.8 atol = 0.15
        @test tp.sigma ≈ 0.5 atol = 0.15
        @test coef(fit, :mu) ≈ [1.0, 0.5] atol = 0.1
        @test all(isfinite, stderror(fit))
    end

    @testset "negative persistence is recovered (sign identified by odd lags)" begin
        d = sim_data(StableRNG(7); S = 300, φ = -0.5)
        tp = temporal_parameters(drm(fAR, Gaussian(); data = d))
        @test tp.phi ≈ -0.5 atol = 0.15
    end

    @testset "ordinary (1 | id) + AR1" begin
        d = sim_data(StableRNG(99); S = 300, sb = 0.5)
        fit = drm(fARi, Gaussian(); data = d)
        tp = temporal_parameters(fit)
        @test tp.sd_iid ≈ 0.5 atol = 0.15
        @test tp.phi ≈ 0.6 atol = 0.2
        @test Set(keys(re_sd(fit))) == Set([:id, :id_iid])
        @test length(ranef(fit)[:id_iid]) == 300
    end

    @testset "invariances" begin
        d = sim_data(StableRNG(5); S = 15)
        fit = drm(fAR, Gaussian(); data = d)
        # row order does not matter
        p = reverse(1:length(d.y))
        d2 = map(c -> c[p], d)
        @test loglik(drm(fAR, Gaussian(); data = d2)) ≈ loglik(fit) atol = 1e-6
        # only gaps matter: shifting the integer origin changes nothing
        d3 = merge(d, (occ = d.occ .+ 1000,))
        @test fit.nll(fit.theta) ≈ drm(fAR, Gaussian(); data = d3).nll(fit.theta) atol = 1e-10
        # φ → 0: the temporal effect is iid N(0, σ_t²) per row, so the marginal
        # is N(Xβ, (σ² + σ_t²) I) — the fixed-effect Gaussian likelihood.
        θ = copy(fit.theta); θ[end] = 0.0
        X = hcat(ones(length(d.y)), d.x)
        s = sqrt(exp(2θ[3]) + exp(2θ[4]))
        @test fit.nll(θ) ≈ -sum(logpdf.(Normal.(X * θ[1:2], s), d.y)) atol = 1e-10
    end

    @testset "accessors" begin
        d = sim_data(StableRNG(31); S = 20)
        fit = drm(fAR, Gaussian(); data = d)
        n = length(d.y)
        X = hcat(ones(n), d.x)
        lbl = "temporal(1 | id, time = occ, structure = \"ar1\")"
        @test first.(fit.blocks) == [:mu, :sigma, :resd, :temporal_phi]
        @test Dict(fit.coefnames)[:temporal_phi] == [lbl]
        @test temporal_parameters(fit).label == lbl
        @test fitted(fit) ≈ X * coef(fit, :mu)
        @test predict(fit, d) ≈ fitted(fit)
        @test predict(fit, (x = [0.0, 1.0],)) ≈ coef(fit, :mu)[1] .+ [0.0, 1.0] .* coef(fit, :mu)[2]
        @test re_sd(fit)[:id] ≈ temporal_parameters(fit).sd
        @test vc(fit)[:id][1, 1] ≈ temporal_parameters(fit).sd^2
        @test structured_effects(fit) == [(dpar = :mu, kind = :temporal, grouping = :id)]
        # conditional temporal effects (row order) = σ_t² R V⁻¹ r, from the dense model
        st2 = exp(2fit.theta[4]); φ = tanh(fit.theta[5])
        R = [d.id[i] == d.id[j] ? φ^abs(d.occ[i] - d.occ[j]) : 0.0 for i in 1:n, j in 1:n]
        V = st2 * R + exp(2fit.theta[3]) * I
        @test ranef(fit)[:id] ≈ st2 * R * (V \ (d.y - X * coef(fit, :mu))) atol = 1e-8
        tg = profile_targets(fit)
        @test any(r -> r.param === :temporal_phi && r.scale === :atanh, tg)
        @test length(simulate(fit)) == n
        ct = coeftable(fit)
        @test length(ct.rownms) == 5
        @test isfinite(aic(fit)) && dof(fit) == 5
    end

    @testset "refusals" begin
        d = sim_data(StableRNG(3); S = 10)
        g(f; kw...) = drm(f, Gaussian(); data = d, kw...)
        ae = ArgumentError
        @test_throws ae g(fAR; method = :REML)
        @test_throws ae g(bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ x)))
        @test_throws ae g(bf(@formula(y ~ x), @formula(sigma ~ 1 + temporal(1 | id, occ, ar1))))
        @test_throws ae g(bf(@formula(y ~ x + temporal(x | id, occ, ar1)), @formula(sigma ~ 1)))
        @test_throws ae g(bf(@formula(y ~ x + temporal(1 | id, occ, ar2)), @formula(sigma ~ 1)))
        @test_throws ae g(bf(@formula(y ~ x + temporal(1 | id, occ)), @formula(sigma ~ 1)))
        @test_throws ae g(bf(@formula(y ~ x + temporal(1 | id, occ, ar1) + temporal(1 | id, occ, ou)),
                             @formula(sigma ~ 1)))
        @test_throws ae g(bf(@formula(y ~ x + (1 | id) + (1 | x) + temporal(1 | id, occ, ar1)),
                             @formula(sigma ~ 1)))
        d2 = merge(d, (grp2 = [i % 3 for i in 1:length(d.y)],))
        @test_throws ae drm(bf(@formula(y ~ x + (1 | grp2) + temporal(1 | id, occ, ar1)),
                               @formula(sigma ~ 1)), Gaussian(); data = d2)
        @test_throws ae drm(bf(@formula(y ~ x + (0 + x | id) + temporal(1 | id, occ, ar1)),
                               @formula(sigma ~ 1)), Gaussian(); data = d)
        @test_throws ae g(bf(@formula(y ~ x + relmat(1 | id) + temporal(1 | id, occ, ar1)),
                             @formula(sigma ~ 1)); K = Matrix(1.0I, 10, 10))
        @test_throws ae g(fAR; algorithm = :sparse)
        @test_throws ae g(fAR; marginal = :Laplace)
        @test_throws ae drm(fAR, Student(); data = d)
        @test_throws ae drm(bf(@formula(y ~ x + temporal(1 | id, occ, ar1))), DRModels.Poisson();
                            data = merge(d, (y = round.(abs.(d.y)),)))
        # data checks (drmTMB's)
        @test_throws ae drm(fAR, Gaussian(); data = merge(d, (occ = d.occ .+ 0.5 .* (d.occ .% 2),)))   # fractional AR1 time
        dup = (y = [d.y; d.y[1]], x = [d.x; d.x[1]], id = [d.id; d.id[1]], occ = [d.occ; d.occ[1]])
        @test_throws ae drm(fAR, Gaussian(); data = dup)                                # duplicate (id, time)
        even = merge(d, (occ = 2 .* d.occ,))
        @test_throws ae drm(fAR, Gaussian(); data = even)                               # no odd lag
        miss = merge(d, (y = Union{Missing,Float64}[i == 1 ? missing : v for (i, v) in enumerate(d.y)],))
        @test_throws ae drm(fAR, Gaussian(); data = miss)                               # missing response
        @test_throws ae drm(fAR, Gaussian(); data = merge(d, (occ = string.(d.occ),)))  # non-numeric time
        @test_throws ae temporal_parameters(drm(bf(@formula(y ~ x), @formula(sigma ~ 1)),
                                                Gaussian(); data = d))
    end

    @testset "R bridge spelling (drmTMB keyword form)" begin
        bt = DRModels._bridge_temporal_expr
        @test bt(Meta.parse("temporal(1 | id, time = occ, structure = \"ar1\")")) ==
              :(temporal(1 | id, occ, ar1))
        @test bt(Meta.parse("temporal(1 | id, structure = \"ou\", time = t)")) ==
              :(temporal(1 | id, t, ou))
        @test_throws ArgumentError bt(Meta.parse("temporal(1 | id, time = occ)"))
        @test_throws ArgumentError bt(Meta.parse("temporal(1 | id, time = occ, structure = \"ar2\")"))
        @test_throws ArgumentError bt(Meta.parse("temporal(1 | id, time = occ, structure = ar1)"))
        @test_throws ArgumentError bt(Meta.parse("temporal(1 | id, occ, structure = \"ar1\")"))
        d = sim_data(StableRNG(41); S = 12)
        out = drm_bridge(; formula = "y ~ x + temporal(1 | id, time = occ, structure = \"ar1\"); sigma ~ 1",
                         family = "gaussian", data = d)
        @test out["loglik"] ≈ loglik(drm(fAR, Gaussian(); data = d)) atol = 1e-8
    end

    @testset "fixture ar1_gapped.csv" begin
        raw, hdr = readdlm(joinpath(@__DIR__, "fixtures", "temporal", "ar1_gapped.csv"), ','; header = true)
        col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
        d = (y = Float64.(col("y")), x = Float64.(col("x")), id = String.(col("id")), occ = Int.(col("occ")))
        @test length(d.y) > 100 && length(unique(d.id)) == 24
        fit = drm(fAR, Gaussian(); data = d)
        @test fit.converged
        X = hcat(ones(length(d.y)), d.x)
        @test loglik(fit) ≈ -dense_nll(fit.theta, d.y, X, d.id, d.occ) atol = 1e-10
        @test -1 < temporal_parameters(fit).phi < 1
    end
end
