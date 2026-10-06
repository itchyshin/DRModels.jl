# Standardised residuals of temporal() fits (`residuals(fit; type = :quantile)`).
#
# AR1 / OU (± `(1 | id)`) and the paired phylo() + OU fit: drmTMB's Pearson and
# quantile residuals are CONDITIONAL on the fitted modes, (y − Xβ̂ − modes)/σ̂.
# For a Gaussian model with marginal covariance V = Σ_signal + σ²I and
# r = y − Xβ̂, the modes are Σ_signal V⁻¹ r, so the conditional residual is
#     (r − Σ_signal V⁻¹ r)/σ = σ V⁻¹ r,
# which is checked here against a DENSE V built independently of the filter.
# Homogeneous Toeplitz keeps its whitened L⁻¹ r (checked in
# test_temporal_homtoep.jl); here only that it is not routed through the
# conditional path.
using DRModels
using Test, Random, LinearAlgebra, StableRNGs, Logging

@testset "temporal residuals (conditional on the modes, as drmTMB)" begin
    quiet(f) = with_logger(f, NullLogger())

    # Unequal-length series, rows shuffled. `kind = :ar1` uses gapped integer
    # occasions, `:ou` irregular real times.
    function sim_data(rng, kind; S = 40, st = 0.7, σ = 0.45, sb = 0.0)
        id = String[]; t = Float64[]; x = Float64[]; y = Float64[]
        for s in 1:S
            k = rand(rng, 2:7)
            ts = kind === :ar1 ? Float64.(sort(randperm(rng, 15)[1:k])) : sort(10 .* rand(rng, k))
            R = kind === :ar1 ? [0.6^abs(a - b) for a in ts, b in ts] :
                                [exp(-0.45 * abs(a - b)) for a in ts, b in ts]
            xs = randn(rng, k)
            a = st .* (cholesky(Symmetric(R)).L * randn(rng, k))
            append!(id, fill("s$s", k)); append!(t, ts); append!(x, xs)
            append!(y, 0.8 .+ 0.4 .* xs .+ a .+ sb * randn(rng) .+ σ .* randn(rng, k))
        end
        p = randperm(rng, length(y))
        tt = kind === :ar1 ? Int.(t[p]) : t[p]
        return (y = y[p], x = x[p], id = id[p], occ = tt)
    end
    # Dense V at θ = [β; log σ; (log σ_b); log σ_t; atanh φ | log λ].
    function dense_V(θ, p, kind, id, t; ordinary)
        σ2 = exp(2θ[p+1])
        sb2 = ordinary ? exp(2θ[p+2]) : 0.0
        st2 = exp(2θ[p + (ordinary ? 3 : 2)])
        ψ = θ[p + (ordinary ? 4 : 3)]
        n = length(id)
        V = zeros(n, n)
        for i in 1:n, j in 1:n
            id[i] == id[j] || continue
            c = kind === :ar1 ? tanh(ψ)^Int(abs(t[i] - t[j])) : exp(-exp(ψ) * abs(t[i] - t[j]))
            V[i, j] = st2 * c + sb2
        end
        return Symmetric(V + σ2 * I)
    end

    forms = Dict(
        (:ar1, false) => bf(@formula(y ~ x + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1)),
        (:ar1, true) => bf(@formula(y ~ x + (1 | id) + temporal(1 | id, occ, ar1)), @formula(sigma ~ 1)),
        (:ou, false) => bf(@formula(y ~ x + temporal(1 | id, occ, ou)), @formula(sigma ~ 1)),
        (:ou, true) => bf(@formula(y ~ x + (1 | id) + temporal(1 | id, occ, ou)), @formula(sigma ~ 1)),
    )

    @testset "$kind, (1 | id) = $ordinary: σ̂ V⁻¹ r from the dense model" for kind in (:ar1, :ou),
                                                                           ordinary in (false, true)
        rng = StableRNG(kind === :ar1 ? 919 : 920)
        d = sim_data(rng, kind; sb = ordinary ? 0.4 : 0.0)
        fit = quiet(() -> drm(forms[(kind, ordinary)], Gaussian(); data = d))
        n = length(d.y); X = hcat(ones(n), d.x)
        r = d.y - X * coef(fit, :mu)
        σ = exp(only(coef(fit, :sigma)))
        @test σ > 0.2                      # an interior fit, not a σ → 0 boundary
        dense = σ .* (dense_V(fit.theta, 2, kind, d.id, d.occ; ordinary) \ r)
        res = residuals(fit; type = :quantile)
        @test res ≈ dense atol = 1e-8
        # the same thing written as drmTMB defines it: y minus every fitted mode
        re = ranef(fit)
        cond = fitted(fit) .+ re[:id]
        ordinary && (cond = cond .+ re[:id_iid][indexin(d.id, unique(d.id))])
        @test res ≈ (d.y .- cond) ./ σ atol = 1e-12
        # not the marginal standardisation the route used to return
        @test maximum(abs.(res .- r ./ σ)) > 0.1
        # `type = :response` is unchanged: population-level y − Xβ̂
        @test residuals(fit) ≈ r
        # deterministic: no randomisation for this continuous family
        @test residuals(fit; type = :quantile, rng = StableRNG(1)) == res
    end

    @testset "paired phylo() + OU: σ̂ V⁻¹ r with the tip-correlation field" begin
        # Ultrametric random tree over m species; C the tip covariance.
        rng = StableRNG(9190)
        m = 12
        lev = ["sp$(lpad(i, 2, '0'))" for i in 1:m]
        C = zeros(m, m)
        cl = [(nwk = lev[i], h = 0.0, mem = [i]) for i in 1:m]
        h = 0.0
        while length(cl) > 1
            i, j = randperm(rng, length(cl))[1:2]
            a, b = cl[i], cl[j]
            h += 0.2 + rand(rng)
            for (c, l) in ((a, h - a.h), (b, h - b.h)), p in c.mem, q in c.mem
                C[p, q] += l
            end
            new = (nwk = "($(a.nwk):$(h - a.h),$(b.nwk):$(h - b.h))", h = h, mem = [a.mem; b.mem])
            deleteat!(cl, sort([i, j])); push!(cl, new)
        end
        nwk = cl[1].nwk * ";"
        dg = sqrt.(diag(C)); Cc = C ./ (dg * dg')
        a = 0.6 .* (cholesky(Symmetric(Cc)).L * randn(rng, m))
        sp = String[]; t = Float64[]; x = Float64[]; y = Float64[]
        for s in 1:m
            k = rand(rng, 3:6)
            ts = sort(10 .* rand(rng, k))
            R = [exp(-0.45 * abs(u - v)) for u in ts, v in ts]
            xs = randn(rng, k)
            b = 0.7 .* (cholesky(Symmetric(R)).L * randn(rng, k))
            append!(sp, fill(lev[s], k)); append!(t, ts); append!(x, xs)
            append!(y, 0.25 .+ 0.45 .* xs .+ a[s] .+ b .+ 0.4 .* randn(rng, k))
        end
        p = randperm(rng, length(y))
        d = (y = y[p], x = x[p], species = sp[p], elapsed = t[p])
        f = bf(@formula(y ~ x + phylo(1 | species) + temporal(1 | species, elapsed, ou)),
               @formula(sigma ~ 1))
        fit = quiet(() -> drm(f, Gaussian(); data = d, tree = nwk))
        θ = fit.theta                                   # [β; log σ; log σ_a; log σ_t; log λ]
        σ2 = exp(2θ[3]); sa2 = exp(2θ[4]); st2 = exp(2θ[5]); λ = exp(θ[6])
        k = Dict(l => i for (i, l) in enumerate(lev))
        n = length(d.y)
        V = Symmetric([sa2 * Cc[k[d.species[i]], k[d.species[j]]] +
                       (d.species[i] == d.species[j] ? st2 * exp(-λ * abs(d.elapsed[i] - d.elapsed[j])) : 0.0)
                       for i in 1:n, j in 1:n] + σ2 * I)
        r = d.y - hcat(ones(n), d.x) * coef(fit, :mu)
        @test residuals(fit; type = :quantile) ≈ sqrt(σ2) .* (V \ r) atol = 1e-8
    end

    @testset "homtoep keeps the whitened residual" begin
        rng = StableRNG(9191)
        S, K = 30, 5
        id = repeat(["s$i" for i in 1:S], inner = K); occ = repeat(1:K, S)
        x = randn(rng, S * K)
        R = [0.5^abs(i - j) for i in 1:K, j in 1:K]
        L = cholesky(Symmetric(R)).L
        y = 0.3 .+ 0.5 .* x .+ vcat([0.8 .* (L * randn(rng, K)) for _ in 1:S]...)
        d = (y = y, x = x, id = id, occ = occ)
        fit = quiet(() -> drm(bf(@formula(y ~ x + temporal(1 | id, occ, homtoep)), @formula(sigma ~ 1)),
                              Gaussian(); data = d))
        tp = temporal_parameters(fit)
        Rh = [i == j ? 1.0 : tp.cor[abs(i - j)] for i in 1:K, j in 1:K]
        Lh = cholesky(Symmetric(tp.sigma^2 .* Rh)).L
        r = y - hcat(ones(S * K), x) * coef(fit, :mu)
        @test residuals(fit; type = :quantile) ≈ vcat([Lh \ r[(s-1)*K+1:s*K] for s in 1:S]...) atol = 1e-8
    end
end
