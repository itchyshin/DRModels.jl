# Gaussian ordinary correlated block of 3 or 4 coefficients with one SD
# formula per coefficient (twin of drmTMB test-gaussian-q3-per-coef-sd*.R):
#   y ~ x + z + (1 + x + z | id), sd(id, coef(intercept)) ~ w, sd(id, coef(x)) ~ w, …
# Exact marginal y_g ~ N(X_g b, s^2 I + U_g D R D U_g'), D = diag(exp(Z a_k)).
# Oracle: independent dense per-group MvNormal log density at the FITTED
# parameters (R rebuilt here from theta by the TMB UNSTRUCTURED_CORR_t map,
# no fitter helper shared). Tolerances, fixed before the first run:
#   logLik vs oracle   atol 1e-6
#   SD-formula slopes  |est - truth| < 1.0  (G = 60 groups, loose, no calibration claim)
#   correlations       |est - truth| < 0.4  (loose)
#   bug emulations (swapped SD columns, R = I) move the oracle by > 1 nat
#   nesting: logLik(w model) >= logLik(constant-SD model) - 1e-6
using DRModels, Test, Random, LinearAlgebra
import Distributions

function _q34_fixture(q; G = 60, m = 8, seed = 20261107)
    rng = MersenneTwister(seed)
    Rm = q == 3 ? [1 0.4 -0.3; 0.4 1 0.2; -0.3 0.2 1.0] :
                  [1 0.4 -0.3 0.2; 0.4 1 0.2 -0.1; -0.3 0.2 1 0.3; 0.2 -0.1 0.3 1.0]
    w = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
    a0 = [-0.5, -0.8, -0.7, -0.6][1:q]
    a1 = [0.5, -0.4, 0.7, 0.3][1:q]
    n = G * m
    gi = repeat(1:G, inner = m)
    U = hcat(ones(n), randn(rng, n, q - 1))
    y = 0.2 .+ 0.5 .* U[:, 2] .- 0.3 .* U[:, 3] .+ 0.4 .* randn(rng, n)
    for g in 1:G
        D = Diagonal(exp.(a0 .+ a1 .* w[g]))
        b = cholesky(Symmetric(D * Rm * D)).L * randn(rng, q)
        rows = ((g-1)*m+1):(g*m)
        y[rows] .+= U[rows, :] * b
    end
    names = (:x, :z, :u)[1:q-1]
    data = (; y, x = U[:, 2], z = U[:, 3], (q == 4 ? (; u = U[:, 4]) : (;))...,
            w = w[gi], id = ["g$(g)" for g in gi])
    return (; data, U, gi, w, G, q, Rm, a0, a1, names)
end

function _q34_formula(fx; sd_members = nothing)
    q = fx.q
    mean = q == 3 ? @formula(y ~ x + z + (1 + x + z | id)) :
                    @formula(y ~ x + z + (1 + x + z + u | id))
    sds = Any[@formula(sd(id, coef(intercept)) ~ w), @formula(sd(id, coef(x)) ~ w),
              @formula(sd(id, coef(z)) ~ w)]
    q == 4 && push!(sds, @formula(sd(id, coef(u)) ~ w))
    return bf(mean, @formula(sigma ~ 1), sds...)
end

# R from theta, TMB UNSTRUCTURED_CORR_t: unit-diagonal L, free entries row-major.
function _q34_R(theta, q)
    L = Matrix{Float64}(I, q, q); k = 0
    for i in 1:q, j in 1:i-1
        k += 1; L[i, j] = theta[k]
    end
    S = L * L'
    d = 1 ./ sqrt.(diag(S))
    return Symmetric(d .* S .* d')
end

function _q34_oracle(fx, beta, sigma, A, Rm)   # A: q x (1 + w) SD coefficients
    q, G = fx.q, fx.G
    n = size(fx.U, 1)
    ll = 0.0
    for g in 1:G
        rows = findall(==(g), fx.gi)
        Ug = fx.U[rows, :]
        D = Diagonal(exp.(A[:, 1] .+ A[:, 2] .* fx.w[g]))
        V = Ug * (D * Rm * D) * Ug' + sigma^2 * I
        Xg = Ug[:, 1:3]    # mean design (1, x, z)
        ll += Distributions.logpdf(Distributions.MvNormal(Xg * beta, Symmetric(V)), fx.data.y[rows])
    end
    return ll
end

function _q34_parts(fit, fx)
    sdkeys = [p.first for p in fit.blocks if p.first ∉ (:mu, :sigma, :recov)]
    A = reduce(vcat, [coef(fit, k)' for k in sdkeys])
    Rm = _q34_R(coef(fit, :recov), fx.q)
    return A, Rm, coef(fit, :mu), exp(coef(fit, :sigma)[1])
end

@testset "Gaussian ordinary q3/q4 correlated block with per-coefficient SD formulas" begin
    for q in (3, 4)
        @testset "q$q" begin
            fx = _q34_fixture(q)
            fit = drm(_q34_formula(fx), Gaussian(); data = fx.data)
            @test fit.converged
            A, Rm, beta, sigma = _q34_parts(fit, fx)
            @test size(A) == (q, 2)
            ll = loglik(fit)
            orc = _q34_oracle(fx, beta, sigma, A, Rm)
            @test abs(ll - orc) < 1e-6
            # the oracle must be able to see misaligned SD columns and a dropped R
            @test abs(orc - _q34_oracle(fx, beta, sigma, A[[2, 1, 3:q...], :], Rm)) > 1
            @test abs(orc - _q34_oracle(fx, beta, sigma, A, Symmetric(Matrix{Float64}(I, q, q)))) > 1
            # loose recovery
            @test all(abs.(A[:, 2] .- fx.a1) .< 1.0)
            @test all(abs.(A[:, 1] .- fx.a0) .< 1.0)
            iu = [(i, j) for i in 1:q for j in 1:i-1]
            @test all(abs(Rm[i, j] - fx.Rm[i, j]) < 0.4 for (i, j) in iu)
            @test all(isfinite, fit.vcov)
            if q == 3
                # nesting: constant-SD model is a special case
                f0 = bf(@formula(y ~ x + z + (1 + x + z | id)), @formula(sigma ~ 1),
                        @formula(sd(id, coef(intercept)) ~ 1), @formula(sd(id, coef(x)) ~ 1),
                        @formula(sd(id, coef(z)) ~ 1))
                fit0 = drm(f0, Gaussian(); data = fx.data)
                @test fit0.converged
                A0, R0, b0, s0 = _q34_parts(fit0, fx)
                @test size(A0) == (3, 1)
                orc0 = _q34_oracle(fx, b0, s0, hcat(A0, zeros(3)), R0)
                @test abs(loglik(fit0) - orc0) < 1e-6
                @test loglik(fit) ≥ loglik(fit0) - 1e-6
                # partial: only the intercept SD regressed; the rest constant
                f1 = bf(@formula(y ~ x + z + (1 + x + z | id)), @formula(sigma ~ 1),
                        @formula(sd(id, coef(intercept)) ~ w))
                fit1 = drm(f1, Gaussian(); data = fx.data)
                @test fit1.converged
                @test loglik(fit1) ≤ loglik(fit) + 1e-6
            end
        end
    end

    @testset "refusals" begin
        fx = _q34_fixture(3)
        d = fx.data
        base(extra...) = bf(@formula(y ~ x + z + (1 + x + z | id)), @formula(sigma ~ 1), extra...)
        sdI = @formula(sd(id, coef(intercept)) ~ w)
        sdx = @formula(sd(id, coef(x)) ~ w)
        # duplicate coefficient target
        @test_throws ArgumentError drm(base(sdI, @formula(sd(id, coef(intercept)) ~ 1)), Gaussian(); data = d)
        # unknown coefficient
        @test_throws ArgumentError drm(base(sdI, @formula(sd(id, coef(v)) ~ w)), Gaussian(); data = d)
        # REML
        err_reml = try
            drm(base(sdI, sdx), Gaussian(); data = d, method = :REML); nothing
        catch e; e; end
        @test err_reml isa ArgumentError && occursin("REML", err_reml.msg)
        # the unstructured q3 block without sd() formulas names REML too
        err_q3 = try
            drm(bf(@formula(y ~ x + z + (1 + x + z | id)), @formula(sigma ~ 1)), Gaussian(); data = d, method = :REML); nothing
        catch e; e; end
        @test err_q3 isa ArgumentError && occursin("method = :REML is not implemented", err_q3.msg)
        # q5 block
        d5 = merge(d, (u = randn(length(d.y)), v = randn(length(d.y))))
        @test_throws ArgumentError drm(bf(@formula(y ~ x + z + (1 + x + z + u + v | id)),
            @formula(sigma ~ 1), sdI), Gaussian(); data = d5)
        # SD predictor varying within group
        dv = merge(d, (w = randn(length(d.y)),))
        @test_throws ArgumentError drm(base(sdI, sdx), Gaussian(); data = dv)
        # an extra random INTERCEPT beside the block now fits (dense route, ML; agreement with R in
        # test_gaussian_sd_gaps.jl); an extra independent slope beside the block is still refused
        de = merge(d, (site = repeat(["a", "b"], length(d.y) ÷ 2),))
        err_slope = try
            drm(bf(@formula(y ~ x + z + (1 + x + z | id) + (0 + x | site)),
                @formula(sigma ~ 1), sdI, sdx), Gaussian(); data = de); nothing
        catch e; e; end
        @test err_slope isa ArgumentError
        @test occursin("beside a random slope term is not implemented", err_slope.msg)
        # the intercept case beside a q3 block with sd() formulas fits (ML)
        fi = drm(bf(@formula(y ~ x + z + (1 + x + z | id) + (1 | site)), @formula(sigma ~ 1), sdI, sdx),
                 Gaussian(); data = de)
        @test fi.converged && isfinite(fi.loglik)
    end
end
