using DRModels
using Test, LinearAlgebra, Random
import Distributions
using SpecialFunctions: loggamma

# Gamma (log link) and NB2 mean models with two latent fields, one predictor-
# dependent SD formula each (src/twofield_sd_laplace.jl):
#   same-group  (1 | g) + (0 + x | g)        sd(g, coef(intercept)), sd(g, coef(x))
#   crossed     (1 | g) + (1 | h)            sd(g), sd(h)
#   ord+struct  (1 | g) + relmat/animal(1 | p)  sd(g), sd(p, relmat|animal)
#   relmat q2   relmat(1 + x | p)  (Gamma)   sd(p, relmat, coef(intercept)), sd(p, relmat, coef(x))
#   labelled    animal/relmat(1 + x | p | g)  correlated fields, rho = tanh(atanh_rho)
#   correlated  (1 + x | g)  (NB2)            correlated fields, Q = I
#
# Oracle: an independent dense Laplace marginal written without DRModels
# internals: b ~ N(0, blockdiag(D1 C1 D1, D2 C2 D2)) on the ORIGINAL latent
# scale (not the standardised one the engine uses), Distributions.jl log
# densities, Newton to the mode, joint(b*) + q/2 log(2 pi) - 0.5 logdet(H).
# Evaluated at the fitted parameters. Tolerance fixed before the first run:
# |loglik(fit) - oracle| < 1e-5. Emulated bugs (reversed SD vector of either
# field, kernel rows/columns permuted) must move the oracle by more than 0.1.

const TF_RNG_SEED = 20261107

tf_logdens(fam::Symbol, y, eta, sigma) = fam === :gamma ?
    Distributions.logpdf(Distributions.Gamma(1 / sigma^2, exp(eta) * sigma^2), y) :
    Distributions.logpdf(Distributions.NegativeBinomial(1 / sigma^2,
        (1 / sigma^2) / (1 / sigma^2 + exp(eta))), y)

# score and observed information of the log density in eta
function tf_derivs(fam::Symbol, y, eta, sigma)
    mu = exp(eta); k = 1 / sigma^2
    fam === :gamma ? (k * (y / mu - 1), k * y / mu) :
        (k * (y - mu) / (k + mu), k * mu * (k + y) / (k + mu)^2)
end

function tf_oracle(fam, y, eta0, Z, Sigma, sigma)
    q = size(Z, 2)
    Sinv = inv(Symmetric(Sigma))
    ldS = logdet(Symmetric(Sigma))
    joint(b) = sum(tf_logdens.(fam, y, eta0 .+ Z * b, sigma)) -
               0.5 * dot(b, Sinv * b) - 0.5 * ldS - 0.5 * q * log(2π)
    b = zeros(q)
    for _ in 1:200
        d = tf_derivs.(fam, y, eta0 .+ Z * b, sigma)
        g = Z' * first.(d) - Sinv * b
        H = Z' * (last.(d) .* Z) + Sinv
        step = H \ g
        s = 1.0
        while joint(b + s * step) < joint(b) - 1e-12 && s > 1e-8
            s /= 2
        end
        b += s * step
        maximum(abs, g) < 1e-10 && break
    end
    d = tf_derivs.(fam, y, eta0 .+ Z * b, sigma)
    @assert maximum(abs, Z' * first.(d) - Sinv * b) < 1e-7
    H = Z' * (last.(d) .* Z) + Sinv
    return joint(b) + 0.5 * q * log(2π) - 0.5 * logdet(Symmetric(H))
end

# indicator design for labels in a given level order
tf_indicator(labels, levels) = Float64[labels[i] == levels[j] for i in eachindex(labels), j in eachindex(levels)]

tf_draw(fam, rng, eta, sigma) = fam === :gamma ?
    [rand(rng, Distributions.Gamma(1 / sigma^2, exp(e) * sigma^2)) for e in eta] :
    Float64[rand(rng, Distributions.NegativeBinomial(1 / sigma^2,
        (1 / sigma^2) / (1 / sigma^2 + exp(e)))) for e in eta]

function tf_kernel(G)
    R = [0.4^abs(i - j) for i in 1:G, j in 1:G]
    s = collect(range(0.7, 1.4; length = G))
    return Diagonal(s) * R * Diagonal(s)       # raw, non-unit diagonal
end

# Returns data, formula, kwargs, and a field description for the oracle:
# fields = [(labels, levels, C, w, zlev)] with zlev the group-level SD predictor
# in `levels` order; each SD formula is ~ 1 + z.
function tf_case(fam::Symbol, shape::Symbol; seed = TF_RNG_SEED)
    # ordanimal reuses the ordrelmat draw, so A and K must give the same fit.
    offset = Dict(:samegroup => 1, :crossed => 2, :ordrelmat => 3, :ordanimal => 3,
                  :relmatq2 => 5, :labanimal => 6, :labrelmat => 6, :corr => 8)[shape]
    rng = MersenneTwister(seed + offset)
    sigma = 0.45
    a1, a2 = [log(0.5), 0.6], [log(0.35), -0.6]
    if shape === :samegroup
        G, m = 40, 10
        lev = ["g$(i)" for i in randperm(rng, G)]
        g = repeat(lev, inner = m)
        x = repeat(collect(range(-1.2, 1.2; length = m)), outer = G)
        z1 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        z2 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        gi = repeat(1:G, inner = m)
        b1 = exp.(a1[1] .+ a1[2] .* z1) .* randn(rng, G)
        b2 = exp.(a2[1] .+ a2[2] .* z2) .* randn(rng, G)
        eta = 0.4 .+ 0.3 .* x .+ b1[gi] .+ x .* b2[gi]
        y = tf_draw(fam, rng, eta, sigma)
        data = (; y, x, g, w1 = z1[gi], w2 = z2[gi])
        formula = bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1),
                     @formula(sd(g, coef(intercept)) ~ w1), @formula(sd(g, coef(x)) ~ w2))
        fields = [(g, lev, Matrix(1.0I, G, G), ones(length(y)), z1),
                  (g, lev, Matrix(1.0I, G, G), x, z2)]
        return (; data, formula, kw = (;), fields)
    elseif shape === :crossed
        G, H, n = 30, 20, 480
        levg = ["g$(i)" for i in 1:G]; levh = ["h$(i)" for i in 1:H]
        gi = repeat(1:G, inner = n ÷ G); hi = rand(rng, 1:H, n)
        x = randn(rng, n)
        z1 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        z2 = collect(range(-1.5, 1.5; length = H))[randperm(rng, H)]
        b1 = exp.(a1[1] .+ a1[2] .* z1) .* randn(rng, G)
        b2 = exp.(a2[1] .+ a2[2] .* z2) .* randn(rng, H)
        eta = 0.4 .+ 0.3 .* x .+ b1[gi] .+ b2[hi]
        y = tf_draw(fam, rng, eta, sigma)
        g, h = levg[gi], levh[hi]
        data = (; y, x, g, h, w1 = z1[gi], w2 = z2[hi])
        formula = bf(@formula(y ~ x + (1 | g) + (1 | h)), @formula(sigma ~ 1),
                     @formula(sd(g) ~ w1), @formula(sd(h) ~ w2))
        fields = [(g, unique(g), Matrix(1.0I, G, G), ones(n), z1[indexin(unique(g), levg)]),
                  (h, unique(h), Matrix(1.0I, H, H), ones(n), z2[indexin(unique(h), levh)])]
        return (; data, formula, kw = (;), fields)
    elseif shape in (:ordrelmat, :ordanimal)
        G, P, n = 25, 20, 500
        levg = ["g$(i)" for i in 1:G]; levp = ["p$(i)" for i in 1:P]
        gi = rand(rng, 1:G, n); pi_ = repeat(1:P, inner = n ÷ P)[randperm(rng, n)]
        x = randn(rng, n)
        z1 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        z2 = collect(range(-1.5, 1.5; length = P))[randperm(rng, P)]
        Cp = tf_kernel(P)                     # in levp order
        b1 = exp.(a1[1] .+ a1[2] .* z1) .* randn(rng, G)
        b2 = exp.(a2[1] .+ a2[2] .* z2) .* (cholesky(Symmetric(Cp)).L * randn(rng, P))
        eta = 0.4 .+ 0.3 .* x .+ b1[gi] .+ b2[pi_]
        y = tf_draw(fam, rng, eta, sigma)
        g, p = levg[gi], levp[pi_]
        data = (; y, x, g, p, w1 = z1[gi], w2 = z2[pi_])
        # The kernel is supplied in first-appearance order of `p`.
        ordp = indexin(unique(p), levp)
        C = Cp[ordp, ordp]
        if shape === :ordrelmat
            formula = bf(@formula(y ~ x + (1 | g) + relmat(1 | p)), @formula(sigma ~ 1),
                         @formula(sd(g) ~ w1), @formula(sd(p, relmat) ~ w2))
            kw = (; K = C)
        else
            formula = bf(@formula(y ~ x + (1 | g) + animal(1 | p)), @formula(sigma ~ 1),
                         @formula(sd(g) ~ w1), @formula(sd(p, animal) ~ w2))
            kw = (; A = C)
        end
        fields = [(g, unique(g), Matrix(1.0I, G, G), ones(n), z1[indexin(unique(g), levg)]),
                  (p, unique(p), C, ones(n), z2[ordp])]
        return (; data, formula, kw, fields)
    elseif shape === :relmatq2
        G, m = 30, 16
        lev = ["u$(i)" for i in 1:G]
        gi = repeat(1:G, inner = m)
        x = randn(rng, G * m)
        z1 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        z2 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        C = tf_kernel(G)
        Lc = cholesky(Symmetric(C)).L
        b1 = exp.(a1[1] .+ a1[2] .* z1) .* (Lc * randn(rng, G))
        b2 = exp.(a2[1] .+ a2[2] .* z2) .* (Lc * randn(rng, G))
        eta = 0.4 .+ 0.3 .* x .+ b1[gi] .+ x .* b2[gi]
        y = tf_draw(fam, rng, eta, sigma)
        g = lev[gi]
        data = (; y, x, g, w1 = z1[gi], w2 = z2[gi])
        formula = bf(@formula(y ~ x + relmat(1 + x | g)), @formula(sigma ~ 1),
                     @formula(sd(g, relmat, coef(intercept)) ~ w1),
                     @formula(sd(g, relmat, coef(x)) ~ w2))
        fields = [(g, lev, C, ones(length(y)), z1), (g, lev, C, x, z2)]
        return (; data, formula, kw = (; K = C), fields)
    end
    if shape in (:labanimal, :labrelmat, :corr)
        G, m = 40, 14
        lev = ["u$(i)" for i in 1:G]
        gi = repeat(1:G, inner = m)
        x = randn(rng, G * m)
        z1 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        z2 = collect(range(-1.5, 1.5; length = G))[randperm(rng, G)]
        C = shape === :corr ? Matrix(1.0I, G, G) : tf_kernel(G)
        Lc = cholesky(Symmetric(C)).L
        rho = 0.5
        f0 = Lc * randn(rng, G); f1 = rho .* f0 .+ sqrt(1 - rho^2) .* (Lc * randn(rng, G))
        b1 = exp.(a1[1] .+ a1[2] .* z1) .* f0
        b2 = exp.(a2[1] .+ a2[2] .* z2) .* f1
        eta = 0.4 .+ 0.3 .* x .+ b1[gi] .+ x .* b2[gi]
        y = tf_draw(fam, rng, eta, sigma)
        g = lev[gi]
        data = (; y, x, g, w1 = z1[gi], w2 = z2[gi])
        fields = [(g, lev, C, ones(length(y)), z1), (g, lev, C, x, z2)]
        if shape === :corr
            formula = bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
                         @formula(sd(g, coef(intercept)) ~ w1), @formula(sd(g, coef(x)) ~ w2))
            return (; data, formula, kw = (;), fields)
        elseif shape === :labanimal
            formula = bf(@formula(y ~ x + animal(1 + x | p | g)), @formula(sigma ~ 1),
                         @formula(sd(g, animal, coef(intercept)) ~ w1),
                         @formula(sd(g, animal, coef(x)) ~ w2))
            return (; data, formula, kw = (; A = C), fields)
        else
            formula = bf(@formula(y ~ x + relmat(1 + x | p | g)), @formula(sigma ~ 1),
                         @formula(sd(g, relmat, coef(intercept)) ~ w1),
                         @formula(sd(g, relmat, coef(x)) ~ w2))
            return (; data, formula, kw = (; K = C), fields)
        end
    end
    error("unknown shape")
end

# Oracle at the fitted parameters. `perturb` emulates a bug.
function tf_fit_oracle(fam, case, fit; perturb = :none)
    targets = DRModels._sd_target_parts(case.formula)
    blocks = [DRModels._sd_target_block(t.target; disambiguate = true) for t in targets]
    beta = coef(fit, :mu); sigma = exp(only(coef(fit, :sigma)))
    eta0 = beta[1] .+ beta[2] .* case.data.x
    Zs = Matrix{Float64}[]; Sig = Matrix{Float64}[]
    for (k, (labels, levels, C, w, z)) in enumerate(case.fields)
        a = coef(fit, blocks[k])
        zz = (perturb === Symbol("rev", k)) ? reverse(z) : z
        tau = exp.(a[1] .+ a[2] .* zz)
        pm = randperm(MersenneTwister(7), size(C, 1))
        CC = (perturb === :kernelpos && !isdiag(C)) ? C[pm, pm] : C
        push!(Zs, w .* tf_indicator(labels, levels))
        push!(Sig, Diagonal(tau) * CC * Diagonal(tau))
    end
    q1, q2 = size(Sig[1], 1), size(Sig[2], 1)
    Sigma = zeros(q1 + q2, q1 + q2)
    Sigma[1:q1, 1:q1] = Sig[1]; Sigma[(q1+1):end, (q1+1):end] = Sig[2]
    if any(p -> p.first === :recov, fit.blocks)
        rho = tanh(only(coef(fit, :recov)))
        perturb === :rho0 && (rho = 0.0)
        S1 = sqrt.(diag(Sig[1])); S2 = sqrt.(diag(Sig[2]))   # tau_k * sqrt(diag C)
        C = case.fields[1][3]
        pm = randperm(MersenneTwister(7), size(C, 1))
        perturb === :kernelpos && (C = C[pm, pm])
        t1 = S1 ./ sqrt.(diag(C)); t2 = S2 ./ sqrt.(diag(C))
        X12 = rho .* (Diagonal(t1) * C * Diagonal(t2))
        Sigma[1:q1, (q1+1):end] = X12; Sigma[(q1+1):end, 1:q1] = X12'
    end
    return tf_oracle(fam, case.data.y, eta0, hcat(Zs...), Sigma, sigma)
end

const TF_FAMS = Dict(:gamma => DRModels.Gamma(), :nb2 => NegBinomial2())

for (fam, shape) in ((:nb2, :samegroup), (:gamma, :samegroup), (:gamma, :crossed),
                     (:gamma, :ordrelmat), (:gamma, :ordanimal), (:gamma, :relmatq2),
                     (:nb2, :crossed), (:nb2, :ordrelmat), (:gamma, :labanimal),
                     (:gamma, :labrelmat), (:nb2, :labanimal), (:nb2, :corr))
    @testset "$fam $shape two-field SD regression: dense Laplace oracle" begin
        case = tf_case(fam, shape)
        fit = drm(case.formula, TF_FAMS[fam]; data = case.data, case.kw...)
        @test fit.converged
        @test all(isfinite, stderror(fit))
        targets = DRModels._sd_target_parts(case.formula)
        blocks = [DRModels._sd_target_block(t.target; disambiguate = true) for t in targets]
        @test all(b -> length(coef(fit, b)) == 2, blocks)
        oracle = tf_fit_oracle(fam, case, fit)
        @test abs(loglikelihood(fit) - oracle) < 1e-5
        for bug in (:rev1, :rev2)
            @test abs(tf_fit_oracle(fam, case, fit; perturb = bug) - oracle) > 0.1
        end
        if any(f -> !isdiag(f[3]), case.fields)
            @test abs(tf_fit_oracle(fam, case, fit; perturb = :kernelpos) - oracle) > 0.1
        end
        if shape === :ordanimal
            ck = tf_case(fam, :ordrelmat)
            fk = drm(bf(@formula(y ~ x + (1 | g) + relmat(1 | p)), @formula(sigma ~ 1),
                        @formula(sd(g) ~ w1), @formula(sd(p, relmat) ~ w2)),
                     TF_FAMS[fam]; data = ck.data, K = ck.kw.K)
            @test abs(loglikelihood(fk) - loglikelihood(fit)) < 1e-6
        end
        if shape in (:labanimal, :labrelmat, :corr)
            @test abs(tf_fit_oracle(fam, case, fit; perturb = :rho0) - oracle) > 0.1
            haskey(fit.scales, :correlation) &&
                @test only(fit.scales[:correlation]) ≈ tanh(only(coef(fit, :recov)))
        end
        if shape === :labanimal && fam === :gamma
            ck = tf_case(fam, :labrelmat)
            fk = drm(ck.formula, TF_FAMS[fam]; data = ck.data, ck.kw...)
            @test abs(loglikelihood(fk) - loglikelihood(fit)) < 1e-6
        end
        # group-level SD scales follow the fitted SD formulas
        for (k, b) in enumerate(blocks)
            @test length(fit.scales[Symbol(string(b), "_group")]) == length(case.fields[k][2])
        end
    end
end

@testset "two-field SD regression: neighbouring shapes stay refused" begin
    case = tf_case(:gamma, :samegroup)
    d = case.data
    # one SD formula on a same-group pair: drmTMB 0.8.1 fits it, the other coefficient keeping an
    # estimated constant SD (the default `sd(g, coef(.)) ~ 1`), so Gamma now takes the same shape
    # (agreement with R: test_struct_sd_counts.jl). Here: two SD blocks, the unselected one constant.
    fit1 = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1),
        @formula(sd(g, coef(x)) ~ w2)), DRModels.Gamma(); data = d)
    @test fit1.converged === true
    @test count(p -> startswith(String(first(p)), "sdtarget_"), fit1.blocks) == 2
    @test sort([length(last(p)) for p in fit1.coefnames if startswith(String(first(p)), "sdtarget_")]) == [1, 2]
    # a coefficient that is not in the pair stays refused
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1),
        @formula(sd(g, coef(w1)) ~ w2)), DRModels.Gamma(); data = d)
    # Gamma ordinary correlated (1 + x | g) with SDs (native drmTMB refuses it too)
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 + x | g)), @formula(sigma ~ 1),
        @formula(sd(g, coef(intercept)) ~ w1), @formula(sd(g, coef(x)) ~ w2)),
        DRModels.Gamma(); data = d)
    # non-constant sigma
    @test_throws ArgumentError drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g)), @formula(sigma ~ 1 + x),
        @formula(sd(g, coef(intercept)) ~ w1), @formula(sd(g, coef(x)) ~ w2)),
        DRModels.Gamma(); data = d)
    # a covariance matrix on an all-ordinary model
    @test_throws ArgumentError drm(case.formula, DRModels.Gamma(); data = d,
        K = Matrix(1.0I, 40, 40))
    # ordinary + animal with the wrong matrix keyword
    c2 = tf_case(:gamma, :ordanimal)
    @test_throws ArgumentError drm(c2.formula, DRModels.Gamma(); data = c2.data, K = c2.kw.A)
end

@testset "two-field simulator and bootstrap accept the animal Ainv the Gamma fit accepted (NB2 two-field refuses Ainv at fit time)" for fam in (:gamma,)
    case = tf_case(fam, :ordanimal)
    C = Matrix(case.kw.A)
    Ainv = Matrix(Symmetric(inv(Symmetric(C))))
    fit_A = drm(case.formula, TF_FAMS[fam]; data = case.data, A = C)
    fit_I = drm(case.formula, TF_FAMS[fam]; data = case.data, Ainv)
    @test abs(loglikelihood(fit_A) - loglikelihood(fit_I)) < 1e-6
    sim_A = DRModels._marginal_simulator_build(fit_A, case.data; A = C)
    sim_I = DRModels._marginal_simulator_build(fit_I, case.data; Ainv)
    @test sim_I(MersenneTwister(3)) ≈ sim_A(MersenneTwister(3)) rtol = 1e-6
    # both A and Ainv is an error, not a silent pick
    @test_throws ArgumentError DRModels._marginal_simulator_build(fit_I, case.data; A = C, Ainv)
    # end-to-end parametric bootstrap with Ainv
    boot = bootstrap_result(case.formula, TF_FAMS[fam]; data = case.data, B = 1, Ainv,
                            rng = MersenneTwister(5), g_tol = 1e-5)
    @test boot !== nothing
end
