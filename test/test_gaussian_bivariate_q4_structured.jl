using DRModels
using Test, LinearAlgebra, Random, SparseArrays, Statistics

# #189 — q=4 coevolution from relmat / animal / spatial (level-indexed Q_cond).

const _Q4S_BETA = (
    mu1 = [0.5, 0.3],
    mu2 = [-0.2, 0.4],
    s1 = [-0.5],
    s2 = [-0.6],
    rho = [0.2],
)

const _Q4S_SIGMA_A = Matrix(Symmetric([
    0.20 0.08 0.03 0.00
    0.08 0.20 0.00 0.03
    0.03 0.00 0.08 0.02
    0.00 0.03 0.02 0.08
]))

function _q4s_simulate(; G::Int = 8, nrep::Int = 3, seed::Int = 189,
                       kind::Symbol = :relmat)
    rng = MersenneTwister(seed)
    # SPD relatedness / correlation over G levels
    Araw = randn(rng, G, G)
    K = Araw * Araw' + 0.5 * I
    K = Matrix(Symmetric(K ./ maximum(diag(K))))
    Q = Matrix(inv(cholesky(Symmetric(K))))
    P = Matrix(prior_precision(sparse(Symmetric(Q)), inv(_Q4S_SIGMA_A)))
    F = cholesky(Symmetric(P))
    u = F.U \ randn(rng, size(P, 1))   # dense Cholesky uses `.U` (not CHOLMOD `.UP`)
    U = reshape(u, 4, G)

    g_levels = [Symbol("g$k") for k in 1:G]
    gidx = repeat(1:G, inner = nrep)
    group = [g_levels[k] for k in gidx]
    n = length(gidx)
    x = randn(rng, n)
    y1 = Vector{Float64}(undef, n)
    y2 = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        k = gidx[i]
        m1 = _Q4S_BETA.mu1[1] + _Q4S_BETA.mu1[2] * x[i] + U[1, k]
        m2 = _Q4S_BETA.mu2[1] + _Q4S_BETA.mu2[2] * x[i] + U[2, k]
        s1 = exp(_Q4S_BETA.s1[1] + U[3, k])
        s2 = exp(_Q4S_BETA.s2[1] + U[4, k])
        ρ = DRModels.RHO_GUARD * tanh(_Q4S_BETA.rho[1])
        e = cholesky(Symmetric([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2])).L * randn(rng, 2)
        y1[i] = m1 + e[1]
        y2[i] = m2 + e[2]
    end
    # Spatial coords: place levels on a line so distances are positive
    coords = hcat(collect(Float64, 1:G), zeros(G))
    data = (; y1, y2, x, id = group, site = group)
    return (; data, K, A = K, coords, kind, G, true_Σ = _Q4S_SIGMA_A)
end

_q4s_formula_relmat() = bf(
    mu1 = @formula(y1 ~ x + relmat(1 | id)),
    mu2 = @formula(y2 ~ x + relmat(1 | id)),
    sigma1 = @formula(sigma1 ~ 1 + relmat(1 | id)),
    sigma2 = @formula(sigma2 ~ 1 + relmat(1 | id)),
    rho12 = @formula(rho12 ~ 1),
)

_q4s_formula_animal() = bf(
    mu1 = @formula(y1 ~ x + animal(1 | id)),
    mu2 = @formula(y2 ~ x + animal(1 | id)),
    sigma1 = @formula(sigma1 ~ 1 + animal(1 | id)),
    sigma2 = @formula(sigma2 ~ 1 + animal(1 | id)),
    rho12 = @formula(rho12 ~ 1),
)

_q4s_formula_spatial() = bf(
    mu1 = @formula(y1 ~ x + spatial(1 | site)),
    mu2 = @formula(y2 ~ x + spatial(1 | site)),
    sigma1 = @formula(sigma1 ~ 1 + spatial(1 | site)),
    sigma2 = @formula(sigma2 ~ 1 + spatial(1 | site)),
    rho12 = @formula(rho12 ~ 1),
)

@testset "Bivariate q=4 relmat front end (#189)" begin
    fx = _q4s_simulate(; kind = :relmat, seed = 1891)
    fit = drm(
        _q4s_formula_relmat(),
        Gaussian();
        data = fx.data,
        K = fx.K,
        q4_iterations = 150,
        q4_n_newton = 30,
        q4_vcov = false,
    )
    @test isfinite(loglik(fit))
    @test fit.ranef isa NamedTuple
    @test size(fit.ranef.Sigma_a) == (4, 4)
    @test fit.ranef.structured_type === :relmat
    @test fit.ranef.phy === nothing
    @test fit.ranef.axes == (:mu1, :mu2, :sigma1, :sigma2)
    C = coevolution_cor(fit)
    @test haskey(C, :cor)
    @test size(C.cor) == (4, 4)
    # Recovery: among-axis SDs within a loose band (small fixture, smoke-level)
    sd_hat = sqrt.(diag(fit.ranef.Sigma_a))
    sd_true = sqrt.(diag(fx.true_Σ))
    @test all(isfinite, sd_hat)
    @test maximum(abs.(sd_hat .- sd_true) ./ max.(sd_true, 0.05)) < 2.5
end

@testset "Bivariate q=4 animal front end (#189)" begin
    fx = _q4s_simulate(; kind = :animal, seed = 1892)
    fit = drm(
        _q4s_formula_animal(),
        Gaussian();
        data = fx.data,
        A = fx.A,
        q4_iterations = 150,
        q4_n_newton = 30,
        q4_vcov = false,
    )
    @test isfinite(loglik(fit))
    @test fit.ranef.structured_type === :animal
    @test size(fit.ranef.Sigma_a) == (4, 4)
    @test size(ranef(fit)[:id], 1) == 4
    @test size(ranef(fit)[:id], 2) == fx.G
end

@testset "Bivariate q=4 spatial fixed-range front end (#189)" begin
    fx = _q4s_simulate(; kind = :spatial, seed = 1893)
    # Simulate under the same fixed-ρ precision the fitter will use
    G = fx.G
    Ddist = [sqrt(sum(abs2, fx.coords[k, :] .- fx.coords[l, :])) for k in 1:G, l in 1:G]
    ρ = sum(Ddist) / (G^2 - G)
    fit = drm(
        _q4s_formula_spatial(),
        Gaussian();
        data = fx.data,
        coords = fx.coords,
        spatial_range = ρ,
        q4_iterations = 150,
        q4_n_newton = 30,
        q4_vcov = false,
    )
    @test isfinite(loglik(fit))
    @test fit.ranef.structured_type === :spatial
    @test fit.ranef.spatial_range ≈ ρ
    @test size(fit.ranef.Sigma_a) == (4, 4)
end

@testset "Bivariate q=4 structured validation (#189)" begin
    fx = _q4s_simulate(; seed = 1894)
    @test_throws Exception drm(
        _q4s_formula_relmat(), Gaussian(); data = fx.data, q4_vcov = false,
    )  # missing K
    @test_throws Exception drm(
        _q4s_formula_spatial(), Gaussian(); data = fx.data, q4_vcov = false,
    )  # missing coords

    # Non-tree bootstrap is fenced
    fit = drm(
        _q4s_formula_relmat(), Gaussian();
        data = fx.data, K = fx.K,
        q4_iterations = 80, q4_n_newton = 25, q4_vcov = false,
    )
    @test_throws ArgumentError bootstrap_sigma_a(fit; data = fx.data, B = 2)
end

@testset "Bivariate q=4 structured gradient wiring smoke (#189)" begin
    # Engine FD ≤1e-6 is gated on BOTH routes now (`test_qgate_fd_gradient.jl`,
    # structured since #510). Here we only prove the structured route wires the
    # same nll / nllgrad closures — on an IDENTIFIED fixture: nrep = 2 made this
    # 4G = 24 latent values against 2n = 24 observations, a saturated model whose
    # Λ is singular, and a smoke assertion on an unidentified fit tests very
    # little (#509, the same lesson as #483). nrep = 4 keeps it small and posed.
    fx = _q4s_simulate(; G = 6, nrep = 4, seed = 1895)
    fit = drm(
        _q4s_formula_relmat(), Gaussian();
        data = fx.data, K = fx.K,
        q4_iterations = 100, q4_n_newton = 30, q4_vcov = false,
    )
    θ = Vector{Float64}(fit.theta)
    @test fit.nll !== nothing && fit.nllgrad !== nothing
    @test isfinite(fit.nll(θ))
    g0 = zeros(length(θ))
    fit.nllgrad(g0, θ)
    @test all(isfinite, g0)
    θp = copy(θ); θp[1:4] .+= 0.05
    gp = zeros(length(θ))
    fit.nllgrad(gp, θp)
    @test norm(gp) > norm(g0)   # informative gradient away from the MLE
end

@testset "converged is gated on Λ admissibility (#509)" begin
    # The gate's CONTRACT, asserted platform-robustly: the public flag never
    # claims success at a numerically inadmissible Λ. Two earlier versions of
    # this testset asserted the saturated fixture's OUTCOME (!converged) and
    # were platform-flaky both ways — on one platform the 24-obs/24-latent
    # knife-edge fixture ran to a singular Λ, on another to an admissible one
    # (and the over-saturated 12/24 variant landed admissible too: latent
    # unidentifiability does not force Λ itself onto the cond >= 1e12
    # boundary). Whatever Λ a platform's optimizer reaches, the invariant the
    # #509 fix guarantees is checkable: converged == true implies the fitted
    # Λ passes the same admissibility test the gate uses.
    for nrep in (1, 2)
        fx = _q4s_simulate(; G = 6, nrep = nrep, seed = 1895)
        fit = drm(
            _q4s_formula_relmat(), Gaussian();
            data = fx.data, K = fx.K,
            q4_iterations = 100, q4_n_newton = 30, q4_vcov = false,
        )
        adm = DRModels._q2_lambda_admissible(Matrix{Float64}(fit.ranef.Sigma_a))
        @test !(fit.converged && !adm)   # never success at an inadmissible Λ
        @test all(isfinite, fit.theta)   # the gate is on the claim, not the fit
    end

    # The gate function itself, deterministically: singular, non-finite, and
    # ill-conditioned matrices refuse; a healthy matrix passes.
    @test !DRModels._q2_lambda_admissible([1.0 1.0; 1.0 1.0])
    @test !DRModels._q2_lambda_admissible([1.0 NaN; NaN 1.0])
    @test !DRModels._q2_lambda_admissible([1.0 0.0; 0.0 1e-13])
    @test DRModels._q2_lambda_admissible(Matrix{Float64}(LinearAlgebra.I, 4, 4))

    # Positive control: the identified default fixture still reports converged,
    # so the gate separates regimes rather than failing everything.
    fx_ok = _q4s_simulate(; seed = 1894)
    fit_ok = drm(
        _q4s_formula_relmat(), Gaussian();
        data = fx_ok.data, K = fx_ok.K,
        q4_iterations = 300, q4_n_newton = 30, q4_vcov = false,
    )
    # 300, not 120: with the whitened ML objective (#857) LBFGS needs more than
    # 120 iterations on this 24-obs fixture (converged=false at 120 on Julia 1.13;
    # main's unwhitened objective converged at 120 and reported a 0.011-nat higher
    # log-likelihood, consistent with its known near-singular-Λ bias); it
    # converges within 300.
    @test fit_ok.converged
end


# A 1-D transect is naturally written as a plain Vector. `Matrix{Float64}(::Vector)` has no
# method, so `coords = [0.0, 1.0, 2.0, 3.0]` used to raise a bare MethodError instead of an
# actionable message. It is now read as the G-by-1 column it obviously means.
@testset "q4 spatial: coords may be a length-G vector for a 1-D transect" begin
    G = 4
    v = [0.0, 1.0, 2.0, 3.0]
    m = reshape(v, G, 1)

    Qv = DRModels._q4_structured_precision(:spatial, :site, G;
                                      K = nothing, A = nothing, coords = v, spatial_range = 1.5)
    Qm = DRModels._q4_structured_precision(:spatial, :site, G;
                                      K = nothing, A = nothing, coords = m, spatial_range = 1.5)

    @test size(Qv) == (G, G)
    @test isposdef(Symmetric(Qv))
    # The Matrix path must be UNTOUCHED by the vector admission, so the two agree exactly
    # rather than merely closely.
    @test Qv == Qm

    # The admission is unambiguous: a FLATTENED 2-D layout has length 2G, so it reshapes to
    # a 2G-by-1 and is still caught by the row check, naming the expected shape.
    flat = vec(rand(G, 2))
    @test_throws ErrorException DRModels._q4_structured_precision(
        :spatial, :site, G; K = nothing, A = nothing, coords = flat, spatial_range = 1.5)

    # Pre-existing shape errors stay clean errors, not MethodErrors.
    @test_throws ErrorException DRModels._q4_structured_precision(
        :spatial, :site, G; K = nothing, A = nothing, coords = zeros(G, 0), spatial_range = 1.5)

    # Something that is neither a real vector nor matrix-convertible now names the expected
    # shape instead of surfacing a MethodError from the constructor.
    vv = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]]
    @test_throws ErrorException DRModels._q4_structured_precision(
        :spatial, :site, G; K = nothing, A = nothing, coords = vv, spatial_range = 1.5)
end
