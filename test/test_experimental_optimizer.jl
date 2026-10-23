# Focused unit tests for the experimental optimizer / EM-robustness fixes
# (#305, #306, #307, #325.1). These exercise the SPECIFIC defects that were
# fixed in the experimental (not-yet-wired) sources under `src/experimental/`:
#
#   #305  location_only.jl  — the LBFGS variance gradient must be a DETERMINISTIC
#         function of θ (exact Takahashi trace), not a fresh stochastic Hutchinson
#         estimate that is inconsistent with the exact objective.
#   #306  location_only.jl  — the conjugate EM must be MONOTONE: the σ² M-step uses
#         the posterior mean at the SAME-iteration β, and a decreasing step is
#         rejected.
#   #307  q4_em_dense.jl     — the oracle E-step must converge on the TRUE gradient
#         norm, never on the step size (which gives false convergence).
#   #325.1 q4_em_dense.jl    — guarded() must not accept a worse point and should
#         probe an interior step (never returns below the incumbent).
#
# The experimental sources are standalone scripts (own `main()`/`run_tests()` and
# self-`include`s that assume the drm-julia-poc layout). We load them into an
# ISOLATED module here, stripping the broken self-`include`s and the trailing
# driver call, so we can call the fixed functions directly on a tiny problem.

using Test, LinearAlgebra, Random, Statistics
using DRModels   # provides `DRModels.Optim` (the Optim module DRModels already depends on)

const _EXP_DIR = joinpath(@__DIR__, "..", "src", "experimental")

# Read a .jl source and strip every top-level `include(...)`, `using …`,
# `import …` line plus the trailing driver call (`main()`/`run_tests()`). The
# `using` lines are re-supplied by the caller (see `_load_experimental`) so the
# module never depends on machine-specific include resolution or a package the
# test environment does not provide.
function _strip_source(path::String)
    kept = String[]
    for ln in split(read(path, String), '\n')
        s = strip(ln)
        startswith(s, "include(") && continue
        startswith(s, "using ") && continue
        startswith(s, "import ") && continue
        (s == "main()" || s == "run_tests()") && continue
        push!(kept, ln)
    end
    return join(kept, '\n')
end

# Load a standalone experimental source into a fresh module.
#   * `usepkgs` are brought in with `using` so their EXPORTED names (e.g.
#     `SparseMatrixCSC`) are in scope — these must all be in the test project.
#   * `Optim` is a dependency of DRModels but NOT of the test project, so a bare
#     `using Optim` inside a test-created module throws "Package Optim not found in
#     current path" on a clean CI checkout. We instead alias it (and the only
#     unqualified Optim name the scripts use, `LBFGS`) from the loaded `DRModels`.
#   * dependency + main sources are `include_string`-ed with all `using`/`import`/
#     `include`/driver lines stripped, so nothing depends on the load path or CWD.
function _load_experimental(mod::Module, relpath::String;
                            deps::Vector{String} = String[],
                            usepkgs::Vector{Symbol} = Symbol[],
                            need_optim::Bool = false)
    isempty(usepkgs) || Core.eval(mod, Expr(:using, (Expr(:., p) for p in usepkgs)...))
    if need_optim
        Core.eval(mod, :(const Optim = $(DRModels.Optim)))
        Core.eval(mod, :(const LBFGS = $(DRModels.Optim.LBFGS)))   # used unqualified in location_only.jl
    end
    for d in deps
        Base.include_string(mod, _strip_source(joinpath(@__DIR__, "..", "src", d)), d)
    end
    Base.include_string(mod, _strip_source(joinpath(_EXP_DIR, relpath)), relpath)
    return mod
end

@testset "experimental optimizer / EM robustness fixes" begin

    # ---------------------------------------------------------------------------
    # location_only.jl  (#305 deterministic gradient, #306 monotone EM)
    # ---------------------------------------------------------------------------
    LocOnly = Module(:LocOnlyTest)
    _load_experimental(LocOnly, "location_only.jl";
                       deps = ["sparse_phy.jl", "takahashi_selinv.jl"],
                       usepkgs = [:LinearAlgebra, :SparseArrays, :Random, :Statistics, :Printf],
                       need_optim = true)

    # A small conjugate location-only phylo problem.
    prob, β_true, σ²_phy_true, σ²_true =
        Base.invokelatest(LocOnly.gen_loc, 32; seed = 7, nrep = 4)

    @testset "#305 LBFGS variance gradient is deterministic in θ" begin
        # With the old fresh-randn Hutchinson estimate the two variance-gradient
        # components differed run-to-run, so the optimiser path (and hence the
        # answer) was RNG-dependent; with the exact Takahashi trace it is not.
        # Call the public fitter twice from the same start under DIFFERENT global
        # RNG states and require bit-for-bit identical results.
        k = prob.k
        Random.seed!(1)
        r1 = Base.invokelatest(LocOnly.lbfgs_fit, prob;
                               β0 = zeros(k), σ²_phy0 = 0.3, σ²0 = 0.2,
                               iterations = 50, g_tol = 1e-7)
        Random.seed!(999)   # different RNG state
        r2 = Base.invokelatest(LocOnly.lbfgs_fit, prob;
                               β0 = zeros(k), σ²_phy0 = 0.3, σ²0 = 0.2,
                               iterations = 50, g_tol = 1e-7)
        @test r1.σ²_phy == r2.σ²_phy      # exact equality: no RNG dependence
        @test r1.σ² == r2.σ²
        @test r1.β == r2.β
        @test r1.loglik == r2.loglik
    end

    @testset "#306 conjugate EM is monotone in the marginal loglik" begin
        # Instrument em_fit by re-implementing its acceptance criterion is
        # overkill; instead run it and check the reported marginal at the fitted
        # point is >= the marginal at the start (the guard forbids a net drop),
        # and that EM and LBFGS agree on the MLE (the gate the bug could break).
        β0 = zeros(prob.k)
        ll_start = Base.invokelatest(LocOnly.marginal_loglik, prob, β0, 0.3, 0.2)
        r_em = Base.invokelatest(LocOnly.em_fit, prob;
                                 β0 = β0, σ²_phy0 = 0.3, σ²0 = 0.2,
                                 max_iter = 200, reltol = 1e-9)
        @test r_em.loglik >= ll_start - 1e-6         # never below the start
        r_lb = Base.invokelatest(LocOnly.lbfgs_fit, prob;
                                 β0 = β0, σ²_phy0 = 0.3, σ²0 = 0.2,
                                 iterations = 300, g_tol = 1e-7)
        # Same MLE (the agreement gate the stale-μ / non-monotone bug undermined).
        @test isapprox(r_em.loglik, r_lb.loglik; atol = 1e-2)
        @test isapprox(r_em.σ², r_lb.σ²; rtol = 0.1)
        @test isapprox(r_em.σ²_phy, r_lb.σ²_phy; rtol = 0.1)
    end

    # ---------------------------------------------------------------------------
    # q4_em_dense.jl  (#307 gradient-norm convergence, #325.1 guarded step)
    # ---------------------------------------------------------------------------
    Q4 = Module(:Q4EmDenseTest)
    _load_experimental(Q4, "q4_em_dense.jl";
                       usepkgs = [:LinearAlgebra, :ForwardDiff, :Statistics])

    # Build a tiny well-posed q4 instance (p species, 1 obs each; intercept-only
    # sigma/rho, an intercept+slope mean). Random phylo Σ from a small AR(1)-like
    # PD matrix so the E-step has a genuine prior to fight against.
    function _make_q4(p; seed = 3)
        rng = MersenneTwister(seed)
        X1 = hcat(ones(p), randn(rng, p)); X2 = hcat(ones(p), randn(rng, p))
        Xs1 = reshape(ones(p), p, 1); Xs2 = reshape(ones(p), p, 1)
        Xr = reshape(ones(p), p, 1)
        D = Base.invokelatest(Q4.Q4Design, X1, X2, Xs1, Xs2, Xr)
        # phylo covariance: exponential-decay PD matrix
        idx = collect(1:p)
        Σ = [0.6^abs(i - j) for i in idx, j in idx] .+ 0.4 * Matrix(I, p, p)
        Σ = Matrix(Symmetric(Σ))
        β_mu1 = [1.0, 0.5]; β_mu2 = [-0.3, 0.4]
        β_s1 = [-0.4]; β_s2 = [-0.5]; β_rho = [0.3]
        Λt = Matrix(Symmetric([0.25 0.10 0.05 0.0;
                               0.10 0.25 0.0 0.04;
                               0.05 0.0 0.09 0.02;
                               0.0 0.04 0.02 0.09]))
        par = Base.invokelatest(Q4.Q4Params, β_mu1, β_mu2, β_s1, β_s2, β_rho, Λt)
        # simulate y from the model at par
        L = cholesky(Λt).L; U = L * randn(rng, 4, p) * cholesky(Symmetric(Σ)).U
        η1 = X1 * β_mu1; η2 = X2 * β_mu2
        ηs1 = Xs1 * β_s1; ηs2 = Xs2 * β_s2; ηr = Xr * β_rho
        y1 = zeros(p); y2 = zeros(p)
        for i in 1:p
            m1 = η1[i] + U[1, i]; m2 = η2[i] + U[2, i]
            s1 = exp(ηs1[i] + U[3, i]); s2 = exp(ηs2[i] + U[4, i])
            ρ = Q4.RHO_GUARD * tanh(ηr[i])
            e = cholesky([s1^2 ρ*s1*s2; ρ*s1*s2 s2^2]).L * randn(rng, 2)
            y1[i] = m1 + e[1]; y2[i] = m2 + e[2]
        end
        return y1, y2, D, par, Matrix(inv(Symmetric(Σ)))
    end

    @testset "#307 E-step reaches the true gradient norm (not a step-size stall)" begin
        y1, y2, D, par, Σ_inv = _make_q4(12; seed = 11)
        Û, Hinv, H = Base.invokelatest(Q4.estep, y1, y2, D, par, Σ_inv;
                                       n_newton = 50, tol = 1e-9)
        u = vec(Û)
        # Recompute the joint gradient at the returned mode and require ‖g‖ small.
        # (Reconstructs the same P·u + per-species data grad the estep uses.)
        p = length(y1)
        Λ_inv = inv(par.Λ)
        Pprec = kron(Σ_inv, Λ_inv)
        η1, η2, ηs1, ηs2, ηr = Base.invokelatest(Q4.eta, D, par)
        g = Pprec * u
        for i in 1:p
            idx = (4(i-1)+1):(4i)
            ui = collect(u[idx])
            args = (y1[i], y2[i], η1[i], η2[i], ηs1[i], ηs2[i], ηr[i])
            g[idx] .+= Base.invokelatest(Q4.data_grad_species, ui, args...)
        end
        @test norm(g) < 1e-6      # a true mode, not a false-converged stall
    end

    @testset "#325.1 guarded() never returns below the incumbent" begin
        # Run the full EM: its per-block guarded() enforces a non-decreasing
        # marginal. The recorded history must be non-decreasing AND converge.
        y1, y2, D, par, Σ_inv = _make_q4(12; seed = 11)
        Σ_phy = Matrix(inv(Symmetric(Σ_inv)))
        res = Base.invokelatest(Q4.fit_q4_em, y1, y2, D, Σ_phy;
                                max_em = 100, tol = 1e-7, verbose = false)
        @test all(diff(res.ll_hist) .>= -1e-8)        # monotone (guard holds)
        @test res.loglik >= res.ll_hist[1] - 1e-8     # never below the start
        @test res.converged                           # identifiable → converges
    end
end
