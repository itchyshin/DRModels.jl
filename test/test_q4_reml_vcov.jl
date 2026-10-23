# test_q4_reml_vcov.jl — #563 S11 slice 2/3: pin the current q4 REML `vcov()`
# behaviour, both through the native (no-bridge) fitter and through the R
# bridge, and root-cause the "all-NaN" `drm_bridge(...)["vcov"]` the
# 2026-09-02 drmTMB SE receipt observed.
#
# FINDING (root cause): `_vcov_from_hessian` (src/vcov_guard.jl) is NOT the
# source of the NaN — it never returns all-NaN, only a possibly-untrustworthy
# `pinv` fallback with a `@warn`. On this fixture it is never even reached
# through the bridge: `_bridge_fit` (src/bridge.jl:458-464) deliberately sets
# `kwargs[:q4_vcov] = false` by default for every bivariate q=4 phylogenetic
# fit built through the bridge (`drm_bridge`/`drm_bridge_inference`), UNLESS
# the caller passes `options["q4_vcov"] = true` explicitly. That default skips
# `_q4_fd_vcov` entirely, so `_fit_bivariate_q4_phylo`/`_fit_bivariate_q4_structured`
# (src/gaussian_bivariate.jl:950-951 / :786-787) fall straight to
# `fill(NaN, length(θ̂), length(θ̂))` — the literal first NaN-producing
# expression. This is a deliberate, ALREADY-DOCUMENTED-AND-TESTED default (the
# comment at bridge.jl:459-463 names the cost/robustness reason; the NaN
# outcome is separately pinned by `test/test_bridge_bivariate_inference.jl`'s
# `@test all(isnan, vec(fit_reml["vcov"]))`), not a numerics defect — so
# nothing here is broken and nothing is fixed.
#
# `_bridge_bivariate_inference` (the function the #563 slice brief also names)
# is a red herring for this specific gap: for a q=4 phylo fit it is reached
# ONLY for the four among-axis Sigma_a SD rows, never for the fixed-effect
# (β_μ, β_ψ) block, and it never falls back to NaN either — it either answers
# via profile/bootstrap or throws `ArgumentError` for `method = "wald"`
# (bridge.jl:2504-2508, matching the boundary-singular-Hessian contract at
# bridge.jl:2469-2475). `drm_bridge_inference` has NO Wald route at all, for
# any target; the fixed-effect Wald covariance is exposed only through
# `drm_bridge`'s full `vcov` payload field (`_bridge_flatten`, bridge.jl:1363).
#
# On this WELL-IDENTIFIED fixture (n=128, converged REML optimum, #575's
# exact-gradient engine) the native `drm(...; method = :REML)` route — which
# defaults `q4_vcov = true` — already returns a fully finite, symmetric,
# positive-definite-on-the-fixed-effect-block covariance matrix; explicitly
# passing `q4_vcov = true` through the bridge reproduces the same finite
# result. So the capability already exists; only the bridge's opt-OUT default
# needs an explicit override to reach it. Reversing that default globally is
# NOT attempted here: it would change behaviour/cost for every bridge q=4
# phylogenetic call (not just this fixture), reversing a documented
# performance/robustness tradeoff (bridge.jl:459-463) without measurement at
# the large-fit scale that comment is about — exactly the re-scoping risk
# the #563 S11 scout note (`s11-inference.md` §3.4) flags as possibly beyond
# a "tonight" slice. That call belongs to a programme owner, not this test.
#
#   julia --project=. -e 'using DRModels, Test; include("test/test_q4_reml_vcov.jl")'

module TestQ4RemlVcov

using DRModels
using Test
using LinearAlgebra
using DelimitedFiles: readdlm

const FIXTURE = joinpath(@__DIR__, "parity", "q4-reml", "biv-q4-phylo-reml")

function _load_data(dir)
    raw, header = readdlm(joinpath(dir, "data.csv"), ','; header = true)
    cols = Symbol.(strip.(string.(vec(header))))
    numeric = Set((:y1, :y2, :x))
    pairs = map(enumerate(cols)) do (j, name)
        col = raw[:, j]
        if name in numeric
            name => Float64[parse(Float64, string(v)) for v in col]
        else
            name => string.(col)
        end
    end
    return NamedTuple(pairs)
end

const DAT = _load_data(FIXTURE)
const TREE = read(joinpath(FIXTURE, "tree.newick"), String)
const FORM = bf(mu1    = @formula(y1 ~ x + phylo(1 | species)),
                mu2    = @formula(y2 ~ x + phylo(1 | species)),
                sigma1 = @formula(sigma1 ~ 1 + phylo(1 | species)),
                sigma2 = @formula(sigma2 ~ 1 + phylo(1 | species)),
                rho12  = @formula(rho12 ~ 1))

const BRIDGE_FORMULA = Dict(
    :mu1 => "y1 ~ x + phylo(1 | species)",
    :mu2 => "y2 ~ x + phylo(1 | species)",
    :sigma1 => "sigma1 ~ 1 + phylo(1 | species)",
    :sigma2 => "sigma2 ~ 1 + phylo(1 | species)",
    :rho12 => "rho12 ~ 1",
)
const BRIDGE_FAMILY = "biv_gaussian"

# Independent double-central-difference Hessian of the fit's own stored NLL
# closure (`fit.nll`, the ML marginal NLL — the SAME quantity `_q4_fd_vcov`'s
# comment (gaussian_bivariate.jl:946-949) says `vcov(fit)` reports for a q4
# REML fit). This is independent of `_q4_fd_vcov` in differentiation SCHEME
# (value-based 2nd-order central difference here vs. central-differencing the
# EXACT analytic gradient there), not independent in target quantity.
#
# NOTE: `reml_objective_at` (named in the slice brief) is NOT used for this
# check — it operates on the REDUCED `phi` parameterisation (rho12 + the
# log-Cholesky of Sigma_a only; beta is profiled OUT, held fixed as `beta0`),
# so it has no curvature in the beta directions at all and is not comparable
# to `vcov(fit)`'s fixed-effect block. Using it here would compare two
# different quantities, not verify the same one independently.
function _independent_fd_hessian(fit; h::Real = 1e-3)
    θ̂ = fit.theta
    f = fit.nll
    nθ = length(θ̂)
    f0 = f(θ̂)
    H = zeros(nθ, nθ)
    for k in 1:nθ
        ep = copy(θ̂); ep[k] += h
        em = copy(θ̂); em[k] -= h
        H[k, k] = (f(ep) - 2 * f0 + f(em)) / h^2
    end
    for k in 1:nθ, j in (k + 1):nθ
        epp = copy(θ̂); epp[k] += h; epp[j] += h
        epm = copy(θ̂); epm[k] += h; epm[j] -= h
        emp = copy(θ̂); emp[k] -= h; emp[j] += h
        emm = copy(θ̂); emm[k] -= h; emm[j] -= h
        val = (f(epp) - f(epm) - f(emp) + f(emm)) / (4h^2)
        H[k, j] = val
        H[j, k] = val
    end
    return H
end

@testset "q4 REML vcov() — current native and bridge behaviour" begin

    @testset "(a) native drm() (no bridge): vcov(fit) is finite, symmetric, PD on fixef block" begin
        fit = drm(FORM, Gaussian(); data = DAT, tree = TREE, method = :REML)
        @test fit.converged

        V = vcov(fit)
        @test size(V) == (17, 17)
        @test all(isfinite, V)
        @test V ≈ V' atol=1e-12

        fixef_idx = 1:7   # mu1(2) + mu2(2) + sigma1(1) + sigma2(1) + rho12(1), per fit.blocks
        @test Dict(fit.blocks)[:rho12] == 7:7
        Vfixef = Symmetric(V[fixef_idx, fixef_idx])
        @test isposdef(Vfixef)

        # Independent double-FD Hessian of the SAME ML marginal NLL (see docstring
        # above): diag(vcov) should agree to a loose 1e-3 relative tolerance.
        H_ind = _independent_fd_hessian(fit)
        V_ind = DRModels._vcov_from_hessian(H_ind; context = "test_q4_reml_vcov independent check")
        d_fit = diag(V)
        d_ind = diag(V_ind)
        relerr = abs.(d_fit .- d_ind) ./ max.(abs.(d_fit), 1e-8)
        @test maximum(relerr) < 1e-3
    end

    @testset "(b) bridge default (no explicit q4_vcov): all-NaN — current, deliberate behaviour" begin
        out = DRModels.drm_bridge(; formula = BRIDGE_FORMULA, family = BRIDGE_FAMILY,
                             data = DAT, tree = TREE,
                             options = Dict{String,Any}("method" => "REML"))
        V = out["vcov"]
        @test size(V) == (17, 17)
        @test all(isnan, V)   # matches test_bridge_bivariate_inference.jl's existing pin
    end

    @testset "(c) bridge with explicit q4_vcov=true: finite SEs — the capability already exists" begin
        out = DRModels.drm_bridge(; formula = BRIDGE_FORMULA, family = BRIDGE_FAMILY,
                             data = DAT, tree = TREE,
                             options = Dict{String,Any}("method" => "REML", "q4_vcov" => true))
        V = out["vcov"]
        @test all(isfinite, V)
        se = sqrt.(diag(V))
        @test all(isfinite, se)
        @test all(se .> 0)
    end

end

end # module TestQ4RemlVcov
