# test_lss_reml_falseconv.jl -- the location-scale-scale REML route on the
# #835 docs-regression model (`y ~ sex + (1 | id)`, `sigma ~ sex`, `sd(id) ~ sex`).
#
# WHAT WAS REPORTED. test/test_twin_gap_747_lss_reml.jl draws its data with
# `Random.MersenneTwister(20260715)` and fitted ll = -412.301 on Julia 1.10 but
# ll = -408.179 on Julia 1.13, both `converged = true`. That was read as "LBFGS
# stops at a non-stationary point on 1.10".
#
# WHAT IS ACTUALLY GOING ON (measured 2026-09-27, both versions):
#   1. The two versions fit DIFFERENT DATA. MersenneTwister(20260715)'s stream
#      differs between Julia 1.10 and 1.13 (sum(y) = 281.352 vs 235.600; the
#      first scalar `randn` is -0.447 vs 0.633). Each answer is a genuine
#      stationary point of its own data: at 1.10's theta-hat a 256-bit
#      BigFloat central-difference gradient of an independent dense-V REML
#      objective has inf-norm 4.6e-10, and at 1.13's 3.1e-13. Given the SAME
#      literal data, 1.10 and 1.13 agree (ll = -412.3006010571 on the 1.10
#      draw, to 4e-12). There was no false convergence.
#   2. The real defect: fitting the 1.13 draw on Julia 1.10 THREW
#      `AssertionError: isfinite(phi_c) && isfinite(dphi_c)` -- even with
#      #835's `invsigb2 == Inf` guard. An early LBFGS line-search probe (the
#      second trial point of the first iteration) goes to log sigma ~ -105,
#      log sigma_b ~ +12. There the REML term's
#      Xmu'V^-1 Xmu, formed by Woodbury SUBTRACTION
#      (Xmu'D^-1 Xmu - sum_k z_k z_k'/M_k), came out as 8.08e110 * [1 1; 1 1]
#      (exactly singular garbage) against a true value of ~1.34e-7 * [1 1; 1 1].
#      With Float64 `cholesky(check = false)` rejects it and the finite barrier
#      fires, but with the ForwardDiff Dual numbers LBFGS actually evaluates,
#      the generic Cholesky ACCEPTED the zero pivot (issuccess = true), so
#      `logdet` = -Inf, the objective = -Inf and the gradient NaN. Which Julia
#      version hits it depends on last-bit differences in the probe.
#
# THE FIX. `_re_xtvinvx_stable` forms Xmu'V^-1 Xmu as the penalised sum of
# squares  sum_i (x_i - w_i U_k)(x_i - w_i U_k)'/D_i + sum_k U_k U_k'/sigma_b,k^2
# at the conditional mode U_k (one refinement step), the matrix analogue of
# `_re_quad_stable` (#746/#747). It is PSD by construction and accurate where the
# subtraction was garbage. Both REML objectives (`_fit_ranef_gaussian` and
# `_fit_ranef_gaussian_lss`) use it, and the barrier now also fires on a
# non-finite `logdet`.
#
# FAILS BEFORE THE FIX: the first testset throws on Julia 1.10 (the 1.13 draw);
# the second testset needs `_re_xtvinvx_stable`. The third testset is the
# smallest direct test of #835's own guard (a `_re_quad_stable` call with
# `invsigb2 = Inf`), which returns NaN before #835's fix -- see the PR body.
#
#   julia --project=test -e 'using DRModels, Test; include("test/test_lss_reml_falseconv.jl")'

module TestLssRemlFalseconv

using DRModels
using Test
using DelimitedFiles: readdlm
using ForwardDiff
using LinearAlgebra: cholesky, Symmetric, logdet, dot, Diagonal

const FIXDIR = joinpath(@__DIR__, "fixtures", "lss_reml_falseconv")
const n_id, n_each = 80, 6
const sex = repeat([0.0, 1.0], inner = n_id ÷ 2)
const id = repeat(1:n_id, inner = n_each)
const sexl = sex[id]
const Xmu = hcat(ones(n_id * n_each), sexl)
const Zg = hcat(ones(n_id), sex)

# Independent REML objective: exact per-group V_k = D_k + sigma_b,k^2 11' (no
# Woodbury), accumulated block by block. Used for the gradient bar.
function dense_reml(θ, y)
    T = eltype(θ)
    βμ = θ[1:2]; βσ = θ[3:4]; α = θ[5:6]
    r = y .- Xmu * βμ
    d = exp.(2 .* (Xmu * βσ)); sb2 = exp.(2 .* (Zg * α))
    ld = zero(T); quad = zero(T); XtVX = zeros(T, 2, 2)
    for k in 1:n_id
        rows = ((k - 1) * n_each + 1):(k * n_each)
        Vk = Matrix(Diagonal(d[rows])) .+ sb2[k]
        F = cholesky(Symmetric(Vk))
        ld += logdet(F)
        quad += dot(r[rows], F \ r[rows])
        Xk = Xmu[rows, :]
        XtVX .+= Xk' * (F \ Xk)
    end
    n = length(y)
    return 0.5 * (ld + quad + logdet(cholesky(Symmetric(XtVX)))) + 0.5 * (n - 2) * log(2π)
end

# Verified best optima (multi-start, both Julia versions, after the fix).
const CASES = [
    ("y_mt20260715_julia1.13.txt", -408.1790006287666),   # threw on 1.10 before the fix
    ("y_mt20260715_julia1.10.txt", -412.3006010571039),
]

@testset "lss REML sd(id) ~ sex: same data, same optimum on every Julia" begin
    for (file, best) in CASES
        y = vec(readdlm(joinpath(FIXDIR, file), Float64))
        fit = drm(bf(@formula(y ~ sex + (1 | id)), @formula(sigma ~ sex), @formula(sd(id) ~ sex)),
                  Gaussian(); data = (; y, sex = sexl, id), method = :REML)
        ll = reml_loglik(fit)
        @test fit.converged
        @test abs(ll - best) < 1e-4
        # Reported ll is the independent objective's value at theta-hat ...
        @test isapprox(-dense_reml(fit.theta, y), ll; atol = 1e-8, rtol = 0)
        # ... and theta-hat is stationary for it (LBFGS g_tol is 1e-8).
        g = ForwardDiff.gradient(θ -> dense_reml(θ, y), fit.theta)
        @test maximum(abs, g) < 1e-6
    end
end

@testset "_re_xtvinvx_stable is PSD and accurate at the probe that broke 1.10" begin
    # LBFGS's failing probe on the 1.13 draw (theta-hat is irrelevant; only the
    # variance parameters enter X'V^-1X).
    βσ = [-105.18645268489132, -37.44440105570058]
    α = [12.448110904762649, -2.6897639483699582]
    invD = exp.(-2 .* (Xmu * βσ)); invσb2 = exp.(-2 .* (Zg * α))
    S = zeros(n_id)
    for i in eachindex(id); S[id[i]] += invD[i]; end
    A = DRModels._re_xtvinvx_stable(Xmu, invD, nothing, id, invσb2, S)
    # Exact value, per group: X_k'(D_k + sigma_b,k^2 11')^-1 X_k with sex constant
    # within group, = x_k x_k' * s_k / (1 + sigma_b,k^2 s_k), s_k = sum_i 1/D_i.
    Aex = zeros(BigFloat, 2, 2)
    for k in 1:n_id
        xk = big.([1.0, sex[k]])
        sk = sum(exp(-2 * big(dot(Xmu[i, :], βσ))) for i in findall(==(k), id))
        Aex .+= xk * xk' .* (sk / (1 + exp(2 * big(dot(Zg[k, :], α))) * sk))
    end
    @test all(isfinite, A)
    @test isapprox(A, Float64.(Aex); rtol = 1e-6)
    # Unchanged at an ordinary interior point (matches the Woodbury form there).
    βσ2 = [-0.8, 0.3]; α2 = [-0.4, -0.2]
    invD2 = exp.(-2 .* (Xmu * βσ2)); invσb22 = exp.(-2 .* (Zg * α2))
    S2 = zeros(n_id); Z2 = zeros(n_id, 2)
    for i in eachindex(id); S2[id[i]] += invD2[i]; Z2[id[i], :] .+= invD2[i] .* Xmu[i, :]; end
    W = Xmu' * (invD2 .* Xmu)
    for k in 1:n_id; W .-= Z2[k, :] * Z2[k, :]' ./ (invσb22[k] + S2[k]); end
    @test isapprox(DRModels._re_xtvinvx_stable(Xmu, invD2, nothing, id, invσb22, S2), W; rtol = 1e-12)
end

@testset "#835 guard: _re_quad_stable with sigma_b,k = 0 exactly (invsigb2 = Inf)" begin
    # Two groups; group 1 has zero random-effect variance, so its rows contribute
    # the plain r'D^-1 r; group 2 is an ordinary random intercept.
    r = [0.3, -0.5, 1.1, 0.2, -0.7]
    invD = [2.0, 1.5, 0.8, 1.2, 3.0]
    gidx = [1, 1, 2, 2, 2]
    invσb2 = [Inf, 1 / 0.49]
    S = [invD[1] + invD[2], invD[3] + invD[4] + invD[5]]
    C = [r[1] * invD[1] + r[2] * invD[2], r[3] * invD[3] + r[4] * invD[4] + r[5] * invD[5]]
    q = DRModels._re_quad_stable(r, invD, nothing, gidx, invσb2, S, C)
    V2 = Matrix(Diagonal(1 ./ invD[3:5])) .+ 0.49
    exact = r[1]^2 * invD[1] + r[2]^2 * invD[2] + dot(r[3:5], V2 \ r[3:5])
    @test isfinite(q)            # NaN before #835's guard (0 * Inf)
    @test isapprox(q, exact; rtol = 1e-12)
end

end # module
