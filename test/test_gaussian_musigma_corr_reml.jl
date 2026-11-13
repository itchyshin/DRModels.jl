# REML for the Gaussian correlated mean + sigma random-intercept pair
#   y ~ x... + (1 | p | g),  sigma ~ 1 [+ x1] + (1 | p | g),  method = :REML      (gap-map G6b)
# Gap-map cell `gaussian|ordinary|lab0|q1|sd-none|cm1|REML` (R fits it, Julia used to refuse it).
#
# Native drmTMB (REML = TRUE) moves beta_mu AND beta_sigma into TMB's random block with flat priors,
# so its objective is ONE Laplace approximation over w = (va, vb, beta_mu, beta_sigma)
# (`estimator_exact = FALSE`, summary: "Laplace/Cox-Reid adjusted profile"):
#   nll_R(alpha) = F(w^) + 1/2 log det H(w^) + (G/2) log(1 - rho^2) - (p/2) log 2pi.
#
# Two checks, thresholds fixed BEFORE the three fixtures were fitted:
#   (1) agreement with drmTMB 0.8.1 REML (`_GMC_R`, produced by
#       test/fixtures/gaussian_musigma_corr_reml_r_reference.R from the committed data CSVs):
#       |logLik_J - logLik_R| <= 1e-4; every coefficient within 1e-3 R-SEs; log SDs and eta within 5e-3.
#       (R's is_converged() was TRUE on all three; a FALSE cell would be flagged, not counted.)
#   (2) an independent dense oracle (own Newton, ForwardDiff derivatives, no DRModels internals):
#       reml_loglik and ml_loglik equal the oracle's restricted / ML Laplace values to 1e-8, and the
#       restricted value at the fitted alpha is not beaten by perturbing alpha.
using DRModels, Test, LinearAlgebra, ForwardDiff, DelimitedFiles, Statistics, Logging

const _GMC_DIR = joinpath(@__DIR__, "fixtures", "gaussian_musigma_corr_reml")
const _GMC_R = Dict(
    :harness => (ll = -489.7542968591,
        est = [0.2877576707, 0.4238241883, 0.2662215849, -0.120582065, -0.4172500525, -0.7250439983, -1.4987881063, -0.1430622434],
        se = [0.083766844, 0.0387964056, 0.0398775718, 0.0355832461, 0.0531748199, 0.1414944162, 0.250797464, 0.2636548426]),
    :s1 => (ll = -295.3300180418,
        est = [0.2434675746, 0.3587600612, -0.3192234995, -0.550374467, -0.4197128688, -0.7551193459, 0.6903557107],
        se = [0.1280165479, 0.0399854854, 0.035330287, 0.0994305032, 0.1616941828, 0.1851922992, 0.2563195036]),
    :s2 => (ll = -377.9756786738,
        est = [0.6461770738, 0.4811524935, -0.3092167852, -0.5467481256, 0.3827666105, -0.6104338841, -1.2336468575, -0.7804085566],
        se = [0.0866102495, 0.0357363626, 0.0360973444, 0.0748149912, 0.0707737655, 0.128013478, 0.2136279382, 0.4317981535]),
)

function _gmc_load(cell)
    raw, hdr = readdlm(joinpath(_GMC_DIR, "data_$(cell).csv"), ','; header = true)
    col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
    d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")),
         x3 = Float64.(col("x3")), g = String.(col("g")))
    return d
end

_gmc_formula(cell) = cell === :harness ?
    bf(@formula(y ~ 1 + x1 + x2 + x3 + (1 | p | g)), @formula(sigma ~ 1 + (1 | p | g))) :
    cell === :s1 ? bf(@formula(y ~ 1 + x1 + x2 + (1 | p | g)), @formula(sigma ~ 1 + (1 | p | g))) :
                   bf(@formula(y ~ 1 + x1 + x2 + (1 | p | g)), @formula(sigma ~ 1 + x1 + (1 | p | g)))

# Independent oracle: restricted and ML Laplace values at alpha = (log s_mu, log s_sigma, eta).
function _gmc_oracle(d, Xμ, Xσ, α; restricted::Bool, βfixed = nothing)
    gl = unique(d.g); G = length(gl); gi = [findfirst(==(s), gl) for s in d.g]
    n = length(d.y); pμ = size(Xμ, 2); pσ = size(Xσ, 2); p = pμ + pσ
    c, dd, ρ = exp(α[1]), exp(α[2]), tanh(α[3])
    function F(w)
        va = w[1:G]; vb = w[G+1:2G]
        lin = restricted ? Xμ * w[2G+1:2G+pμ] : Xμ * βfixed[1:pμ]
        eta = restricted ? Xσ * w[2G+pμ+1:2G+p] : Xσ * βfixed[pμ+1:p]
        r = d.y .- lin .- c .* va[gi]; e = eta .+ dd .* vb[gi]
        s = sum(e) + 0.5 * sum(abs2, r .* exp.(-e)) + 0.5 * n * log(2π)
        return s + 0.5 / (1 - ρ^2) * (sum(abs2, va) - 2ρ * dot(va, vb) + sum(abs2, vb))
    end
    q = restricted ? 2G + p : 2G
    w = zeros(q)
    if restricted
        β0 = Xμ \ d.y; w[2G+1:2G+pμ] .= β0; w[2G+pμ+1] = log(0.7 * std(d.y - Xμ * β0))
    end
    for _ in 1:400
        g = ForwardDiff.gradient(F, w); H = Symmetric(ForwardDiff.hessian(F, w))
        norm(g) < 1e-10 && break
        λ = 0.0
        while !isposdef(H + λ * I)
            λ = max(2λ, 1e-4)
        end
        step = (H + λ * I) \ g
        t = 1.0; f0 = F(w)
        # `+ 1e-12`: near the mode the predicted decrease (about |g|^2, below 1e-14) is under the rounding
        # noise of F (about 5e-14 at F near 300), so a strict Armijo test rejects the exact Newton step and the
        # loop stalls at |g| of 1e-8 to 6e-8 on Julia 1.10. The final |g| < 1e-8 assertion is unchanged.
        while !(F(w .- t .* step) <= f0 - 1e-4 * t * dot(g, step) + 1e-12) && t > 1e-12
            t /= 2
        end
        w .-= t .* step
    end
    @assert norm(ForwardDiff.gradient(F, w)) < 1e-8 "oracle inner Newton did not converge"
    @assert isposdef(Symmetric(ForwardDiff.hessian(F, w))) "oracle inner mode is not a minimum"
    H = ForwardDiff.hessian(F, w)
    v = F(w) + 0.5 * logdet(Symmetric(H)) + 0.5 * G * log(1 - ρ^2)
    return restricted ? v - 0.5 * p * log(2π) : v
end

@testset "REML Gaussian (1 | p | g) in mu and sigma vs drmTMB 0.8.1 and a dense oracle" begin
    for cell in (:harness, :s1, :s2)
        @testset "$cell" begin
            d = _gmc_load(String(cell)); R = _GMC_R[cell]
            fit = drm(_gmc_formula(cell), Gaussian(); data = d, method = :REML)
            @test estimation_method(fit) === :REML
            @test fit.converged
            @test isfinite(reml_loglik(fit)) && isfinite(ml_loglik(fit))
            @test loglik(fit) == reml_loglik(fit)
            @test abs(reml_loglik(fit) - R.ll) <= 1e-4
            θ = fit.theta; p = length(θ) - 3
            @test length(R.est) == length(θ)
            sdr = sqrt.(diag(fit.vcov))
            for i in 1:p
                @test abs(θ[i] - R.est[i]) <= 1e-3 * R.se[i]
            end
            for i in (p+1):(p+3)
                @test abs(θ[i] - R.est[i]) <= 5e-3
            end
            # Julia order is [beta_mu; beta_sigma; log s_mu; log s_sigma; eta], same as `_GMC_R`.
            @test all(isfinite, sdr)
            @test all(abs.(sdr[1:p] ./ R.se[1:p] .- 1) .< 5e-3)

            # (2) independent oracle
            Xμ = d.x3 isa Vector && cell === :harness ? hcat(ones(length(d.y)), d.x1, d.x2, d.x3) :
                 hcat(ones(length(d.y)), d.x1, d.x2)
            Xσ = cell === :s2 ? hcat(ones(length(d.y)), d.x1) : ones(length(d.y), 1)
            α = θ[p+1:p+3]
            @test reml_loglik(fit) ≈ -_gmc_oracle(d, Xμ, Xσ, α; restricted = true) atol = 1e-8
            @test ml_loglik(fit) ≈ -_gmc_oracle(d, Xμ, Xσ, α; restricted = false, βfixed = θ[1:p]) atol = 1e-8
            base = _gmc_oracle(d, Xμ, Xσ, α; restricted = true)
            for k in 1:3, δ in (-0.05, 0.05)
                αp = copy(α); αp[k] += δ
                @test _gmc_oracle(d, Xμ, Xσ, αp; restricted = true) >= base - 1e-7
            end
        end
    end

    @testset "ML route unchanged and REML != ML" begin
        d = _gmc_load("s1")
        ml = drm(_gmc_formula(:s1), Gaussian(); data = d)
        rm = drm(_gmc_formula(:s1), Gaussian(); data = d, method = :REML)
        @test estimation_method(ml) === :ML
        @test isnan(reml_loglik(ml))
        @test loglik(rm) != loglik(ml)
    end

    @testset "neighbouring shapes stay refused under REML" begin
        d = _gmc_load("s1")
        # slope in a labelled pair, labelled in mu only, and an sd() formula are not the admitted cell
        @test_throws ArgumentError drm(bf(@formula(y ~ 1 + x1 + (1 + x2 | p | g)),
            @formula(sigma ~ 1 + (1 + x2 | p | g))), Gaussian(); data = d, method = :REML)
        @test_throws Exception drm(bf(@formula(y ~ 1 + x1 + (1 | p | g)), @formula(sigma ~ 1 + (1 | g))),
            Gaussian(); data = d, method = :REML)
        @test_throws Exception drm(bf(@formula(y ~ 1 + x1 + (1 | p | g)), @formula(sigma ~ 1 + (1 | p | g)),
            @formula(sd(g) ~ 1)), Gaussian(); data = d, method = :REML)
    end
end

# Rows with a missing response are dropped before the fit (with a warning that gives the count), as on the
# other random-effect routes (#1089). The fit equals the fit on the complete rows, under REML and ML.
# `docs/src/capabilities.md` says so; the second test keeps that sentence true.
@testset "REML (1 | p | g) with missing responses: dropped, equal to the complete-case fit" begin
    d = _gmc_load("s1")
    n = length(d.y)
    ymiss = Vector{Union{Missing,Float64}}(d.y)
    miss = collect(7:17:n)
    ymiss[miss] .= missing
    keep = setdiff(1:n, miss)
    dm = (; y = ymiss, x1 = d.x1, x2 = d.x2, x3 = d.x3, g = d.g)
    dc = (; y = d.y[keep], x1 = d.x1[keep], x2 = d.x2[keep], x3 = d.x3[keep], g = d.g[keep])
    for meth in (:REML, :ML)
        fit = @test_logs (:warn, r"missing/NaN response") match_mode = :any drm(_gmc_formula(:s1), Gaussian(); data = dm, method = meth)
        ref = drm(_gmc_formula(:s1), Gaussian(); data = dc, method = meth)
        @test fit.converged && ref.converged
        @test fit.theta == ref.theta
        @test fit.loglik == ref.loglik
    end
    caps = read(joinpath(@__DIR__, "..", "docs", "src", "capabilities.md"), String)
    row = only(filter(l -> occursin("labelled pair `(1 \\| p \\| g)` in both `mu` and `sigma` (fixed-effect predictors, no `sd()` formula) with `method = :REML`", l),
                      split(caps, '\n')))
    @test !occursin(r"missing responses[^.]*refused", row)
    @test occursin("missing response are dropped", row)
end

# Correlation near the boundary. On these two datasets drmTMB 0.8.1 converges (REML log-likelihood -235.221 and
# -443.4403, eta_cor -10 and 10.91) while this route does not reach a usable optimum: the correlation runs to
# the edge, the objective is almost flat there, and the standard errors are not finite. The ML route on the
# same data fails too (it is the shared engine, not this slice). The contract tested here is honesty, not
# agreement: the fit is not reported as converged, no failure sentinel is shown as a log-likelihood, and the
# standard errors are not finite numbers. If the optimiser is later fixed, the first two assertions of each
# branch still hold.
@testset "REML (1 | p | g) near the correlation boundary: an honest failure, never a sentinel logLik" begin
    for case in ("b2", "b4")
        raw, hdr = readdlm(joinpath(_GMC_DIR, "data_$(case).csv"), ','; header = true)
        col(nm) = raw[:, findfirst(==(nm), vec(hdr))]
        d = (; y = Float64.(col("y")), x1 = Float64.(col("x1")), x2 = Float64.(col("x2")), g = String.(col("g")))
        f = bf(@formula(y ~ x1 + x2 + (1 | p | g)), @formula(sigma ~ 1 + x1 + (1 | p | g)))
        fit = Logging.with_logger(Logging.NullLogger()) do
            drm(f, Gaussian(); data = d, method = :REML)
        end
        ll = Logging.with_logger(Logging.NullLogger()) do
            loglikelihood(fit)
        end
        @test isnan(ll) || ll > -1e15                      # the -1e18 failure value is never shown as a logLik
        @test ll != -1e18
        if is_converged(fit)
            @test isfinite(ll) && all(isfinite, stderror(fit))   # a real fit is allowed
        else
            @test string(convergence_status(fit)) != "converged"
            @test !all(isfinite, stderror(fit))                  # Inf, never a made-up finite SE
        end
        @test !(is_converged(fit) && isnan(ll))
    end
end
