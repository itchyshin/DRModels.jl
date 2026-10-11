# #977: the Gaussian multi-component scalar RE route `(1 | g) + (1 | h) + …`
# moved from a dense QR of [D^{-1/2}Z̃; I] to a sparse QR + Takahashi selected
# inverse (src/gaussian_multi_re_sparse.jl). The owner decision (11 Oct 2026,
# decision 27a) is that estimates must NOT change. This file keeps a verbatim
# copy of the pre-#977 dense objective and gradient as the reference, runs the
# SAME optimiser from the SAME start, and pins logLik and every coefficient to
# 1e-8 on several fixtures (crossed, three components, a varying-σ design, a
# weighted (0 + x | h) component, and q > the dense-Hessian threshold).
using DRModels, Test, LinearAlgebra, Random, SparseArrays, Statistics, ForwardDiff
const Optim = DRModels.Optim

const _SK = DRModels

# Pre-#977 dense objective / gradient (copied from src/gaussian_ranef.jl at
# origin/main 740e735), returned as closures over a dense Z.
function _sk_dense_route(y, Xμ, Xσ, comps)
    n = length(y); pμ, pσ = size(Xμ, 2), size(Xσ, 2)
    K = length(comps)
    Gks = [c[3] for c in comps]; q = sum(Gks); offs = cumsum([0; Gks])
    Z = zeros(n, q); colcomp = Vector{Int}(undef, q)
    for (k, (w, gidx, Gk, _)) in enumerate(comps)
        for c in 1:Gk
            colcomp[offs[k]+c] = k
        end
        for i in 1:n
            Z[i, offs[k]+gidx[i]] = w[i]
        end
    end
    function nll(θ)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ησ = clamp.(Xσ * βσ, -30.0, 30.0)
        invD = exp.(-2 .* ησ); r = y .- Xμ * βμ
        σk = [exp(clamp(θ[pμ+pσ+k], -30.0, 30.0)) for k in 1:K]
        Z̃ = Z .* [σk[colcomp[c]] for c in 1:q]'
        _, bhat, logdetM = _SK._multi_re_qr(Z̃, exp.(-ησ), r)
        e = r .- Z̃ * bhat
        val = 0.5 * (sum(2 .* ησ) + logdetM + sum(invD .* e .^ 2) + sum(abs2, bhat)) + 0.5 * n * log(2π)
        isfinite(val) || return oftype(val, 1e18)
        return val
    end
    function grad!(G, θ)
        fill!(G, 0.0)
        βμ = θ[1:pμ]; βσ = θ[pμ+1:pμ+pσ]
        ησ = clamp.(Xσ * βσ, -30.0, 30.0)
        invD = exp.(-2 .* ησ); r = y .- Xμ * βμ
        σk = [exp(clamp(θ[pμ+pσ+k], -30.0, 30.0)) for k in 1:K]
        Z̃ = Z .* [σk[colcomp[c]] for c in 1:q]'
        R, b, _ = _SK._multi_re_qr(Z̃, exp.(-ησ), r)
        Rinv = UpperTriangular(R) \ Matrix{Float64}(I, q, q)
        Hinv = Rinv * Rinv'
        α = invD .* (r .- Z̃ * b)
        G[1:pμ] .= -(Xμ' * α)
        lever = vec(sum((Z̃ * Hinv) .* Z̃, dims = 2))
        G[pμ+1:pμ+pσ] .= Xσ' * (((invD .- invD .^ 2 .* lever) .- α .^ 2) ./ invD)
        dH = diag(Hinv)
        for k in 1:K
            cols = (offs[k]+1):offs[k+1]
            G[pμ+pσ+k] = sum(1 .- dH[cols]) - sum(abs2, b[cols])
        end
        return G
    end
    βμ0 = Xμ \ y; s0 = std(y - Xμ * βμ0) / sqrt(K + 1)
    θ0 = zeros(pμ + pσ + K); θ0[1:pμ] .= βμ0; θ0[pμ+1] = log(s0 + eps())
    θ0[pμ+pσ+1:end] .= log(s0 + eps())
    return nll, grad!, θ0
end

function _sk_crossed(rng, n, G, H; x2 = false)
    g = rand(rng, 1:G, n); h = rand(rng, 1:H, n); x = randn(rng, n)
    sx = x2 ? 0.3 .* randn(rng, n) : zeros(n)
    y = 1.0 .+ 0.5 .* x .+ 0.7 .* randn(rng, G)[g] .+ 0.4 .* randn(rng, H)[h] .+
        exp.(-0.5 .+ sx) .* randn(rng, n)
    return y, g, h, x, sx
end

_sk_names(p) = ["b$j" for j in 1:p]

function _sk_check(y, Xμ, Xσ, comps; nlltol = 1e-8, coeftol = 1e-8)
    fit = _SK._fit_multi_ranef_gaussian(Gaussian(), y, Xμ, Xσ, comps,
                                       _sk_names(size(Xμ, 2)), _sk_names(size(Xσ, 2)), 1e-8)
    nll0, g0!, θ0 = _sk_dense_route(y, Xμ, Xσ, comps)
    # objective + gradient identity at the start and at random nearby points
    rng = MersenneTwister(5)
    for θ in (θ0, fit.theta, fit.theta .+ 0.3 .* randn(rng, length(θ0)))
        @test isapprox(fit.nll(θ), nll0(θ); rtol = 1e-11)
        ga = zeros(length(θ)); gb = zeros(length(θ))
        fit.nllgrad(ga, θ); g0!(gb, θ)
        @test isapprox(ga, gb; rtol = 1e-8, atol = 1e-9)
    end
    # same optimiser, same start, dense reference objective
    res = Optim.optimize(Optim.OnceDifferentiable(nll0, g0!, θ0), θ0, Optim.LBFGS(),
                         Optim.Options(g_tol = 1e-8))
    θref = Optim.minimizer(res)
    @test abs(fit.loglik - (-Optim.minimum(res))) < nlltol
    @test maximum(abs, fit.theta .- θref) < coeftol
    return fit, θref, nll0
end

@testset "#977 sparse multi-RE kernel: identity with the dense route" begin
    rng = MersenneTwister(977)

    @testset "crossed (1|g)+(1|h), sigma ~ 1" begin
        n = 400; y, g, h, x, _ = _sk_crossed(rng, n, 40, 25)
        comps = [(ones(n), g, 40, "g"), (ones(n), h, 25, "h")]
        fit, θref, nll0 = _sk_check(y, hcat(ones(n), x), ones(n, 1), comps)
        @test fit.converged
        # q = 65 ≤ the dense-Hessian threshold: SEs are the same ForwardDiff
        # Hessian of the same objective as before.
        Vref = inv(Symmetric(ForwardDiff.hessian(nll0, fit.theta)))
        @test isapprox(fit.vcov, Vref; rtol = 1e-8)
        # BLUPs: same penalised least-squares mode as the dense route
        re = ranef(fit)
        @test length(re[:g]) == 40 && length(re[:h]) == 25
        # BLUPs equal the dense GLS conditional means at θ̂: b̂ = σ_col ⊙ M⁻¹ Z̃ᵀD⁻¹r
        let θ = fit.theta, pμ = 2
            r = y .- hcat(ones(n), x) * θ[1:pμ]
            ησ = θ[pμ+1]; σcol = vcat(fill(exp(θ[pμ+2]), 40), fill(exp(θ[pμ+3]), 25))
            Zt = Matrix(sparse(repeat(1:n, 2), vcat(g, 40 .+ h), 1.0, n, 65)) .* σcol'
            M = I + Zt' * Zt .* exp(-2ησ)
            bd = σcol .* (M \ (Zt' * r .* exp(-2ησ)))
            @test maximum(abs, vcat(re[:g], re[:h]) .- bd) < 1e-8
        end
    end

    @testset "three components + varying sigma" begin
        n = 500; y, g, h, x, sx = _sk_crossed(rng, n, 30, 20; x2 = true)
        a = rand(rng, 1:12, n)
        y .+= 0.5 .* randn(rng, 12)[a]
        comps = [(ones(n), g, 30, "g"), (ones(n), h, 20, "h"), (ones(n), a, 12, "a")]
        _sk_check(y, hcat(ones(n), x), hcat(ones(n), sx), comps)
    end

    @testset "weighted (0 + x | h) component" begin
        n = 300; y, g, h, x, _ = _sk_crossed(rng, n, 25, 10)
        y .+= x .* 0.4 .* randn(rng, 10)[h]
        comps = [(ones(n), g, 25, "g"), (x, h, 10, "h")]
        _sk_check(y, hcat(ones(n), x), ones(n, 1), comps)
    end

    @testset "q above the dense-Hessian threshold (finite-difference vcov)" begin
        n = 1200; y, g, h, x, _ = _sk_crossed(rng, n, 220, 60)
        comps = [(ones(n), g, 220, "g"), (ones(n), h, 60, "h")]
        @test 280 > _SK._MULTI_RE_DENSE_HESSIAN_MAXQ
        fit, θref, nll0 = _sk_check(y, hcat(ones(n), x), ones(n, 1), comps)
        Vref = inv(Symmetric(ForwardDiff.hessian(nll0, fit.theta)))
        @test isapprox(sqrt.(diag(fit.vcov)), sqrt.(diag(Vref)); rtol = 1e-5)
    end

    @testset "factor + selected inverse = dense, both routes" begin
        n = 300; y, g, h, x, _ = _sk_crossed(rng, n, 50, 30)
        des = _SK._multi_re_design([(ones(n), g, 50, "g"), (ones(n), h, 30, "h")], n)
        sd = exp.(0.2 .* randn(rng, n))
        Z̃ = des.Z * Diagonal(vcat(fill(1.3, 50), fill(0.6, 30)))
        M = I + Matrix(Z̃)' * Diagonal(sd .^ 2) * Matrix(Z̃)
        Minv = inv(M)
        for allow_chol in (true, false)
            fac = _SK._multi_re_factor(Z̃, sd, y; allow_chol = allow_chol)
            @test fac.route === (allow_chol ? :cholesky : :qr)
            @test fac.logdetM ≈ logdet(M) rtol = 1e-12
            @test fac.bhat ≈ M \ (Matrix(Z̃)' * (sd .^ 2 .* y)) rtol = 1e-10
            S = _SK._multi_re_selinv(fac, des.Mpat)
            @test maximum(abs, _SK._selinv_diag(S, 80) .- diag(Minv)) < 1e-12
            @test maximum(abs, [_SK._selinv_get(S, g[i], 50 + h[i]) - Minv[g[i], 50+h[i]] for i in 1:n]) < 1e-12
        end
        # the gate: an extreme σ_e/σ_k ratio makes max diag(M) huge -> QR route
        fac = _SK._multi_re_factor(Z̃, fill(1e6, n), y)
        @test fac.route === :qr
    end

    @testset "both routes give the same fit (gate forced either way)" begin
        n = 400; y, g, h, x, _ = _sk_crossed(rng, n, 40, 25)
        comps = [(ones(n), g, 40, "g"), (ones(n), h, 25, "h")]
        args = (Gaussian(), y, hcat(ones(n), x), ones(n, 1), comps, ["a", "b"], ["s"], 1e-8)
        f1 = _SK._fit_multi_ranef_gaussian(args...)
        old = _SK._MULTI_RE_CHOL_MAXDIAG[]
        _SK._MULTI_RE_CHOL_MAXDIAG[] = 0.0          # force the QR route
        f2 = try _SK._fit_multi_ranef_gaussian(args...) finally _SK._MULTI_RE_CHOL_MAXDIAG[] = old end
        @test abs(f1.loglik - f2.loglik) < 1e-8
        @test maximum(abs, f1.theta .- f2.theta) < 1e-6
    end
end
