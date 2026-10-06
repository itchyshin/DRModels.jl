using DRModels
using Test, TOML, LinearAlgebra, SparseArrays, SpecialFunctions, ForwardDiff
using LinearAlgebra: issuccess

# Independent analytic Gamma derivatives; P and eta0/psi0 are lifted exactly
# from their Float64 inputs before multiplying or accumulating. This isolates
# gradient evaluation from the separate precision-construction question.
function _test_gamma_fixed_precision_gradient(y, eta0, psi0, gidx, a, P, Ze, Zp)
    setprecision(BigFloat, 256) do
        ab = BigFloat.(a)
        g = BigFloat.(P) * ab
        for i in eachindex(y)
            j = 2gidx[i] - 1
            ze = BigFloat.(Ze[i, :]); zp = BigFloat.(Zp[i, :])
            eta = BigFloat(eta0[i]) + dot(ze, ab[j:j+1])
            psi = BigFloat(psi0[i]) + dot(zp, ab[j:j+1])
            shape = exp(psi); ratio = BigFloat(y[i]) / exp(eta)
            ge = shape * (1 - ratio)
            gp = shape * (digamma(shape) - psi + eta - 1 - log(BigFloat(y[i])) + ratio)
            g[j:j+1] .+= ge .* ze .+ gp .* zp
        end
        return g
    end
end

@testset "location-scale compensated arithmetic" begin
@testset "fixed-precision Gamma inner gradient at retained failed point" begin
    f = TOML.parsefile(joinpath(@__DIR__, "fixtures", "locscale_precision",
                               "locscale_gamma_l21.toml"))
    # Exact binary coordinates from no-refit receipt1001c0ac (2026-08-31).
    theta = [0.6370343554783728, 0.2441002825139284, 1.475213074870776,
             -1.6628786422610105, -0.09608128405153676, -9.126034773908929]
    failed_a = [-0.19461138381554569, 0.09862478013277647,
                 0.13400282799345947, -0.06790971799124475,
                 0.12408946215539773, -0.06288579669679849,
                -0.10754978308685482, 0.05450388751680325]
    y, gidx, G = f["y"], f["gidx"], f["G"]
    eta0 = hcat(ones(length(y)), f["x"]) * theta[1:2]
    psi0 = fill(theta[3], length(y))
    P = DRModels.prior_precision(sparse(1.0I, G, G), DRModels._ls_inv2x2(DRModels._ls_lc_to_Λ(theta[4:6])))
    Ze = DRModels._ls_canonical_Zeta(length(y)); Zp = DRModels._ls_canonical_Zpsi(length(y))
    reference = _test_gamma_fixed_precision_gradient(y, eta0, psi0, gidx, failed_a, P, Ze, Zp)
    actual = DRModels._ls_joint_grad(Val(:gamma), y, eta0, psi0, gidx, failed_a, P, Ze, Zp)
    @test maximum(abs, BigFloat.(actual) - reference) < big"1e-12"
    # The old rounded gradient incorrectly certified this separate point.
    false_accepted = [-0.19461138381538123, 0.09862478013269314,
                      0.13400282798766738, -0.06790971798830944,
                      0.12408946215686391, -0.06288579669754152,
                      -0.10754978309036813, 0.05450388751858372]
    false_residual = _test_gamma_fixed_precision_gradient(y, eta0, psi0, gidx, false_accepted, P, Ze, Zp)
    @test norm(false_residual) > big"1e-9" * (1 + norm(BigFloat.(false_accepted)))
    _, certified = DRModels._ls_inner_certificate(Val(:gamma), y, eta0, psi0, gidx,
                                             G, P, Ze, Zp, false_accepted, 1e-9)
    @test !certified
    mode, factor, ok = DRModels._ls_inner_mode(Val(:gamma), y, eta0, psi0, gidx, G, P, Ze, Zp)
    @test ok
    @test factor !== nothing && issuccess(factor)
    exact_residual = _test_gamma_fixed_precision_gradient(y, eta0, psi0, gidx, mode, P, Ze, Zp)
    @test norm(exact_residual) <= big"1e-9" * (1 + norm(BigFloat.(mode)))
end

@testset "fixed-precision gradient preserves general loadings and dual inputs" begin
    G = 3; n = 9
    gidx = repeat(1:G, inner=3)
    Q = sparse(SymTridiagonal(fill(2.0, G), fill(-0.3, G-1)))
    P = DRModels.prior_precision(Q, [2.0 0.2; 0.2 1.5])
    a = [0.2, -0.1, 0.05, 0.1, -0.2, 0.15]
    y = collect(range(0.7, 2.3; length=n))
    eta0 = fill(0.2, n); psi0 = fill(0.5, n)
    Ze = hcat(ones(n), collect(range(-0.5, 0.5; length=n)))
    Zp = hcat(fill(0.25, n), ones(n))
    actual = DRModels._ls_joint_grad(Val(:gamma), y, eta0, psi0, gidx, a, P, Ze, Zp)
    reference = _test_gamma_fixed_precision_gradient(y, eta0, psi0, gidx, a, P, Ze, Zp)
    @test maximum(abs, BigFloat.(actual) - reference) < big"1e-12"
    J = ForwardDiff.jacobian(x -> DRModels._ls_joint_grad(Val(:gamma), y, eta0, psi0, gidx, x, P, Ze, Zp), a)
    H = DRModels._ls_joint_hess(Val(:gamma), y, eta0, psi0, gidx, G, a, P, Ze, Zp)
    @test J ≈ Matrix(H) rtol=1e-12 atol=1e-12
    objective_gradient = ForwardDiff.gradient(x -> DRModels._ls_joint(Val(:gamma), y, eta0, psi0, gidx, x, P, Ze, Zp), a)
    @test maximum(abs, BigFloat.(objective_gradient) - reference) < big"1e-12"
    # Primal specialization with nonidentity Q and noncanonical loadings.
    frozen_objective = setprecision(BigFloat, 256) do
        total = dot(BigFloat.(a), BigFloat.(P) * BigFloat.(a)) / 2
        for i in eachindex(y)
            j = 2gidx[i] - 1
            eta = eta0[i] + Ze[i,1]*a[j] + Ze[i,2]*a[j+1]
            psi = psi0[i] + Zp[i,1]*a[j] + Zp[i,2]*a[j+1]
            total += BigFloat(DRModels._ls_nll(Val(:gamma), y[i], eta, psi))
        end
        total
    end
    primal = DRModels._ls_joint(Val(:gamma), y, eta0, psi0, gidx, a, P, Ze, Zp)
    @test abs(BigFloat(primal) - frozen_objective) <= 2eps(primal)


end

@testset "fixed-precision joint objective retains Newton descent" begin
    f = TOML.parsefile(joinpath(@__DIR__, "fixtures", "locscale_precision", "locscale_gamma_l21.toml"))
    theta = [0.6370343554783728, 0.2441002825139284, 1.475213074870776,
             -1.6628786422610105, -0.09608128405153676, -9.126034773908929]
    # Immutable no-fit direction receipt174337Z: same Float64 points and P.
    a = [-0.19461152617192382, 0.09862485227590727, 0.13400282330226113,
         -0.06790971561384698, 0.12408945925318163, -0.06288579522601816,
         -0.10754978431388457, 0.05450388813863535]
    trial = [-0.1946113838135039, 0.09862478013174175, 0.1340028279892676,
             -0.0679097179891204, 0.12408946215195589, -0.06288579669505424,
             -0.10754978308506444, 0.05450388751589591]
    y, gidx, G = f["y"], f["gidx"], f["G"]
    eta0 = hcat(ones(length(y)), f["x"]) * theta[1:2]
    psi0 = fill(theta[3], length(y))
    P = DRModels.prior_precision(sparse(1.0I, G, G), DRModels._ls_inv2x2(DRModels._ls_lc_to_Λ(theta[4:6])))
    Ze = DRModels._ls_canonical_Zeta(length(y)); Zp = DRModels._ls_canonical_Zpsi(length(y))
    # Reference isolates multiplication and accumulation error: lift each
    # unchanged Float64 kernel output, but evaluate the prior exactly in BigFloat.
    frozen_reference(v) = setprecision(BigFloat, 256) do
        vb = BigFloat.(v)
        total = dot(vb, BigFloat.(P) * vb) / 2
        for i in eachindex(y)
            j = 2gidx[i] - 1
            total += BigFloat(DRModels._ls_nll(Val(:gamma), y[i], eta0[i] + v[j], psi0[i] + v[j+1]))
        end
        total
    end
    actual(v) = DRModels._ls_joint(Val(:gamma), y, eta0, psi0, gidx, v, P, Ze, Zp)
    for v in (a, trial)
        @test abs(BigFloat(actual(v)) - frozen_reference(v)) <= 2eps(actual(v))
    end
    @test frozen_reference(trial) < frozen_reference(a)
    @test actual(trial) < actual(a)
end
end # all arithmetic tests: retain every failure, including neighbours.
