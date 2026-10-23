using DRModels
using Test, TOML, LinearAlgebra, SparseArrays
using SpecialFunctions: digamma

# This self-contained literal was exported from the independent 128/256-bit
# reference receipts, SHA90469e7c304453c0e400d4c19897e263b61beb0912c0a1c5bc677b4f39fee6ad.
const _WHITENED_BOUNDARY_TOML = raw"""
G = 4
Q = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
Xmu = [[1.0, -1.0], [1.0, -0.7142857142857143], [1.0, -0.42857142857142855], [1.0, -0.14285714285714285], [1.0, 0.14285714285714285], [1.0, 0.42857142857142855], [1.0, 0.7142857142857143], [1.0, 1.0], [1.0, -1.0], [1.0, -0.7142857142857143], [1.0, -0.42857142857142855], [1.0, -0.14285714285714285], [1.0, 0.14285714285714285], [1.0, 0.42857142857142855], [1.0, 0.7142857142857143], [1.0, 1.0], [1.0, -1.0], [1.0, -0.7142857142857143], [1.0, -0.42857142857142855], [1.0, -0.14285714285714285], [1.0, 0.14285714285714285], [1.0, 0.42857142857142855], [1.0, 0.7142857142857143], [1.0, 1.0], [1.0, -1.0], [1.0, -0.7142857142857143], [1.0, -0.42857142857142855], [1.0, -0.14285714285714285], [1.0, 0.14285714285714285], [1.0, 0.42857142857142855], [1.0, 0.7142857142857143], [1.0, 1.0]]
Xpsi = [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
gidx = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4]
provenance = "Independent128/256-bit intended-L/Q Gamma reference; current175800Z dataset; generated values only"
source_sha256 = "fabf3e4c1a7d016ecff94a6926944553bb5238a744f03d178b7e3548e9d6c89c"
y = [0.5256033893547037, 1.8967439518330154, 1.175651841639344, 2.3700794356642976, 1.1816619195669293, 1.2763877759203521, 0.8708821723420906, 1.2311159551424984, 0.7710761675282595, 2.007434804749293, 1.650352456196985, 2.151003639751376, 4.8787513502586135, 0.844170829305904, 1.6260245096149282, 5.7175585591678155, 3.810402674902769, 1.3371894136777023, 1.8033210050489294, 2.122529689910351, 1.7294013256862668, 2.7080360162240633, 2.3466882525793844, 3.4204784585982675, 1.1776804810840462, 0.6304154605034238, 1.7759695972463598, 2.3745683458357854, 1.463598531077478, 0.887758773507171, 2.1653887205827993, 2.569100351899458]

[[cases]]
gradient = [-7.976487069778576, 1.2134104488433283e-5, 2.9543682733357564e-7, 5.406275605835117e-10, -2.344028407768623e-6, 8.66691988126389e-8]
idx = 1
nll = 42.615608821190385
receipt = "whitened-boundary-case1-20260831T183848Z.jls"
receipt_sha256 = "310bdc7c0b70b5190d380dea9d881eec55135df4ac92b8f80e3c58067fd916b1"
reference_z = [-0.2543039484884757, 3.33169427499429e-5, 1.1958303119359281, -0.00025258479247487834, 1.1982579567368377, 0.00022121640336785395, 0.16576104236428332, 0.00016622480924984804]
side = "lower"
theta = [0.3869274557252737, 0.2411075562242334, 1.5435209845450362, -1.1607137494915993, -0.09792091409713123, -9.024495328711623]
[[cases]]
gradient = [7.683384793805085, -7.77664559519335e-9, 5.7338337298014275e-9, 1.2329921144648551e-7, 7.637296592719425e-7, 1.0595928299739356e-7]
idx = 1
nll = 42.565597487170336
receipt = "whitened-boundary-case2-20260831T183848Z.jls"
receipt_sha256 = "cf8b5906fe8f86d26383728bc89a5c1a018e3909e1a6c889294980b4fe797355"
reference_z = [-1.4980949582649257, -2.0684677615091737e-5, -0.018932838688196233, -0.00020519498902616057, -0.08238734293071366, 0.0002634087486285999, -1.09194082216876, 0.00014688145931625925]
side = "upper"
theta = [0.8871413171473519, 0.23659455874423052, 1.376992396830441, -1.1614232125231732, -0.1521646170529175, -9.024495362089349]
[[cases]]
gradient = [-2.084594681781658e-6, -18.733105833109597, -1.901002172964179e-6, 7.148028109655591e-7, 6.5580611532979675e-6, 8.635802009854864e-8]
idx = 2
nll = 45.61354037882836
receipt = "whitened-boundary-case3-20260831T183728Z.jls"
receipt_sha256 = "05ac5b6d32f9565b79e8f1bdec7096dfc67055519521f0e8c272e672e3af4652"
reference_z = [-1.184224380020203, 6.0851960509902345e-5, 0.8871738217257413, -0.00024242788082191755, 0.32954428395923824, 0.00026244642110236007, -0.43262034874496424, 1.7989618444863436e-5]
side = "lower"
theta = [0.6759019201999692, -0.17122905407281544, 1.228882865622504, -1.591075512257643, -0.23172958113576636, -9.02449568331201]
[[cases]]
gradient = [6.834906552917458e-8, 13.214303306609864, 1.6965124722179798e-8, 1.1431160495215723e-7, 1.3246432846263874e-7, 1.672107498398887e-7]
idx = 2
nll = 43.0585740880204
receipt = "whitened-boundary-case4-20260831T183950Z.jls"
receipt_sha256 = "f8a1c7aecdb5952164c7b5f6829bc6ff4300929d1d83931fed99d91d023b0424"
reference_z = [-0.8686139665732262, -7.847392925890036e-5, 0.505905961490344, -9.719712689838313e-5, 0.7704799608037054, 8.881868868565504e-5, -0.5357607774371816, 0.0001987427122947436]
side = "upper"
theta = [0.65326963194467, 0.5036808351782416, 1.3480366418696152, -1.7683043472913598, -0.02900711286114456, -9.024495617300094]
"""

# Exercise the module's production include, never a test-only fallback.
@assert isdefined(DRModels, :_ls_whitened_eval)

function _white_moderate(kind; general = true, unobserved = false, c = 0.06)
    G = unobserved ? 4 : 3
    n = 18
    x = collect(range(-1, 1; length = n))
    gidx = repeat(1:3; inner = 6)
    Xmu = hcat(ones(n), x)
    Xpsi = hcat(ones(n), cos.(2 .* x))
    y = kind isa Val{:gamma} ? exp.(0.2 .+ 0.4 .* x .+ 0.25 .* sin.(1:n)) :
        kind isa Val{:nb2} ? Float64.(mod.(collect(1:n) .* 7, 9)) :
        kind isa Val{:beta} ? 0.15 .+ 0.7 .* (1 .+ sin.(collect(1:n))) ./ 2 :
        [(mod(3i, 9), 10) for i in 1:n]
    Q = general ? sparse([1.7 -0.2 0.1 0.0; -0.2 1.3 -0.15 0.0;
                          0.1 -0.15 1.9 -0.05; 0.0 0.0 -0.05 1.4][1:G, 1:G]) :
                  sparse(1.0I, G, G)
    Zeta = hcat(1 .+ 0.1 .* x, 0.2 .* sin.(x))
    Zpsi = hcat(0.15 .* cos.(x), 1 .- 0.1 .* x)
    theta = [0.15, 0.25, 0.4, 0.06, log(0.4), c, log(0.45)]
    return (; kind, y, Xmu, Xpsi, gidx, G, Q, Zeta, Zpsi, theta)
end

function _white_call(d, theta = d.theta; seed = nothing, gradient = true)
    DRModels._ls_whitened_eval(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G, d.Q,
                          theta, d.Zeta, d.Zpsi; seed, gradient)
end

function _white_information(d, theta = d.theta; seed = nothing, h = 1e-5)
    DRModels._ls_whitened_information(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G, d.Q,
                                 theta, d.Zeta, d.Zpsi; seed, h)
end

function _blockwise_a(L, z)
    a = similar(z)
    for g in 1:(length(z) ÷ 2)
        ix = 2g-1:2g
        a[ix] .= L * view(z, ix)
    end
    return a
end

_toml_matrix(rows) = permutedims(hcat((Float64.(row) for row in rows)...))

# Independent BigFloat Gamma certificate at the actual Float64 z returned by
# the helper, mapped by the intended BigFloat L.  It neither calls production
# Gamma gradients nor treats a rounded a64 as the solver state.
function _boundary_big_original_certificate(y, Xmu, Xpsi, Q, gidx, theta, z64)
    setprecision(BigFloat, 256) do
        G = size(Q, 1)
        t, z = BigFloat.(theta), BigFloat.(z64)
        L = BigFloat[exp(t[end-2]) 0; t[end-1] exp(t[end])]
        C = kron(BigFloat.(Matrix(Q)), Matrix{BigFloat}(I, 2, 2))
        gz = C * z
        eta0 = BigFloat.(Xmu) * t[1:2]
        psi0 = BigFloat.(Xpsi) * t[3:3]
        anorm2 = big"0"
        for g in 1:G
            ix = 2g-1:2g
            a = L * view(z, ix)
            anorm2 += dot(a, a)
        end
        for i in eachindex(gidx)
            g = gidx[i]; ix = 2g-1:2g
            a = L * view(z, ix)
            eta, psi = eta0[i] + a[1], psi0[i] + a[2]
            @test -big"30" < eta < big"30" && -big"30" < psi < big"30"
            s = exp(psi); rate = s * exp(-eta); ry = rate * BigFloat(y[i])
            score = BigFloat[s - ry, s * (digamma(s) - log(rate) - 1 - log(BigFloat(y[i]))) + ry]
            gz[ix] .+= transpose(L) * score
        end
        ga2 = big"0"
        for g in 1:G
            ga = transpose(L) \ view(gz, 2g-1:2g)
            ga2 += dot(ga, ga)
        end
        residual, bound = sqrt(ga2), big"1e-9" * (1 + sqrt(anorm2))
        return (; residual, bound, pass = residual <= bound)
    end
end

function _selected_blocks_match_dense(d, result)
    L, z = result.status.L, result.status.transformed_state
    pμ, pψ = size(d.Xmu, 2), size(d.Xpsi, 2)
    eta0 = d.Xmu * d.theta[1:pμ]
    psi0 = d.Xpsi * d.theta[pμ+1:pμ+pψ]
    Pz = kron(d.Q, sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2))
    H = Matrix(DRModels._ls_joint_hess(d.kind, d.y, eta0, psi0, d.gidx, d.G, z, Pz,
                                  d.Zeta * L, d.Zpsi * L))  # dense oracle only in this small test
    dense = inv(Symmetric(H))
    for g in 1:(size(H, 1) ÷ 2)
        i, j = 2g - 1, 2g
        got = result.status.selected_inverse_blocks[g]
        @test isapprox(got[1], dense[i, i]; rtol = 1e-10, atol = 1e-11)
        @test isapprox(got[2], dense[i, j]; rtol = 1e-10, atol = 1e-11)
        @test isapprox(got[3], dense[j, j]; rtol = 1e-10, atol = 1e-11)
    end
end

@testset "private whitened paired location-scale route" begin
    @testset "moderate four-family, general Q and noncanonical loadings" begin
        for kind in (Val(:gamma), Val(:nb2), Val(:beta), Val(:betabinomial))
            d = _white_moderate(kind)
            result = _white_call(d)
            @test result.status.ok
            @test result.status.undamped_hpd
            @test result.status.inside_clamp
            @test result.status.original_residual <= result.status.original_bound
            @test length(result.gradient) == length(d.theta)
            legacy_value = DRModels._ls_fit_nll(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G,
                                            d.Q, d.theta, d.Zeta, d.Zpsi)
            legacy_gradient = DRModels._ls_marginal_grad(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx,
                                                      d.G, d.Q, d.theta, d.Zeta, d.Zpsi)
            @test isapprox(result.value, legacy_value; rtol = 0, atol = 1e-8)
            @test maximum(abs, result.gradient .- legacy_gradient) <= 2e-6
            _selected_blocks_match_dense(d, result)
        end
    end

    @testset "unobserved group, c=0, and refusal" begin
        d = _white_moderate(Val(:gamma); unobserved = true, c = 0.0)
        result = _white_call(d)
        @test result.status.ok
        @test length(result.status.selected_inverse_blocks) == d.G
        _selected_blocks_match_dense(d, result)
        bad = copy(d.theta); bad[1] = Inf
        refused = _white_call(d, bad)
        @test !refused.status.ok
        @test all(isnan, refused.gradient)
        Qbad = copy(d.Q); Qbad[1, 2] += 0.1
        nonsymmetric = _white_call(merge(d, (; Q = Qbad)))
        @test !nonsymmetric.status.ok
        @test nonsymmetric.status.reason === :invalid_input
    end

    @testset "owned seeds are guesses, not certificates" begin
        d = _white_moderate(Val(:gamma))
        cold = _white_call(d)
        @test cold.status.ok
        before = copy(cold.seed.z)
        typed = _white_call(d; seed = cold.seed)
        @test typed.status.ok
        @test typed.status.seed_kind === :typed
        @test cold.seed.z == before
        @test typed.seed.z !== cold.seed.z
        @test typed.status.transformed_state !== typed.seed.z
        @test isapprox(typed.value, cold.value; rtol = 0, atol = 1e-10)

        legacy_a = _blockwise_a(cold.seed.L, cold.seed.z)
        legacy = _white_call(d; seed = legacy_a)
        @test legacy.status.ok
        @test legacy.status.seed_kind === :legacy_vector
        @test isapprox(legacy.status.initial_z, cold.seed.z; rtol = 0, atol = 2e-14)

        changed = copy(d.theta); changed[end] += 0.1
        transported = _white_call(d, changed; seed = cold.seed)
        Lnew = [exp(changed[end-2]) 0.0; changed[end-1] exp(changed[end])]
        expected = similar(cold.seed.z)
        T = Lnew \ cold.seed.L
        for g in 1:d.G
            ix = 2g-1:2g
            expected[ix] .= T * view(cold.seed.z, ix)
        end
        @test isapprox(transported.status.initial_z, expected; rtol = 0, atol = 2e-14)
        # A finite typed seed can still overflow only when transported to Lnew.
        extreme = copy(d.theta); extreme[end] = -700.0
        overflowing = DRModels._LSWhitenedSeed(fill(1e10, 2d.G), cold.seed.L, cold.seed.theta)
        rejected_transport = _white_call(d, extreme; seed = overflowing)
        @test !rejected_transport.status.ok
        @test rejected_transport.status.reason === :invalid_seed
        # A call at another theta cannot mutate the first state or certify it.
        other = copy(d.theta); other[1] += 0.03
        _white_call(d, other; seed = cold.seed)
        @test cold.seed.z == before
    end

    @testset "paired observed information owns every finite-difference seed" begin
        # These two moderate families exercise the paired central-difference
        # route against the established raw information without invoking any
        # profile or outer fit.  The comparison tolerance accommodates the
        # already-tested value/gradient arithmetic route at h = 1e-5.
        for kind in (Val(:gamma), Val(:nb2))
            d = _white_moderate(kind)
            cold = _white_call(d)
            @test cold.status.ok
            z_before, L_before, theta_before = copy(cold.seed.z), copy(cold.seed.L), copy(cold.seed.theta)
            Hpaired = _white_information(d; seed = cold.seed)
            Hraw = DRModels._ls_obs_information(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G,
                                            d.Q, d.theta, d.Zeta, d.Zpsi)
            @test all(isfinite, Matrix(Hpaired))
            @test isapprox(Matrix(Hpaired), Matrix(Hraw); rtol = 2e-4, atol = 2e-5)
            @test cold.seed.z == z_before
            @test cold.seed.L == L_before
            @test cold.seed.theta == theta_before
            @test DRModels._ls_whitened_vcov(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G,
                                         d.Q, d.theta, d.Zeta, d.Zpsi; seed = cold.seed) !== nothing
        end

        d = _white_moderate(Val(:gamma))
        bad = copy(d.theta); bad[1] = Inf
        Hbad = _white_information(d, bad)
        @test all(isnan, Matrix(Hbad))
        @test DRModels._ls_whitened_vcov(d.kind, d.y, d.Xmu, d.Xpsi, d.gidx, d.G,
                                     d.Q, bad, d.Zeta, d.Zpsi) === nothing
    end

    @testset "all four retained Gamma boundary references" begin
        fixture = TOML.parse(_WHITENED_BOUNDARY_TOML)
        Xmu = _toml_matrix(fixture["Xmu"])
        Xpsi = _toml_matrix(fixture["Xpsi"])
        y = Float64.(fixture["y"])
        Q = sparse(_toml_matrix(fixture["Q"]))
        gidx, G = Int.(fixture["gidx"]), fixture["G"]
        Zeta = DRModels._ls_canonical_Zeta(length(y))
        Zpsi = DRModels._ls_canonical_Zpsi(length(y))
        for case in fixture["cases"]
            theta = Float64.(case["theta"])
            result = DRModels._ls_whitened_eval(Val(:gamma), y, Xmu, Xpsi, gidx, G, Q,
                                            theta, Zeta, Zpsi)
            @testset "$(case["idx"])-$(case["side"])" begin
                raw_objective = DRModels.LocScaleObjective(Val(:gamma), y, Xmu, Xpsi, gidx, G, Q)
                paired_objective = DRModels.LocScaleObjective(Val(:gamma), y, Xmu, Xpsi, gidx, G, Q;
                                                        whitened=true)
                @test !raw_objective.whitened
                @test paired_objective.whitened
                @test isapprox(paired_objective(theta), case["nll"]; rtol=0, atol=1e-8)
                @test maximum(abs, DRModels._ls_objective_gradient(paired_objective, theta) .-
                              Float64.(case["gradient"])) <= 1e-7
                @test result.status.ok
                @test result.status.original_residual <= result.status.original_bound
                @test result.status.undamped_hpd && result.status.inside_clamp
                @test isapprox(result.value, case["nll"]; rtol = 0, atol = 1e-8)
                @test maximum(abs, result.gradient .- Float64.(case["gradient"])) <= 1e-7
                @test maximum(abs, -result.gradient .- Float64.(case["gradient"])) > 1e-7
                independent = _boundary_big_original_certificate(y, Xmu, Xpsi, Q, gidx,
                                                                    theta, result.status.transformed_state)
                @test independent.pass
            end
        end
    end
end

println("LOCSCALE_WHITENED_FOCUSED_OK")
