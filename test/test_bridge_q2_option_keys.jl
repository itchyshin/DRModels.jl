using DRModels
using Test, Random, LinearAlgebra

# #1011 items 1 and 2: the q2 bridge entry points fail closed on an unknown
# option key (as `_bridge_fit` does), and the profile-failure error only
# recommends a method `drm_bridge_inference` accepts.

function _q2_thrown(f)
    try
        f()
        return nothing
    catch e
        return e
    end
end

@testset "q2 bridge entry points reject unknown option keys (#1011)" begin
    Random.seed!(1011)
    tree = "((a:1,b:1):1,(c:1,d:1):1);"
    sp = repeat(1:4, inner = 10)
    Y = randn(40, 2)
    X = reshape(randn(40), :, 1)
    Q = Matrix{Float64}(I, 4, 4)

    e1 = _q2_thrown(() -> DRModels.drm_bridge_q2_phylo(; Y, X, species = sp, tree,
        options = Dict("iteratons" => 5, "bogus_key" => 1)))
    @test e1 isa ArgumentError
    msg1 = sprint(showerror, e1)
    @test occursin("iteratons", msg1)
    @test occursin("bogus_key", msg1)
    @test occursin("known keys are", msg1)
    @test occursin("iterations", msg1)

    e2 = _q2_thrown(() -> DRModels.drm_bridge_q2_known_precision(; Y, X, group = sp, Q,
        options = Dict("g_tolerance" => 1e-3)))
    @test e2 isa ArgumentError
    msg2 = sprint(showerror, e2)
    @test occursin("g_tolerance", msg2)
    @test occursin("known keys are", msg2)

    # Valid keys, string or symbol keyed, are still accepted.
    ok = DRModels.drm_bridge_q2_phylo(; Y, X, species = sp, tree,
        options = Dict("iterations" => 2, "g_tol" => 1e-3, "fd_h" => 1e-6))
    @test haskey(ok, "converged")
    ok2 = DRModels.drm_bridge_q2_known_precision(; Y, X, group = sp, Q,
        options = (iterations = 2, g_tol = 1e-3))
    @test haskey(ok2, "converged")
end

@testset "profile-failure text names only accepted methods (#1011)" begin
    msg = DRModels._bridge_profile_failure_message("fixef", "x", "no root")
    @test occursin("did not converge", msg)
    @test occursin("`fixef:x`", msg)
    @test !occursin("wald", msg)
    @test occursin("method = \"bootstrap\"", msg)
end
