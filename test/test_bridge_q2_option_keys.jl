using DRModels
using Test, Random, LinearAlgebra

# #1011: the q2 bridge entry points must fail closed on an unknown option key
# (as `_bridge_fit` does), and the profile-failure error must only recommend a
# method `drm_bridge_inference` actually accepts.

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

    # Misspelled key on drm_bridge_q2_phylo: refused, key named, known keys listed.
    e1 = _q2_thrown(() -> DRModels.drm_bridge_q2_phylo(; Y, X, species = sp, tree,
        options = Dict("iteratons" => 5, "bogus_key" => 1)))
    @test e1 isa ArgumentError
    msg1 = sprint(showerror, e1)
    @test occursin("iteratons", msg1)
    @test occursin("bogus_key", msg1)
    @test occursin("known keys are", msg1)
    @test occursin("iterations", msg1)

    # Same on drm_bridge_q2_known_precision.
    e2 = _q2_thrown(() -> DRModels.drm_bridge_q2_known_precision(; Y, X, group = sp, Q,
        options = Dict("g_tolerance" => 1e-3)))
    @test e2 isa ArgumentError
    msg2 = sprint(showerror, e2)
    @test occursin("g_tolerance", msg2)
    @test occursin("known keys are", msg2)

    # The valid keys are still accepted (no unknown-key refusal).
    ok = DRModels.drm_bridge_q2_phylo(; Y, X, species = sp, tree,
        options = Dict("iterations" => 2, "g_tol" => 1e-3, "fd_h" => 1e-6))
    @test haskey(ok, "converged")
end

@testset "profile-failure error does not recommend an unsupported method (#1011)" begin
    # A genuine profile failure is hard to trigger on demand, so check the
    # message text in the source: the advice that follows the profile-failure
    # refusal must not name `method = "wald"`, which drm_bridge_inference
    # itself refuses.
    src = read(joinpath(pkgdir(DRModels), "src", "bridge.jl"), String)
    i = findfirst("profile endpoint search did not converge for", src)
    @test i !== nothing
    window = src[first(i):min(lastindex(src), first(i) + 700)]
    @test !occursin("wald", window)
    @test occursin("bootstrap", window)
end
