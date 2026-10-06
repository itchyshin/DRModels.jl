using DRModels, Test, LinearAlgebra
BLAS.set_num_threads(1)
println("RUNTIME source=", pathof(DRModels), " Julia=", VERSION,
        " threads=", Threads.nthreads(), " BLAS=", BLAS.get_num_threads())
@test Threads.nthreads() == 1
@test BLAS.get_num_threads() == 1
@testset "coefficient labels and bridge neighbours" begin
    for name in ("test_bridge_materialization_collision.jl",
                 "test_bridge_formula_labels.jl", "test_bridge_lss_labels.jl", "test_bridge.jl",
                 "test_bridge_formula_translation.jl", "test_bridge_profile_target.jl")
        include(joinpath(@__DIR__, "..", "test", name))
    end
    # Run the actual new reader example, not a nearby hand-written surrogate.
    page = read(joinpath(@__DIR__, "..", "docs", "src", "r-julia-bridge.md"), String)
    example = match(r"```@example bridge_coefficient_labels\n(.*?)\n```"s, page)
    @test example !== nothing
    include_string(Main, example.captures[1], "r-julia-bridge.md:bridge_coefficient_labels")
end
println("COEFFICIENT_LABEL_COMBINED_PASS")
