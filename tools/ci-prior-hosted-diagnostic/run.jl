using SHA, InteractiveUtils, LinearAlgebra, ForwardDiff, DRModels
root, label, mode = ARGS
route = joinpath(root, label)
@assert realpath(dirname(pathof(DRModels))) == realpath(joinpath(route, "src"))
println("LOADED_PACKAGE_PATH=", pathof(DRModels))
versioninfo(; verbose = true)
println("CPU_NAME=", Sys.CPU_NAME)
println("BLAS_CONFIG=", BLAS.get_config(), " threads=", BLAS.get_num_threads())
println("JULIA_THREADS=", Threads.nthreads())
victim = joinpath(route, "test", "test_binomial_sd_public.jl")
original = read(victim, String)
needle = "    @test slope_fit.converged"
@assert length(findall(needle, original)) == 1
instrumented = replace(original, needle => read(joinpath(@__DIR__, "fit_diagnostics.jl"), String) * "\n" * needle)
@assert count("@test", original) == count("@test", instrumented)
println("ORIGINAL_VICTIM_SHA256=", bytes2hex(sha256(original)))
write(victim, instrumented)
try
    if mode == "standalone"
        include(victim)
        println("DIAGNOSTIC_STANDALONE_COMPLETE")
    elseif mode == "prefix"
        runner = joinpath(route, "test", "runtests.jl")
        s = read(runner, String)
        println("ORIGINAL_RUNNER_SHA256=", bytes2hex(sha256(s)))
        marker = "for f in _discovered\n    _shard_include(f)\nend"
        @assert length(findall(marker, s)) == 1
        replacement = "for f in _discovered\n    _shard_include(f)\n    if f == \"test_binomial_sd_public.jl\"\n        println(\"DIAGNOSTIC_PREFIX_COMPLETE\")\n        break\n    end\nend"
        # Stop before the tail after the discovery loop as well.
        prefix = first(split(s, marker)) * replacement
        diagnostic = joinpath(route, "test", ".diagnostic_prefix_runner.jl")
        write(diagnostic, prefix)
        include(diagnostic)
    else
        error("Unknown diagnostic mode")
    end
finally
    write(victim, original)
    @assert bytes2hex(sha256(read(victim))) == bytes2hex(sha256(original))
end
