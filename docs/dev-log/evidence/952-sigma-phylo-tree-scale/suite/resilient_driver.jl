# Include each listed test file as runtests.jl would, but keep going after a failing file.
using DRModels
using Test, LinearAlgebra, SparseArrays, Random
BLAS.set_num_threads(1)
listfile, logfile = ARGS[1], ARGS[2]
for f in readlines(listfile)
    isempty(strip(f)) && continue
    t0 = time()
    status = try
        @testset "$f" begin
            include(joinpath("/home/user/DRModels.jl/test", f))
        end
        "PASS"
    catch err
        err isa InterruptException && rethrow(err)
        "FAIL ($(typeof(err)))"
    end
    open(logfile, "a") do io
        println(io, f, " ", status, " secs=", round(Int, time() - t0)); flush(io)
    end
end
open(io -> println(io, "ALLDONE"), logfile, "a")
