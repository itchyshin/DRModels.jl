# Census of every Hessian that reaches the covariance guard (`src/vcov_guard.jl`).
#
# Used to calibrate the guard's thresholds on the measured distribution rather
# than by feel (the 2026-08-26 `_VCOV_RTOL` calibration of #513, and the
# unit-free rescaling of #936). It attaches a recorder to `_VCOV_CENSUS_HOOK`,
# runs the test suite (or one shard of it), and appends one tab-separated row per
# guard call. The guard's decision never reads the hook, so the run is the
# ordinary suite.
#
#   DRM_VCOV_CENSUS_OUT=census.tsv DRM_TEST_SHARD=1/4 \
#       julia --check-bounds=yes --project=test tools/vcov_guard_census.jl
#
# Each row stores the symmetrised Hessian itself, so a candidate rule can be
# replayed offline on every call without re-running the suite
# (`tools/vcov_guard_replay.jl`). Columns:
#   file, line   innermost test_*.jl frame on the call stack
#   context      the guard's `context` string
#   p            dimension
#   unitfree     the caller's `unitfree` indices, comma-separated
#   H            the p*p entries, column-major, comma-separated (full precision)

using LinearAlgebra, DRModels

const _CENSUS_OUT = get(ENV, "DRM_VCOV_CENSUS_OUT", "vcov_guard_census.tsv")
const _CENSUS_IO = open(_CENSUS_OUT, "a")

function _census_row(Hs::AbstractMatrix, context::AbstractString, unitfree)
    file, line = "", 0
    for fr in stacktrace()
        f = string(fr.file)
        if occursin(r"test_[^/]*\.jl$", f)
            file, line = basename(f), fr.line
            break
        end
    end
    ctx = replace(String(context), '\t' => ' ', '\n' => ' ')
    println(_CENSUS_IO, join((file, line, ctx, size(Hs, 1), join(collect(Int, unitfree), ','),
                              join((repr(Float64(x)) for x in Hs), ',')), '\t'))
    flush(_CENSUS_IO)
    return nothing
end

DRModels._VCOV_CENSUS_HOOK[] = _census_row
try
    include(joinpath(@__DIR__, "..", "test", "runtests.jl"))
finally
    DRModels._VCOV_CENSUS_HOOK[] = nothing
    close(_CENSUS_IO)
end
