using Test

@testset "canonical import is quiet and keeps the DRM alias" begin
    project = dirname(Base.active_project())
    script = "using DRModels; @assert DRModels.DRM === DRModels"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(project) -e $(script)`

    mktemp() do stderr_path, stderr_io
        close(stderr_io)
        proc = run(pipeline(cmd; stdout = devnull, stderr = stderr_path); wait = false)
        wait(proc)
        stderr = read(stderr_path, String)

        @test success(proc)
        @test !occursin("`DRM` is deprecated", stderr)
    end
end
