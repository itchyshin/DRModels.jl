using Pkg, TOML, SHA
root = abspath(ARGS[1])
frozen = TOML.parsefile(joinpath(@__DIR__, "Manifest.frozen.toml"))
for label in ("baseline", "candidate")
    env = joinpath(root, label * "-env")
    mkpath(env)
    project = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
    project["sources"]["DRModels"]["path"] = joinpath(root, label)
    open(joinpath(env, "Project.toml"), "w") do io
        TOML.print(io, project)
    end
    manifest = deepcopy(frozen)
    manifest["deps"]["DRModels"][1]["path"] = joinpath(root, label)
    open(joinpath(env, "Manifest.toml"), "w") do io
        TOML.print(io, manifest)
    end
    Pkg.activate(env)
    Pkg.instantiate()
    actual = TOML.parsefile(joinpath(env, "Manifest.toml"))
    @assert actual["deps"] == manifest["deps"] "Dependency manifest changed"
    Pkg.status(; mode = Pkg.PKGMODE_MANIFEST)
    println("ENVIRONMENT_LOCKED ", label, " ", bytes2hex(sha256(read(joinpath(env, "Manifest.toml")))))
    flush(stdout)
    println(stderr, "DIAGNOSTIC_WARMUP_START ", label)
    flush(stderr)
    package_path = joinpath(root, label)
    warmup_code = "using DRModels, JET, Aqua, StableRNGs; println(\"WARMUP_LOADED_PATH=\",pathof(DRModels)); flush(stdout)"
    warmup_seconds = @elapsed run(`$(Base.julia_cmd()) --startup-file=no --threads=1 --check-bounds=yes --compiled-modules=yes --depwarn=yes --warn-overwrite=yes --inline=yes --track-allocation=none --code-coverage=@$package_path --project=$env -e $warmup_code`)
    println(stderr, "DIAGNOSTIC_WARMUP_COMPLETE ", label, " seconds=", warmup_seconds)
    flush(stderr)
end
