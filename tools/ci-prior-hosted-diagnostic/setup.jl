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
end
