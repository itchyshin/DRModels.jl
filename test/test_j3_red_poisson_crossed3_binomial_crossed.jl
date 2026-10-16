# RED (throwaway, #1073 J3b): on the base commit the reported SEs of two crossed routes miss native
# drmTMB sdreport by more than 1% (K=3 Poisson crossed: 30%; binomial crossed: 1.4%).
using DRModels, Test, TOML, LinearAlgebra
J3 = joinpath(@__DIR__, "fixtures", "j3_se")
function rd(route)
    lines = readlines(joinpath(J3, route, "data.csv"))
    hdr = [strip(c, '"') for c in split(lines[1], ',')]
    Dict(Symbol(h) => String[strip(split(l, ',')[j], '"') for l in lines[2:end]] for (j, h) in enumerate(hdr))
end
num(v) = [parse(Float64, s) for s in v]
@testset "RED: crossed-route SEs vs drmTMB sdreport (base commit)" begin
    c = rd("poisson_crossed3"); ref = TOML.parsefile(joinpath(J3, "poisson_crossed3", "ref.toml"))
    d = (; x = num(c[:x]), y = num(c[:y]), a = c[:a], b = c[:b], c = c[:c])
    fit = drm(bf(@formula(y ~ x + (1 | a) + (1 | b) + (1 | c))), Poisson(); data = d)
    @test all(abs.(sqrt.(diag(vcov(fit))) ./ Float64.(ref["se"]) .- 1) .< 0.01)
    c = rd("binomial_crossed"); ref = TOML.parsefile(joinpath(J3, "binomial_crossed", "ref.toml"))
    d = (; x = num(c[:x]), succ = num(c[:succ]), fail = num(c[:fail]), a = c[:a], b = c[:b])
    fit = drm(bf(@formula(cbind(succ, fail) ~ x + (1 | a) + (1 | b))), Binomial(); data = d)
    @test all(abs.(sqrt.(diag(vcov(fit))) ./ Float64.(ref["se"]) .- 1) .< 0.01)
end
