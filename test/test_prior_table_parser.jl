using Test, Tables

struct PriorTableDuplicateColumns end
Tables.istable(::Type{PriorTableDuplicateColumns}) = true
Tables.columnaccess(::Type{PriorTableDuplicateColumns}) = true
Tables.columns(x::PriorTableDuplicateColumns) = x
Tables.columnnames(::PriorTableDuplicateColumns) = (:prior, :class, :coef, :coef)

@testset "literal prior table parsing" begin
    parser = joinpath(@__DIR__, "..", "src", "prior_table.jl")
    @test isfile(parser)
    if isfile(parser)
        include(parser)
        normal = (prior=["normal( 0, 2 )"], class=["fixef"], coef=["x"])
        rows = _parse_prior_table(normal)
        @test length(rows) == 1
        @test rows[1].kind == :normal
        @test rows[1].arguments == (0.0, 2.0)
        @test rows[1].dpar == "mu"
        gamma = (prior=["gamma(0.6666666666666666,2)"], class=["ranef_sd"], coef=["site"])
        @test only(_parse_prior_table(gamma)).arguments == (0.6666666666666666, 2.0)
        for text in ("normal(0,2);system('x')", "normal(1+1,2)", "normal(0,Inf)", "normal(0,1e-999)", "normal(1e-999,2)", "normal(0,1e200)", "ridge(-1)", "gamma(1,1)")
            @test_throws ArgumentError _parse_prior_table((prior=[text], class=[startswith(text,"gamma") ? "ranef_sd" : "fixef"], coef=["x"]))
        end
        @test only(_parse_prior_table((prior=["ridge(0)"], class=["fixef"], coef=["x"]))).arguments == (0.0,)
        @test_throws ArgumentError _parse_prior_table((prior=["gamma(1e-320,2)"], class=["ranef_sd"], coef=["site"]))
        @test_throws ArgumentError _parse_prior_table((prior=["normal(0,2)"], class=["fixef_disp"], coef=["x"]))
        @test_throws ArgumentError _parse_prior_table((prior=["normal(0,2)"], class=["fixef"], coef=["all"]))
        @test_throws ArgumentError _parse_prior_table((prior=["normal(0,2)"], class=["fixef"], coef=["x"], group=["site"]))
        broad = _parse_prior_table((prior=["gamma(1,2)"], class=["ranef_sd"], coef=[""]))
        @test getproperty.(_expand_prior_table(broad, ["site", "observer"]), :group) == ["site", "observer"]
        specific = _parse_prior_table((prior=["gamma(1,2)"], class=["ranef_sd"], coef=["(Intercept)"], group=["site"]))
        @test only(_expand_prior_table(specific, ["site", "observer"])).group == "site"
        overlapping = _parse_prior_table((prior=["gamma(1,2)","gamma(1,2)"], class=["ranef_sd","ranef_sd"], coef=["","site"]))
        @test_throws ArgumentError _expand_prior_table(overlapping, ["site", "observer"])
        @test_throws ArgumentError _expand_prior_table(broad, String[])
        @test_throws ArgumentError _expand_prior_table(specific, ["observer"])
        @test_throws ArgumentError _parse_prior_table(PriorTableDuplicateColumns())
        @test_throws ArgumentError _parse_prior_table((prior=["ridge(1)","ridge(1)"], class=["fixef"], coef=["x","y"]))
        @test_throws ArgumentError _parse_prior_table((prior=["ridge(1)"], class=["fixef"]))
        @test_throws ArgumentError _parse_prior_table((prior=["ridge(1)"], class=["fixef"], coef=[missing]))
        @test_throws ArgumentError _parse_prior_table((prior=["ridge(1)"], class=["fixef"], coef=["x"], extra=["x"]))
        @test_throws ArgumentError _parse_prior_table((prior=String[], class=String[], coef=String[]))
    end
end
