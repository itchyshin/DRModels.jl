# cov lane 1125n7 — temporal AR/HOMTOEP kernels and parsing (src/temporal.jl).
using DRModels
using Test, StatsModels, LinearAlgebra, Random

_msg(f) = try
    f()
    ""
catch e
    e isa ArgumentError ? e.msg : sprint(showerror, e)
end

function _temporal_rhs(sym::AbstractString)
    t = Core.eval(@__MODULE__, Meta.parse("@formula(y ~ $sym)"))
    return t.rhs
end

@testset "cov 1125n7: _homtoep_levinson and correlations" begin
    κ = [0.2, -0.15, 0.05]
    Φ, logv = DRModels._homtoep_levinson(κ)
    @test size(Φ) == (3, 3)
    @test length(logv) == 4
    @test all(isfinite, Φ)
    @test all(isfinite, logv)
    for m in 1:3
        @test Φ[m, m] ≈ tanh(κ[m])
    end
    ρ = DRModels._homtoep_correlations(κ)
    @test length(ρ) == 3
    @test all(abs.(ρ) .< 1)
    ρ0 = DRModels._homtoep_correlations(zeros(0))
    @test isempty(ρ0)
end

@testset "cov 1125n7: _homtoep_series NLL" begin
    κ = [0.3, 0.1]
    Φ, logv = DRModels._homtoep_levinson(κ)
    r = [1.2, -0.4, 0.7]
    ll1 = DRModels._homtoep_series(Φ, logv, log(0.5), r)
    r_alt = [0.1, 0.5, -0.2]
    ll2 = DRModels._homtoep_series(Φ, logv, log(0.5), r_alt)
    @test isfinite(ll1) && isfinite(ll2)
    @test ll1 != ll2
    r1 = [0.0, 0.0, 0.0]
    ll0 = DRModels._homtoep_series(Φ, logv, 0.0, r1)
    @test isfinite(ll0)
end

@testset "cov 1125n7: _parse_temporal_term" begin
    t = _temporal_rhs("temporal(1 | subject, time, ar1)")
    tt = DRModels._parse_temporal_term(t)
    @test tt.group == :subject
    @test tt.time == :time
    @test tt.structure == :ar1
    lbl = DRModels._temporal_label(tt)
    @test occursin("subject", lbl)
    @test occursin("ar1", lbl)
    for bad in (
        "temporal(x, time, ar1)",
        "temporal(1 | subject)",
        "temporal(1 | subject, time, bad)",
        "temporal(0 + x | subject, time, ou)",
    )
        term = try
            _temporal_rhs(bad)
        catch
            nothing
        end
        term === nothing && continue
        @test_throws ArgumentError DRModels._parse_temporal_term(term)
    end
end

@testset "cov 1125n7: _collect_temporal" begin
    rhs = _temporal_rhs("1 + x + temporal(1 | id, t, homtoep)")
    cols = DRModels._collect_temporal(rhs)
    @test length(cols) == 1
    @test only(cols).structure == :homtoep
    plain = _temporal_rhs("1 + x")
    @test isempty(DRModels._collect_temporal(plain))
end

@testset "cov 1125n7: _temporal_layout homtoep validation" begin
    tt = (; group = :id, time = :occ, structure = :homtoep)
    data = (; id = repeat(1:3, inner = 3), occ = repeat([1, 2, 3], 3), y = randn(9))
    lay = DRModels._temporal_layout(tt, data; has_ordinary = false)
    @test lay.nseries == 3
    @test all(length.(lay.rows) .== 3)
    bad_frac = (; id = [1, 1, 1], occ = [1, 1.5, 2], y = zeros(3))
    msg = _msg(() -> DRModels._temporal_layout(tt, bad_frac; has_ordinary = false))
    @test occursin("integer", msg) || occursin("HOMTOEP", msg)
    dup = (; id = [1, 1, 1], occ = [1, 1, 2], y = zeros(3))
    msg2 = _msg(() -> DRModels._temporal_layout(tt, dup; has_ordinary = false))
    @test !isempty(msg2)
end

@testset "cov 1125n7: _temporal_smooth" begin
    hist = (
        mf = [0.1, 0.2, 0.3],
        mp = [0.0, 0.15, 0.25],
        Pf = [1.0, 0.9, 0.8],
        Pp = [1.1, 1.0, 0.95],
        a = [0.0, 0.5, 0.5],
    )
    sm = DRModels._temporal_smooth(hist)
    @test length(sm) == 3
    @test sm[end] ≈ hist.mf[end]
    @test all(isfinite, sm)
end