# #1397 J2: Binomial, Beta, Gamma and NegBinomial2 have no REML route. Asking for
# `method = :REML` must name that refusal. The old text said `unknown `method``,
# which reads as if `:REML` were not a method symbol at all.
using DRModels
using Test

function _reml_err(fam, f, data)
    try
        drm(f, fam; data, method = :REML)
        nothing
    catch e
        e
    end
end

@testset "REML on ML-only families is refused by name (#1397)" begin
    n = 8
    x = [0.2, -0.4, 0.1, 0.5, -0.2, 0.3, -0.6, 0.0]
    ybin = [0, 1, 0, 1, 1, 0, 0, 1]
    ycount = [0, 1, 2, 0, 3, 1, 0, 2]
    ypos = [0.4, 1.2, 0.7, 0.5, 1.1, 0.8, 0.6, 0.9]
    ybeta = [0.2, 0.7, 0.4, 0.3, 0.8, 0.55, 0.25, 0.6]
    f1 = bf(@formula(y ~ x))
    fσ = bf(@formula(y ~ x), @formula(sigma ~ 1))
    cases = (
        ("Binomial", Binomial(), f1, (; y = ybin, x)),
        ("Beta", Beta(), fσ, (; y = ybeta, x)),
        ("Gamma", Gamma(), fσ, (; y = ypos, x)),
        ("NegBinomial2", NegBinomial2(), fσ, (; y = ycount, x)),
    )
    for (name, fam, f, data) in cases
        err = _reml_err(fam, f, data)
        msg = err === nothing ? "" : sprint(showerror, err)
        @test err isa ArgumentError
        @test occursin("`method = :REML` is not available for $name", msg)
        @test !occursin("unknown", msg)
    end

    # An unrecognised symbol is still an unknown method, not a REML refusal.
    err_unk = try
        drm(f1, Binomial(); data = (; y = ybin, x), method = :restricted)
        nothing
    catch e
        e
    end
    msg_unk = sprint(showerror, err_unk)
    @test err_unk isa ArgumentError
    @test occursin("unknown `method = :restricted`", msg_unk)

    # The same Binomial data still fits under ML. The refusal is the REML request.
    fit = drm(f1, Binomial(); data = (; y = ybin, x))
    @test estimation_method(fit) === :ML
    @test isfinite(loglik(fit))
end
