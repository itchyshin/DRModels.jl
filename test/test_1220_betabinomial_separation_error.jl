# #1220: BetaBinomial() on perfectly separated cbind data used to throw LAPACK's bare
# "matrix contains Infs or NaNs" from `eigvals` inside `_vcov_from_hessian`. It must
# now throw an ArgumentError that says what went wrong, why, and what to do. Fit
# results (and `se = false` behaviour) are unchanged.
using DRModels
using Test

@testset "#1220 non-finite Hessian gives an informative error" begin
    n = 40
    x  = collect(range(-2, 2, length = n))
    s  = Float64.(x .> 0) .* 10          # perfectly separated successes out of 10
    fl = 10 .- s
    f = bf(@formula(cbind(s, fl) ~ x), @formula(sigma ~ 1))

    err = try
        drm(f, BetaBinomial(); data = (; s, fl, x))
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    msg = sprint(showerror, err)
    @test !occursin("matrix contains Infs or NaNs", msg)
    @test occursin("Hessian", msg)
    @test occursin("Inf or NaN", msg)
    @test occursin("separation", msg)
    @test occursin("se = false", msg)

    # Unchanged: se = false still returns a fit (no Hessian step).
    fz = drm(f, BetaBinomial(); data = (; s, fl, x), se = false)
    @test length(coef(fz)) == 3

    # Guard-level: same exception type, informative message, context included.
    H = [1.0 NaN; NaN 1.0]
    e2 = try
        DRModels._vcov_from_hessian(H; context = "unit")
        nothing
    catch e
        e
    end
    @test e2 isa ArgumentError
    @test occursin("(unit)", sprint(showerror, e2))
    @test occursin("[1, 2]", sprint(showerror, e2))
end
