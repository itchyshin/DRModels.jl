using DRModels, Test, Random

# P11c: Gamma and Binomial `(1 | g) + (0 + x | g)` have no fresh random-effect draw. Before this
# change `bootstrap_summary` fell back to a draw with the random effects at zero and returned a
# table (a variance-component interval that collapses); it now refuses by name.
@testset "no fresh draw: Gamma and Binomial (1 | g) + (0 + x | g) refuse bootstrap and default simulate" begin
    r = Random.Xoshiro(3); n = 240; G = 20
    g = string.("g", rand(r, 1:G, n)); x = randn(r, n)
    gi = parse.(Int, SubString.(g, 2)); bg = 0.5 .* randn(r, G); bs = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ bg[gi] .+ bs[gi] .* x
    yg = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    p = 1 ./ (1 .+ exp.(-eta)); s = Float64[sum(rand(r) < pi for _ in 1:10) for pi in p]
    dg = (; y = yg, x, g); db = (; s, f = 10.0 .- s, x, g)
    fg = drm(bf(@formula(y ~ x + (1 | g) + (0 + x | g))), DRModels.Gamma(); data = dg)
    fb = drm(bf(@formula(cbind(s, f) ~ x + (1 | g) + (0 + x | g))), Binomial(); data = db)
    for (fit, d) in ((fg, dg), (fb, db))
        err = try; bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err isa ArgumentError
        @test occursin("no fresh random-effect draw", sprint(showerror, err))
        err2 = try; simulate(fit; data = d, rng = Random.Xoshiro(1)); nothing; catch e; e; end
        @test err2 isa ArgumentError && occursin("no fresh random-effect draw", sprint(showerror, err2))
    end
    # the escape hatch named in the message works for Gamma (Binomial needs stored trials, not on main)
    @test length(simulate(fg; data = dg, re = :zero, rng = Random.Xoshiro(1))) == n
    # a single (1 | g) keeps its simulator and bootstrap
    f1 = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = dg)
    @test size(bootstrap_summary(f1; data = dg, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end
