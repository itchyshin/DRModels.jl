using DRModels, Test, Random

# Decision f: a single Gamma `(0 + z | g)` or `(1 + z | g)` has no fresh random-effect draw either.
# Before this change `bootstrap_summary`, `bootstrap_result` and the default `simulate` fell back to a draw
# with the random effects at zero (the variance-component interval collapses) and returned a table. They
# now refuse by name, as the shapes with two or more terms already do.
function _gsl_data(seed)
    r = Random.Xoshiro(seed); n = 300; G = 20
    gi = rand(r, 1:G, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ b0[gi] .+ b1[gi] .* z
    y = [exp(e) * (0.6 + 0.8 * rand(r)) for e in eta]
    return (; x, z, g = string.("g", gi), y)
end

@testset "Gamma single (0 + z | g) and (1 + z | g): bootstrap and default simulate refuse by name" begin
    d = _gsl_data(5)
    cases = [
        ("(0 + z | g)", "a random slope without an intercept"),
        ("(1 + z | g)", "a correlated random intercept and slope"),
    ]
    for (terms, label) in cases
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula(y ~ x + $terms))"))
        fit = drm(fo, DRModels.Gamma(); data = d)
        @test fit.converged
        calls = (
            "bootstrap_summary" => () -> bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "bootstrap_result" => () -> bootstrap_result(fit; data = d, B = 2, rng = Random.Xoshiro(1)),
            "default simulate" => () -> simulate(fit; data = d, rng = Random.Xoshiro(1)),
        )
        for (cname, call) in calls
            err = try; call(); nothing; catch e; e; end
            @test err isa ArgumentError
            err isa ArgumentError || continue
            msg = sprint(showerror, err)
            @test occursin("no fresh random-effect draw", msg)
            @test occursin("for Gamma() with $label ($terms)", msg)
            @test occursin("call `simulate(fit; data = data, re = :zero)`", msg)
        end
        # the escape named in the message works: a conditional draw with the random effects at zero
        @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.y)
    end
end

@testset "Gamma slope refusal leaves the working shapes alone" begin
    d = _gsl_data(5)
    # a single (1 | g) keeps its fresh draw and its bootstrap
    f1 = drm(bf(@formula(y ~ x + (1 | g))), DRModels.Gamma(); data = d)
    @test length(simulate(f1; data = d, rng = Random.Xoshiro(1))) == length(d.y)
    @test size(bootstrap_summary(f1; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end
