using DRModels, Test, Random

# A single Binomial `(0 + z | g)` or `(1 + z | g)` has no fresh random-effect draw. Before this change
# `bootstrap_summary` and `bootstrap_result` fell back to a draw with the random effects at zero (the
# variance-component interval collapses) and returned a table. They now refuse by name, as the shapes with
# two or more terms do on the Gamma side.
function _bsl_data(seed)
    r = Random.Xoshiro(seed); n = 300; G = 20
    gi = rand(r, 1:G, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G)
    eta = 0.3 .+ 0.4 .* x .+ b0[gi] .+ b1[gi] .* z
    p = 1 ./ (1 .+ exp.(-eta))
    s = Float64[sum(rand(r) < pi for _ in 1:10) for pi in p]
    return (; x, z, g = string.("g", gi), s, f = 10.0 .- s)
end

@testset "Binomial single (0 + z | g) and (1 + z | g): bootstrap and default simulate refuse by name" begin
    d = _bsl_data(5)
    cases = [
        ("(0 + z | g)", "a random slope without an intercept"),
        ("(1 + z | g)", "a correlated random intercept and slope"),
    ]
    for (terms, label) in cases
        fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula(cbind(s, f) ~ x + $terms))"))
        fit = drm(fo, Binomial(); data = d)
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
            @test occursin("for Binomial() with $label ($terms)", msg)
            @test occursin("does not carry trials", msg)
        end
    end
end

@testset "Binomial slope refusal leaves the working shapes alone" begin
    d = _bsl_data(5)
    # a single (1 | g) keeps its fresh draw and its bootstrap
    f1 = drm(bf(@formula(cbind(s, f) ~ x + (1 | g))), Binomial(); data = d)
    @test size(bootstrap_summary(f1; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(f1))
end
