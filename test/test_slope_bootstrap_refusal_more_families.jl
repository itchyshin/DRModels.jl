using DRModels, Test, Random

# Beta, Student and LogNormal fits with a random slope, or with two or more random-effect terms, have no
# fresh random-effect draw. `bootstrap_summary`, `bootstrap_result`, `bootstrap_ci` and the default
# `simulate(fit; data = data)` used to fall back to a draw with the random effects at zero, which
# collapses the variance-component interval without a message. They now refuse by name, with the same
# message form as the Gamma and Binomial refusals.
function _sbr_data(seed)
    r = Random.Xoshiro(seed); n = 400; G = 25
    gi = rand(r, 1:G, n); hi = rand(r, 1:6, n); x = randn(r, n); z = randn(r, n)
    b0 = 0.4 .* randn(r, G); b1 = 0.3 .* randn(r, G); c0 = 0.3 .* randn(r, 6)
    eta = 0.3 .+ 0.4 .* x .+ b0[gi] .+ b1[gi] .* z .+ c0[hi]
    mu = 1 ./ (1 .+ exp.(-eta))
    ybeta = clamp.(mu .+ 0.08 .* randn(r, n), 0.02, 0.98)
    ystu = eta .+ 0.3 .* randn(r, n) ./ sqrt.(0.3 .+ rand(r, n))
    ylog = exp.(eta .+ 0.3 .* randn(r, n))
    return (; x, z, g = string.("g", gi), h = string.("h", hi), ybeta, ystu, ylog)
end

const _SBR_FAMILIES = [("Beta", DRModels.Beta(), :ybeta), ("Student", Student(), :ystu),
                       ("LogNormal", LogNormal(), :ylog)]

@testset "Beta, Student, LogNormal slopes: bootstrap and default simulate refuse by name" begin
    d = _sbr_data(7)
    cases = [
        ("(0 + z | g)", "a random slope without an intercept"),
        ("(1 + z | g)", "a correlated random intercept and slope"),
        ("(1 | g) + (0 + z | g)", "an independent random slope with an intercept"),
        ("(1 | g) + (1 | h)", "crossed or nested random intercepts"),
    ]
    for (fname, fam, resp) in _SBR_FAMILIES, (terms, label) in cases
        @testset "$fname $terms" begin
            fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula($resp ~ x + $terms), @formula(sigma ~ 1))"))
            fit = drm(fo, fam; data = d)
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
                @test occursin("for $fname() with $label ($terms)", msg)
                @test occursin("call `simulate(fit; data = data, re = :zero)`", msg)
            end
            # the escape named in the message works: a conditional draw with the random effects at zero
            @test length(simulate(fit; data = d, re = :zero, rng = Random.Xoshiro(1))) == length(d.x)
        end
    end
end

@testset "slope refusal leaves the working shapes alone" begin
    d = _sbr_data(7)
    for (fname, fam, resp) in _SBR_FAMILIES
        # fixed effects only and a single (1 | g) keep their draws and their bootstrap
        for terms in ("", " + (1 | g)")
            fo = Core.eval(@__MODULE__, Meta.parse("bf(@formula($resp ~ x$terms), @formula(sigma ~ 1))"))
            fit = drm(fo, fam; data = d)
            @test length(simulate(fit; data = d, rng = Random.Xoshiro(1))) == length(d.x)
            @test size(bootstrap_summary(fit; data = d, B = 2, rng = Random.Xoshiro(1)), 1) == length(coef(fit))
        end
    end
end
